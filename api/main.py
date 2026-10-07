"""TriageIQ API — score a CFPB complaint: will the company pay? -> senior analyst or template response.

Run on the Mac (from the repo root):  sh api/run_local.sh     then open http://localhost:8000/docs
(it reads the database password from .env; the API itself reads the standard PG* environment variables)

Design (Learning/Phase3/directions.md, step 3.4):
  - The caller sends NAMES + TEXT, never features. Features come from SQL (serving.model_input), the same
    formulas training used, skew-tested — one source of truth (ground rule 4). Trade-off: every request needs
    a database round trip, and the API is down when the database is.
  - The model, tokenizer and database pool are loaded ONCE at startup (lifespan), not per request.
  - Endpoints are plain `def`, not `async def`: scoring keeps the CPU busy (no waiting to hand over), so
    FastAPI runs them on its thread pool and one slow request doesn't freeze the others.
"""
import math, os, time
from threading import Lock
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from api import db
from api.schemas import ComplaintIn, ComplaintOut, ExplanationOut, Prediction, XAIStatus
from api.scorer import Scorer
from api.xai import Explainer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "training/outputs/serving_v3"))
THREADS = int(os.environ.get("ORT_THREADS", "2"))
XAI_ENABLED = os.environ.get("XAI_ENABLED", "0") == "1"
XAI_LOCK = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = Scorer(MODEL_DIR, THREADS)          # ~1 s: the 266 MB model is read once
    app.state.explainer = None
    db.pool.open(wait=True, timeout=30)
    with db.pool.connection() as conn:
        app.state.as_of = db.snapshot_day(conn)
    yield
    db.pool.close()


app = FastAPI(
    title="TriageIQ",
    version="v3",
    lifespan=lifespan,
    description=(
        "Predicts whether a CFPB consumer complaint will end with the company paying money, so a compliance desk "
        "can send the riskiest complaints to a senior analyst.\n\n"
        "**Try it:** `GET /complaint/random` shows a real 2024 complaint, the model's call and what actually "
        "happened. `POST /predict` scores a complaint you write (an example is filled in).\n\n"
        "A DistilBERT reads the text; 19 track-record numbers (computed in SQL, as of the end of 2024) describe "
        "the company and the kind of problem. Tested on 150,000 complaints from 2024. Details: `GET /model-info`."),
)


def _route(senior: bool) -> str:
    return "senior analyst" if senior else "template response"


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.post("/predict", response_model=Prediction, tags=["score"])
def predict(c: ComplaintIn):
    """Score a NEW complaint: names + text in, chance of a payout and a route out."""
    t0 = time.perf_counter()
    with db.pool.connection() as conn:
        try:
            ids, notes = db.resolve(conn, c.company, c.product, c.sub_product, c.issue, c.state)
        except db.NotFound as e:
            raise HTTPException(status_code=422, detail=str(e))
        inputs = db.model_input(conn, ids, c.older_american, c.servicemember)
        text = db.clean(conn, c.narrative)
    s = app.state.scorer.score(text, inputs)
    return Prediction(
        payout_probability=round(s["probability"], 5), route=_route(s["senior"]),
        senior_threshold=round(app.state.scorer.cal["senior_threshold_p"], 5),
        model_version=app.state.scorer.card["model_version"], features_as_of=app.state.as_of,
        inputs={k: round(float(v), 6) for k, v in inputs.items()},
        text_word_pieces=s["word_pieces"], text_cut_at_512=s["cut_at_512"], notes=notes,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.get("/xai/status", response_model=XAIStatus, tags=["explain"])
def xai_status():
    """Whether optional explanations are enabled and their runtime artifacts are present."""
    if not XAI_ENABLED:
        return XAIStatus(enabled=False, reason="Explanations are disabled pending split-graph parity and resource checks.")
    missing = [
        filename for filename in ("xai_encoder.onnx", "xai_head.onnx", "xai_background.npz")
        if not (MODEL_DIR / filename).is_file()
    ]
    if missing:
        return XAIStatus(enabled=False, reason=f"XAI artifacts are missing: {', '.join(missing)}")
    return XAIStatus(
        enabled=True,
        reason="XAI is enabled by the operator; runtime parity and additivity checks run for each explanation.",
    )


@app.post("/explain", response_model=ExplanationOut, tags=["explain"])
def explain(c: ComplaintIn):
    """Explain one score on demand; this deliberately stays separate from /predict."""
    if not XAI_ENABLED:
        raise HTTPException(status_code=503, detail="Explanations are disabled until XAI parity and resource checks pass.")
    explainer = app.state.explainer
    if explainer is None:
        with XAI_LOCK:
            explainer = app.state.explainer
            if explainer is None:
                try:
                    explainer = Explainer(MODEL_DIR, app.state.scorer, THREADS)
                except FileNotFoundError as e:
                    raise HTTPException(status_code=503, detail=str(e))
                app.state.explainer = explainer

    t0 = time.perf_counter()
    with db.pool.connection() as conn:
        try:
            ids, _ = db.resolve(conn, c.company, c.product, c.sub_product, c.issue, c.state)
        except db.NotFound as e:
            raise HTTPException(status_code=422, detail=str(e))
        inputs = db.model_input(conn, ids, c.older_american, c.servicemember)
        text = db.clean(conn, c.narrative)

    score = app.state.scorer.score(text, inputs)
    result = explainer.explain(text, inputs)
    logit_delta = abs(result["modality"]["current_logit"] -
                      result["modality"]["baseline_logit"] -
                      result["modality"]["text_logit_contribution"] -
                      result["modality"]["track_record_logit_contribution"])
    if logit_delta > 1e-5:
        raise HTTPException(status_code=503, detail="XAI modality additivity check failed.")
    if result["feature_additivity_error"] > 1e-5 or result["sentence_additivity_error"] > 1e-5:
        raise HTTPException(status_code=503, detail="XAI attribution additivity check failed.")

    calibrated = 1.0 / (1.0 + math.exp(-(
        app.state.scorer.cal["a"] *
        (result["modality"]["current_logit"] + app.state.scorer.cal["offset"]) +
        app.state.scorer.cal["b"]
    )))
    if abs(calibrated - score["probability"]) > 1e-4:
        raise HTTPException(status_code=503, detail="XAI and production model scores do not match.")

    return ExplanationOut(
        payout_probability=round(score["probability"], 5),
        route=_route(score["senior"]),
        model_version=app.state.scorer.card["model_version"],
        features_as_of=app.state.as_of,
        text_word_pieces=result["text_word_pieces"],
        text_cut_at_512=result["text_cut_at_512"],
        modality=result["modality"],
        features=result["features"],
        feature_baseline_logit=result["feature_baseline_logit"],
        feature_additivity_error=result["feature_additivity_error"],
        sentences=result["sentences"],
        sentence_additivity_error=result["sentence_additivity_error"],
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.get("/complaint/random", response_model=ComplaintOut, tags=["demo"])
def random_complaint(paid: bool | None = Query(default=None, description="true = one the company paid; "
                                                "false = one it didn't; empty = any")):
    """A random real complaint from 2024 (the test set the model never saw): its score vs what happened."""
    with db.pool.connection() as conn:
        cid = db.random_test_complaint(conn, paid)
    return complaint(cid)


@app.get("/complaint/{complaint_id}", response_model=ComplaintOut, tags=["demo"])
def complaint(complaint_id: int):
    """One real 2024 complaint by its CFPB id: the model's call (with the track record as of the day it arrived)
    next to what the company actually did."""
    t0 = time.perf_counter()
    with db.pool.connection() as conn:
        r = db.test_complaint(conn, complaint_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"complaint {complaint_id} is not in the 2024 demo set "
                                                     "(try GET /complaint/random)")
    inputs = {k: r[k] for k in db.INPUTS}
    s = app.state.scorer.score(r["text"], inputs)
    return ComplaintOut(
        payout_probability=round(s["probability"], 5), route=_route(s["senior"]),
        senior_threshold=round(app.state.scorer.cal["senior_threshold_p"], 5),
        model_version=app.state.scorer.card["model_version"], features_as_of=r["date_received"],
        inputs={k: round(float(v), 6) for k, v in inputs.items()},
        text_word_pieces=s["word_pieces"], text_cut_at_512=s["cut_at_512"], notes=[],
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        complaint_id=r["complaint_id"], date_received=r["date_received"], company=r["company_name"],
        product=r["product_name"], sub_product=r["sub_product_name"], issue=r["issue_name"], state=r["state_code"],
        narrative=r["text"], what_actually_happened=r["company_response"], actually_paid=bool(r["paid"]))


@app.get("/options/{what}", tags=["options"])
def options(what: str, search: str | None = Query(default=None, min_length=2, max_length=100,
                                                   description="companies only: part of the name, e.g. 'wells'")):
    """Valid names for /predict: `products` (with sub-products), `issues`, `states`, `companies?search=...`."""
    if what not in ("products", "issues", "states", "companies"):
        raise HTTPException(status_code=404, detail="use products, issues, states or companies")
    if what == "companies" and not search:
        raise HTTPException(status_code=422, detail="add ?search= with part of the company name")
    with db.pool.connection() as conn:
        return db.options(conn, what, search)


@app.get("/health", tags=["service"])
def health():
    """Is the service up, the model loaded, and the database reachable?"""
    try:
        with db.pool.connection() as conn:
            day = db.snapshot_day(conn)
        return {"status": "ok", "model_loaded": True, "database": "ok", "features_as_of": day}
    except Exception as e:                                  # report, don't crash: health must always answer
        raise HTTPException(status_code=503, detail=f"database unavailable: {type(e).__name__}")


@app.get("/model-info", tags=["service"])
def model_info():
    """What the model is, how it was tested, how probabilities are calibrated, and its limits."""
    return app.state.scorer.card | {"features_as_of_live": str(app.state.as_of)}
