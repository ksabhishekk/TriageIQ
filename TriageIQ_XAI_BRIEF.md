# TriageIQ — XAI (SHAP / LIME) Project Brief

Purpose of this file: give Claude Code (and humans) full context to add per-prediction explainability to TriageIQ.
Everything in sections 1–6 comes from the repo README and `NLP.md`. Anything marked **[VERIFY]** was NOT visible when this
brief was written (the `api/` folder could not be read), so read the code before relying on it.

---

## 1. The goal

Add explainable AI so that for any scored complaint the system can say **why** it produced that risk score:

1. How much came from **the customer's words** vs the **company's track record**.
2. Which **sentences/phrases** in the complaint pushed the risk up or down.
3. Which **track-record inputs** (of the 19) pushed it up or down, and by how much.
4. Evidence that the explanations are **faithful** (they reflect what the model really does), not decorative.

Non-goals: do not change the model, retrain it, or alter `/predict` outputs. XAI is an add-on.

---

## 2. What TriageIQ is

- **Question answered:** which CFPB consumer complaints will end with the company paying money ("Closed with monetary relief")?
- **Use:** score each new complaint the day it arrives. High risk -> senior analyst. Low risk -> template reply.
  Decides who looks first, not how it is resolved.
- **Live demo:** https://triageiq-mu.vercel.app (API docs at `/docs`). Essay: https://bajaj30.github.io/TriageIQ/
- **Status:** built and live (Oct 2026). A learning project built end to end by hand. Planned-but-unbuilt: similar-complaint retrieval (pgvector).

### Data
- 100% real CFPB Consumer Complaint Database. 4.8M complaints (2022–2024), 4,946 companies.
- 1.6M have a written narrative (1,639,068). 292,940 sampled for training/testing the text model.
- Base rate: ~1 in 80 complaints pays out (2.74% -> 1.71% among narrative complaints, 2022 -> 2024; payouts getting rarer).
- Temporal split: train Jan 2022–Sep 2023, validate Oct–Dec 2023, **test all of 2024** (150,000 complaints, 3,471 payouts).

### The core idea: two readers, one score
- **Track record** (19 inputs, built in SQL): company/product/issue history *as of the complaint's date*, outcomes counted only
  if >= 60 days old, smoothed toward product rate (K = 5), plus trend, volume and cold-start flags.
- **Words**: DistilBERT reads the narrative.
- Fused into one logit -> calibrated probability -> route.

---

## 3. The model (what you will be explaining)

**Fusion network** (`training/fusion_distilbert.ipynb`):
- Text branch: DistilBERT (uncased, ~66M params) -> `[CLS]` vector 768 -> 128.
- Table branch: 19 track-record inputs -> 64. Categorical inputs use **entity embeddings**
  (product 4-dim, sub-product 8, issue 12, state 8). Numeric inputs scaled (`log1p` for days-since-last-complaint, then
  standardised with **train-only** mean/std saved in `preprocessing.json`).
- Join: 128 + 64 = 192 -> 64 -> 1 (logit). GELU, dropout 0.1.

**Tokenisation:** WordPiece, max 512 pieces (510 + `[CLS]`/`[SEP]`, head truncation). Two added special tokens `[DATE]` and
`[REDACTED]` (vocab 30,522 -> 30,524). CFPB redactions (`XXXX`, `XX/XX/XXXX`) are converted to these by the SQL function
`clean_narrative()` (`sql/04_training_set/01_clean_narrative.sql`), used by both training export and the live API.
API requires >= 20 real words.

**Score pipeline at serving:**
1. logit z from the model
2. add case-control prior correction **-2.4165** (= ln 0.0892; undoes the non-payout undersampling)
3. **Platt scaling** sigmoid(a·z + b) (NLP.md quotes a = 0.70, b = -0.76; source of truth is `training/serving/calibrate.py`),
   fitted on Jul–Dec 2024
4. route: **senior if p >= 0.0367** (the riskiest 10%), else template

Both steps 2 and 3 are monotone in z. So explanations should be computed **in logit space**; direction and relative size carry over
to the probability.

**Performance to preserve (2024 test):** within-company ROC-AUC 0.8224 (95% CI 0.815–0.830), within product×issue 0.9492,
pooled 0.9731, PR-AUC 0.4395, riskiest 10% catches 92.6% of payouts. Cheaper baselines exist for sanity checks:
TF-IDF + logistic regression (text+SQL) 0.8115 within company.

---

## 4. Serving stack and the constraints it imposes

```
visitor -> Vercel static site (web/) --forwards API paths + secret header--> NGINX on AWS EC2 (403 w/o secret, 5 req/s/visitor)
  -> FastAPI (api/)
       |- Postgres `serving` schema: 19 track-record inputs as of 2025-01-01, clean_narrative(), 150,000 real 2024 complaints for demo
       '- ONNX Runtime (CPU): DistilBERT + 19 inputs -> logit -> Platt -> senior/template
```

- **No PyTorch at serving time** (ONNX Runtime only). Gradient methods (Captum / Integrated Gradients) are NOT available in the API.
  Use them offline only (laptop / Kaggle).
- Server: **EC2 t4g.small, ARM, 2 vCPU, ~1.8 GB RAM**, api container already ~550 MiB. ~$21/month budget.
- Latency today: ~0.2 s typical, ~0.8 s p95 per complaint on the server (0.02 / 0.08 s on an M4 laptop). The model is almost all of it;
  feature lookup < 1 ms.
- Endpoints known from README: `POST /predict`, `GET /complaint/random`, `GET /complaint/{id}`, `/options/*`, `/model-info`, `/health`.
- Model bundle lives in `training/outputs/serving_v3/` and is **not in git** (too large). Ask the user for it or for the path.
- Frontend is plain HTML/CSS/JS (`web/`). Deployment: `deploy/` (Docker Compose db + api + NGINX).

Implications: keep heavy libs (full `shap`, `numba`) out of the API image unless proven to fit in RAM and build on ARM. Prefer a small
hand-written Shapley/KernelSHAP estimator in numpy for serving, and use the real `shap` and `lime` packages offline for validation.

---

## 5. Current explainability status (honest audit)

Searched README and `NLP.md`: **no SHAP, LIME, attention maps or Integrated Gradients.** What partly exists:
- TF-IDF + logistic regression baseline: coefficients are interpretable, but it is not the shipped model.
- The 19 inputs have names/definitions (`sql/03_features/feature_dictionary.md`) but no per-prediction contribution.
- Similar-complaint retrieval (planned, not built) would be example-based explanation.
- `/predict` returns a probability and route. **[VERIFY]** whether it returns anything else.

---

## 6. Repo conventions you MUST follow (from the README design rules)

- **No leakage:** every input computable at complaint receipt. Do not touch `complaint_events` outcomes as features.
- **One source of truth for features:** all feature logic is in SQL. Explanations must reuse `serving.model_input()` and
  `clean_narrative()`. No re-implementing feature logic in Python.
- **Reproducible:** SEED = 42; every quoted number has the code that produced it; numbers are recorded in `Context/FACTS.md`;
  each SQL/training folder keeps a `log.md`.
- **Parity discipline:** the repo proves ONNX matches PyTorch (1e-5; Spearman 0.999999) and that serving features match training
  (19/19 identical). Do the same for anything new.
- Read `CLAUDE.md` in the repo root first; it may add rules not listed here.

---

## 7. The XAI design

### 7.1 Key insight: cache the encoder, explain with the cheap head
The expensive part is DistilBERT (~0.2 s). The fusion head (192 -> 64 -> 1) is tiny. So: run the encoder once, cache the `[CLS]`
vector, then evaluate the head hundreds of times for free.

**[VERIFY]** how the current ONNX graph is built. If it is a single graph (ids, mask, 19 inputs -> logit) and does not expose the
`[CLS]`/128-d text vector, export two graphs: (a) encoder -> text vector, (b) head: text vector + tabular -> logit.
Prove parity against the existing single graph before using them (same bar as the repo: ~1e-5 on logits, Spearman ~1.0).

### 7.2 Layer A — Track-record attribution (SHAP, cheap)
- Players: the 19 inputs, with each categorical group (product, sub-product, issue, state) as **one** player.
- Text vector held fixed at the complaint's real vector.
- Background: ~100 rows sampled from training/serving data (fixed seed). Baseline value = mean logit over background.
- Output: signed contribution per input in logit units. Efficiency check: base value + sum of contributions ≈ model logit.
- With the head this cheap, exact or near-exact Shapley is feasible for ~19 players with a sampling estimator.

### 7.3 Layer B — Words vs track record (the headline bar)
- Text effect = logit(real text vector, real table) - logit(baseline text vector, real table), where baseline text vector =
  **mean text vector over the background sample** (not "empty text").
- Table effect = same idea with the table swapped for background.
- Present as one stacked bar: "words contributed X, track record contributed Y". This lines up with the project's own finding that
  inside one company, words matter more than history.

### 7.4 Layer C — Sentence-level text attribution (SHAP primary, LIME cross-check)
- Word-level perturbation needs hundreds of encoder passes (0.2 s each on the server) -> too slow. Use **sentences/clauses as
  units** (about 8–15 per complaint). Treat `[REDACTED]` / `[DATE]` runs as their own units.
- Masking: remove the unit's text (or replace with `[MASK]`/nothing; document the choice and test sensitivity to it).
- Estimator: partition-style / sampled Shapley over units, batched through ONNX, with a hard cap on coalitions
  (start ~64–128; tune with a convergence check).
- LIME: same units, exponential-kernel weighted linear surrogate, for cross-checking only.
- Live endpoint `POST /explain` (separate from `/predict`, triggered by an "Explain" button) so `/predict` latency does not change.
  Define a latency budget and enforce it with the sample cap (proposal: p95 under ~5 s on the 2-vCPU server; confirm with user).
- Alternative for the public demo: **precompute** explanations offline for the demo complaints and store them in the `serving` schema,
  and keep live `/explain` for custom text. Decide with the user.

### 7.5 Offline research layer (laptop/Kaggle, not in the API)
- **Global SHAP** over a few thousand 2024 test rows: which inputs and which phrases matter overall; summary plots.
- **Sanity check** against TF-IDF + LR coefficients (do top phrases overlap?).
- **Integrated Gradients** (Captum, PyTorch) as an independent third view on tokens.
- **Compare LIME vs SHAP:** top-3 sentence agreement rate, rank correlation, stability across seeds/runs.

### 7.6 Faithfulness evaluation (this is what makes it credible)
1. **Deletion test:** remove top-k attributed sentences, measure logit drop; compare with removing k random sentences. Report across
   a few hundred 2024 test complaints.
2. **Insertion test:** the mirror.
3. **Additivity:** Layer A and B satisfy the efficiency property (check numerically); report max error.
4. **Stability:** rerun with different seeds/background samples; report variance.
5. Record every number in `Context/FACTS.md` and add a `log.md`, per repo convention.

### 7.7 UI
In `web/`: highlight sentences by signed contribution (red = raises risk, blue = lowers), a bar chart of the top track-record
contributions, and the words-vs-history bar. Always show this caveat:
> Explains the model, not the cause. The repo's own error analysis says the deciding fact is often a company decision that is not in the text.

---

## 8. Suggested repo layout

```
training/explain/            offline: global SHAP, IG, LIME-vs-SHAP, faithfulness tests (+ log.md)
training/serving/            export of split ONNX graphs (encoder / head) + parity test
api/                         POST /explain, small Shapley estimator, tests
web/                         explanation UI
Context/FACTS.md             record the new measured numbers
```

---

## 9. Work plan for Claude Code (in order)

0. **Read first:** `CLAUDE.md`, `NLP.md`, `Context/FACTS.md`, `Context/TriageIQ.md`, `sql/03_features/feature_dictionary.md`,
   everything in `api/`, `training/serving/export_onnx.py`, `training/serving/calibrate.py`, `training/fusion_distilbert.ipynb`.
   Write a short note of what `/predict` returns and how the ONNX session is built (resolves the **[VERIFY]** items).
1. **Split-graph export + parity test** (7.1). Gate: matches existing model to ~1e-5.
2. **Layer A + B** (tabular SHAP, words-vs-history) in Python with unit tests (additivity). Gate: efficiency error tiny.
3. **Layer C** sentence segmentation + masking + sampled Shapley; LIME cross-check offline. Gate: converges, within latency cap.
4. **Faithfulness + global analysis notebook** (7.5, 7.6). Record numbers.
5. **`/explain` endpoint** + regression test that `/predict` output and latency are unchanged.
6. **UI** and the caveat banner.
7. **Docs:** update README (a short XAI section with honest limits), `NLP.md` (SHAP, LIME, Shapley values, masking, faithfulness
   metrics), `Context/FACTS.md`.

Ask the user before: adding heavy dependencies to the API image, changing the deployed server size, or changing any existing
`/predict` behaviour.

---

## 10. Pitfalls to watch

- **Explaining probability vs logit:** do logit space; convert narrative to "raises/lowers risk".
- **Baseline choice drives SHAP values.** Document it; a poor text baseline gives misleading "words" effects.
- **Masking artefacts:** deleting a sentence creates off-distribution text; test two masking strategies.
- **Redaction tokens** carry signal (70.3% of narratives contain XXXX); don't strip them from explanations.
- **Truncation:** 8.4% of 2024 test complaints are cut at 512 pieces; sentences beyond the cut have zero effect. Flag this in the UI.
- **Duplicate/template texts** (copy-paste floods) will give identical explanations; fine, but note it.
- **ARM + RAM:** verify any new wheel builds on aarch64 and that the container stays well under the 1.8 GB limit.
- **Correlated features:** SHAP splits credit among correlated track-record inputs (company vs company×issue rates); say so, don't over-read single bars.
- **Do not claim causality.** Attributions describe the model's behaviour.

---

## 11. Paste-ready prompt to start Claude Code

> Read `XAI_BRIEF.md` (this file), then `CLAUDE.md`, `NLP.md`, and everything under `api/` and `training/serving/`. Don't write code yet.
> First report: (1) what `/predict` returns, (2) how the ONNX graph is structured and whether the text `[CLS]`/128-d vector is
> accessible, (3) where `clean_narrative()` and `serving.model_input()` are called, (4) any conflicts between this brief and the code.
> Then propose the concrete plan for steps 1–2 and wait for my approval.
