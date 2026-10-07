# TriageIQ — Agent Context

> **Read order for a fresh session:** §0 below (where we are, how to resume) → the rest of this file →
> `Context/FACTS.md` (every number) → `Learning/Phase3/revision.md` (how the deployed system works) →
> `Context/TriageIQ.md` (full spec, when needed).
> **`Context/FACTS.md` is the only source of numbers.** If this file disagrees with it, FACTS.md wins.
> Regenerate it with `python training/canonical_facts.py`.
> **Keep this file updated as work progresses** — it is the handoff artifact between sessions.

Last updated: 2026-10-07 · **core phases built and live; optional XAI implementation gated off** · model = DistilBERT fusion v3 (19 inputs), Platt-calibrated ·
site on Vercel, API on AWS EC2 (Sydney) · blog on GitHub Pages · remaining work is optional (§0)

---

## 0. Where we are — read this first (2026-10-07)

**State: built, deployed, written up.** Phases 0–3 are done; nothing is half-finished. What's left is optional.

**Optional XAI (2026-10-07):** source for split ONNX graphs, an isolated `/explain` endpoint, status route, and an opt-in
website control are now added. `XAI_ENABLED` defaults to `0`; `/predict` and the live site remain unchanged. This checkout
does not contain the ignored model bundle, training checkpoint, or training parquet, so the export/parity check and API
resource checks have **not** run. Do not enable or deploy XAI yet. On the Mac, restore those existing artifacts, run
`python training/serving/export_xai_onnx.py`, then the unit/offline validation; only consider enabling it after parity,
faithfulness, latency, and RAM checks pass. No XAI measurements belong in `Context/FACTS.md` until measured.

| what | where |
|---|---|
| Live demo (score a complaint · real 2024 complaints · how it works) | https://triageiq-mu.vercel.app · API docs `/docs` on the same address |
| The essay "Finding the two percent" | https://bajaj30.github.io/TriageIQ/ (`docs/`) |
| Code | https://github.com/Bajaj30/TriageIQ (branch `main`) |
| Headline (2024 test, 150,000 complaints) | within-company **0.8224** (0.815–0.830) · riskiest 10% → **92.6%** of payouts |
| Running cost | ≈ $21/month from $140 AWS credits; plan expires **2027-04-02** (§12) |

**Request path:** visitor → Vercel (static site + `routes` forwarding API paths with a secret header) → NGINX on
EC2 `3.106.107.237` (403 without the secret; 5 req/s per visitor) → FastAPI (`api/`) → Postgres `serving` schema
(19 inputs via `serving.model_input`, text via `serving.clean_narrative`) + ONNX DistilBERT fusion → Platt → route.

**Not in git — where it lives:**
- **Model bundle** `training/outputs/serving_v3/` (model.onnx 266 MB, tokenizer.json, preprocessing.json,
  calibration.json, model_card.json, serving_db.dump 57 MB): **only on the Mac and the server** (`~/triageiq/…`).
  Training weights `model.pt`: Mac (`training/outputs/fusion_distilbert_full/`) + the saved Kaggle notebook version.
  ⚠️ No third copy — worth putting the bundle in a private Kaggle dataset or a drive.
- **Full database (16 GB):** the Docker volume of the root `docker-compose.yml` on the Mac. Rebuildable from
  `Data/complaints.csv` by running `sql/` in number order (README "Run it yourself").
- **Secrets:** root `.env` (local DB) · `deploy/.env` on the server (DB password + ORIGIN_SECRET) · Vercel env
  `ORIGIN_SECRET` · `~/.ssh/triageiq-key.pem`. Never print or commit any of them.

**Resume — the commands:**
| to | run |
|---|---|
| start the full local DB | `docker compose up -d` (port 5433) |
| run the API on the Mac | `sh api/run_local.sh` → http://localhost:8000/docs (needs the local DB) |
| run the shipped stack locally | `docker compose -f deploy/docker-compose.yml up -d --build` → localhost:8000/docs |
| update the server | `sh deploy/push.sh 3.106.107.237` (SSH only from Shivam's IP — if refused, his ISP IP changed: re-authorise port 22 in security group `sg-065abe94829601344`) |
| server shell | `ssh -i ~/.ssh/triageiq-key.pem ubuntu@3.106.107.237` (docker needs `sudo`) |
| update the website | `cd web && npx vercel deploy --prod --yes` |
| update the blog | edit `docs/`, push to `main` (GitHub Pages rebuilds in ~1 min) |
| AWS CLI | `aws login`, then `export AWS_PROFILE=triageiq` (session expires — "session expired" = log in again) |

**Open — all optional, in order of value:**
1. **60–90 s screen recording** of the demo for README / resume (step 3.10).
2. **Read-only DB role** for the API (step 3.7) — today it connects as the DB owner. Low risk (DB not reachable
   from outside, SQL is parameterised) but a cheap hardening.
3. **Minimal CI** — ruff + pytest + `docker build` on push (step 3.8).
4. **Monitoring** — a `prediction_log` table + one drift query; "replay 2024" demo (step 3.9).
5. **v5 retrain through 2025** — after exams; plan at the end of §12.
6. **"5 most similar past complaints"** (pgvector) — in the original design, never built (README marks it).
7. Hygiene: scikit-learn undeclared in `pyproject.toml` (trap 8); audit items L1/L4/L5; FACTS.md's "F3" frame
   row still describes the v2 parquet (301,460) — generated file: fix in `training/canonical_facts.py`, then regenerate.
8. **Finish XAI validation** (optional): restore the Mac-only model/training artifacts, run the split-graph parity check,
   run `python -m unittest api.test_xai`, install `training/explain/requirements.txt` in the research environment,
   run `python training/explain/validate_xai.py` and `python training/explain/integrated_gradients.py`, then measure
   latency and RAM on the server. Only then decide whether to enable it by setting `XAI_ENABLED=1` both in the deploy
   environment and for `deploy/push.sh` (which transfers the optional artifacts only when enabled). It is currently
   gated off; details in `training/serving/log.md`.
9. **Teardown** at the end (by 2027-04-02): delete every resource in §12, then release the Elastic IP.

Learning side (Shivam): courses + the DMLS book after exams (`Learning/README.md`); interview prep from
`Learning/Phase0–3/revision.md` + `Context/interview.md`.

---

## 1. What this project is

**Predict whether an incoming CFPB consumer complaint will cost the company money**, so a compliance
desk can staff senior analysts against a 15-day regulatory response deadline.

```
complaint arrives → P(monetary relief) → high: senior analyst / low: template response
```

- **PostgreSQL** holds 4.8M real complaints and computes point-in-time company / issue features.
- **A fine-tuned DistilBERT** reads the complaint narrative.
- **A fusion model + FastAPI** (ONNX on CPU, AWS EC2, behind NGINX; website on Vercel) combines both, reading
  features computed by the same SQL formulas training used (skew-tested).

**The thesis, measured on training set v3** (test = 150,000 complaints from 2024, 2.31% payouts;
TF-IDF + logistic regression; `training/test.ipynb`; FACTS.md "Baselines on training set v3"):

| | within (Product×Issue) | **within-company** |
|---|---|---|
| text only | 0.8916 | **0.7956** |
| features only (19 SQL inputs) | 0.9276 | 0.7500 |
| fusion | **0.9457** | **0.8115** |
| **DistilBERT fusion** (Kaggle, 3 epochs) | **0.9492** | **0.8224** (0.815–0.830) |
| DistilBERT text only | 0.9163 | 0.8036 (0.796–0.812) |

*(v2, historical: 0.8905 / 0.9076 / 0.9330 and 0.7900 / 0.7463 / 0.8034 — a different test set.)*

Neither modality subsumes the other. Note the flip: holding product and issue fixed, metadata wins;
**inside one company's queue — the deployment view — text wins.** Never quote one frame as if it
were the other.

---

## 2. How to work with Shivam — read before responding

**Learning project, not a delivery contract.** He does a sub-task, reports back, gets reviewed,
gets the next step.

**Style**
- **Small steps.** One decision or concept at a time, so he can hold the context himself.
- **Crisp, simple language, no story-type responses.** Tie each step back to basics and to the core
  goal above.
- **Do not circle.** When he states a requirement, map every decision to it directly.
- **Teaching (his request, 2026-10-03):** intuition first (an everyday comparison), then the **complete**
  concept even if the project uses only part of it (e.g. all of Docker, not just Compose), then the same
  idea implemented in TriageIQ, then he explains it back. Phase 3 follows `Learning/Phase3/directions.md`.
  **Prefers video lectures over blogs/articles** when pointing him to resources.
- **Resume from the docs, not the chat (Shivam, 2026-10-06).** He works in multi-day gaps and must be able to pick the
  project up from the .md files alone. **At the end of every working session, update §0** (state, what changed,
  next steps) and any doc the work touched, then commit + push. A fresh session starts by reading §0.
- **No new .md files unless he asks for one (2026-10-03).** Answer in chat, or update an existing file
  (CLAUDE.md, blog_log, logs) as standing duties require. Never create a doc just to hold an answer.

**Rigor**
- **Ask when in doubt; never assume.** If an instruction is ambiguous, ask — or state the
  interpretation explicitly before acting on it.
- **Every number states its frame** (F1 / F2 / F3 — see `FACTS.md`). Numbers come from `FACTS.md`
  or a fresh query, **never from memory**. Quoting stats without a frame caused a whole round of
  doc corrections and a re-evaluation of the entire project.
- **Verify before agreeing.** Check the data before accepting a design assumption — e.g. "each
  sub-product has one parent" was false for 87.5% of rows.
- **Never claim an action you haven't verified** (a file saved, a commit made). Both happened once.
- **Correct the reasoning, not just the conclusion.** If he reaches the right answer for the wrong
  reason (e.g. "surrogate keys are more readable"), say so — wrong reasons resurface later.

**Boundaries**
- **Division of labour (current, agreed 2026-09-25):** I write infrastructure, DDL and the `02_load/`
  queries — **one file per "go N"**, with reasoning in the SQL comments. He reviews each before the
  next. Chat replies stay short and only add what the comments don't say.
- **From Phase 0.5 on (agreed 2026-09-28): back and forth.** He writes some queries himself (review them
  properly — correct the reasoning, not just the SQL), I write others on request. He picks per file.
- Analysis / verification / profiling code on his behalf is fine and expected.
- **Blog log (standing duty):** he is writing a blog about the project. Whenever something blog-worthy
  happens — an idea discarded (and why), a realisation, textbook theory used in practice, a SQL
  technique — add a line to `Context/blog_log.md`, with its frame on every number.
  **Write it for readers with little tech knowledge** (his standing rule, also for the README): plain
  words, everyday comparisons, intuitive and interesting; jargon only as a small *(tech: …)* tag;
  frames described in words ("all complaints", not "F1").
  **Deliverable DONE (2026-10-05): the blog is LIVE at https://bajaj30.github.io/TriageIQ/** — "Finding the two
  percent", `docs/` (index.html · blog.css · blog.js · data.js · .nojekyll), GitHub Pages from `main` /docs (legacy
  build, auto-rebuilds on push). 12 chapters, 13 figures; interactive: waffle (2 in 100), the **snowflake** (click a
  table), the 60-day timeline (illustration), the smoothing-K slider + measured K bars, real word-pieces of complaint
  11216352 (raw 92 → clean 80), the read-X%-catch-Y% curve (from the 2024 test scores). `data.js` is GENERATED from
  project files (recall curve, tokens) — regenerate, don't hand-edit. Same look as the demo site.
  **Dark theme (2026-10-05):** every colour is a CSS variable in `blog.css` `:root`, redefined under `html.dark`
  (Shivam's toggle button: follows the OS until clicked, choice kept in localStorage). No darkreader-lock any more —
  Dark Reader re-colours the page, verified readable (it once made the snowflake's SVG text invisible). **Never hard-code a colour in docs/, and never
  name a colour in a caption that flips between themes** ("black box"). Text in figures = HTML, not SVG `<text>`. Add new chapters by
  writing them in blog_log.md first. (The original requirement:) an interactive
  **snowflake-schema visual**. Diagrams live ONLY in `README.md` (§5 = snowflake, journey,
  track-record steps); `blog_log.md` §8 lists what to reuse. Never duplicate a diagram.
- **Learning folder (standing duty, 2026-10-03):** when a phase finishes, write its `Learning/PhaseN/revision.md`
  (same card format); fill README's 2-minute story step 7 after Phase 3. **The gap rule:** when he reports a
  term a Phase 3 step assumed but didn't teach, explain it, then add it to that step's **Assumes** (or to P1–P11).
  Resources: video lectures first; text only where no good lecture exists — say so.
- He values honesty about limitations over polish; diagnosing a flaw is an explicit project goal.

**Machine:** MacBook M4, 16GB. Mac does SQL, data prep, and 1k-row training smoke tests on MPS.
**All real training runs go to Kaggle free tier (T4).**

**Git:** remote `git@github.com:Bajaj30/TriageIQ.git`, branch `main`. Commit + push after doc/decision
updates. Personal documents (`Context/*.docx`, `*.pages`) are gitignored — never commit them.

---

## 3. Ground rules — never violate without writing down why

1. **Sequencing:** SQL now, MLOps later. Each phase ends in a demo-able artifact.
2. ~~No JavaScript, ever.~~ **LIFTED by Shivam 2026-10-05** — the rule existed only because he doesn't write
   frontend code; Claude builds it. Frontend = a static JS site on **Vercel** (free Hobby plan) that reaches the API
   through Vercel **rewrites** (same https address, no CORS / mixed content). Swagger UI stays (`/docs`).
3. **Free tier or student laptop only.** No GPU bills.
4. **One source of truth for features.** All feature logic in SQL. Python never re-implements a
   feature. The central architectural claim.
5. **No leakage.** Every feature computable only from what existed at complaint receipt.
6. **Reproducible.** SEED=42, versioned snapshots, logged configs, code for every quoted number.

---

## 4. Repo map

```
CLAUDE.md                     this file
README.md                     public, plain-language (non-technical reader), Mermaid diagrams.
                              **Fill its ⏳ placeholders as phases finish; numbers must match FACTS.md.**
Context/FACTS.md              every number, three frames — generated, never hand-edit
sql/03_features/feature_dictionary.md   every feature: meaning, frame, model input yes/no and why
Context/metrics.md            every metric: what it answers, how computed, which code, why (not) used — no numbers
Context/schema_explanation.md Phase 0.3 decisions, each tied to a concept
Context/TriageIQ.md           full engineering spec (bible v2); §1.4a = verified window-frame rules
Context/WHAT_WHY.md           pitch and positioning
Context/interview.md          per-phase: what broke, how it was fixed
Context/blog_log.md           blog raw material: discarded ideas, realisations, theory used, SQL — keep adding
Context/audit.md              prompt for an independent audit
Context/audit_findings_2026-09-12.md   audit results + resolution status
Context/old_context/          v1 archives (synthetic-customer design) — do not delete
Learning/README.md            index: directions (roadmaps) vs revision (interview prep) · reading order · 2-minute story
Learning/Phase0/              directions.md · learning_log.md (his notes, 0.1–0.3) · revision.md
Learning/Phase1/revision.md   features: concept cards (idea · in TriageIQ · Q→A · lecture) + what broke + numbers
Learning/Phase2/revision.md   model: same format
Learning/book_DMLS.md         how to read Chip Huyen's DMLS: per-chapter loop (survey → section → recall → TriageIQ table → act → review), chapter order tied to Phase 3 steps
Learning/Phase3/directions.md roadmap: prerequisites P1–P11, lecture map, schedule; per step Assumes → Watch →
                              Learn → Build (+ Bridge to the course) → Explain back → Done when
Data/EDA.ipynb                profiling + v2 training-set build (Cell 8)
Data/docs/data_profile.md     dataset profile
Data/complaints.csv           9.2GB raw — gitignored
Data/data/interim/            meta.parquet · narratives.parquet · triageiq_training_v2.parquet — gitignored
training/verify_ablation.py   reproduces every baseline number
training/export_training_set.py   copies v_training_export → triageiq_training_v3.parquet + manifest (moves rows only)
                              NOTE: FACTS.md 'F3' and all baselines are still the v2 artifact — re-measure on v3
training/KAGGLE_SETUP.md      upload v3 Parquet + manifest as a private Kaggle Dataset; GPU notebook; Cell 1 checks sha256 + counts vs manifest, confirms T4
training/fusion_distilbert.ipynb   Phase 2 model. Switches in cell 1: RUN = mac/smoke/full · MODE = fusion/text/
                              features · MODEL = distilbert-base-uncased / microsoft/deberta-v3-base. On the Mac
                              set them as env vars (RUN=full MODE=features …). Output folder <mode>_<model>_<run>. Saves model.pt, preprocessing.json, metrics.json, test predictions.
                              Mac: run with USE_TF=0 (conda env's TensorFlow is broken); test pass on CPU.
training/serving/             model packaging for the API: calibrate.py (Platt + senior threshold) · export_onnx.py
                              (→ training/outputs/serving_v3/: model.onnx, tokenizer.json, preprocessing.json, calibration.json)
training/test.ipynb           v3 baselines (TF-IDF + LR) — the bar: within-company 0.8115
training/evaluate_fusion.py   training/outputs/<run>/ → training/results/<run>_v3.json (tracked) → FACTS.md ablation table
training/canonical_facts.py   regenerates FACTS.md, canonical_facts.json AND sql/02_load/00_expected_facts.sql
                              (generated — never hand-edit). It re-implements the load rules in pandas; the
                              SQL load must match it (11_validate reads expected('key')). Change a crosswalk
                              rule in BOTH places, or validation fails — on purpose.
docker-compose.yml            Postgres 16 + pgvector, host port **5433** (Postgres.app owns 5432)
sql/                          numbered SQL pipeline, run in pgAdmin — every folder has a log.md
                              (1–2 lines per file). **Update the log.md whenever a file is added or done.**
```

sql/06_serving/ = the serving schema (snapshot, model_input(), lists, demo complaints) → pg_dump.
api/ = FastAPI app (main.py endpoints · scorer.py model · db.py queries · schemas.py) · requirements.txt · run_local.sh.
deploy/ = api.Dockerfile · docker-compose.yml (db + api + nginx[profile public]) · nginx/default.conf.template (envsubst:
          secret-header check + per-visitor rate limit) · initdb/01_restore.sh · server_setup.sh · push.sh <ip> · .env.example.
web/ = the website on Vercel: index.html (score) · real.html · how.html · styles.css · app.js · vercel.json (routes +
          secret header) · dev_server.py (local stand-in for Vercel).
docs/ = the blog on GitHub Pages: index.html · blog.css · blog.js · data.js (generated) · .nojekyll.
training/serving/ = calibrate.py · export_onnx.py · model_card.py → training/outputs/serving_v3/ (git-ignored, see §0).
frontend.md = the frontend brief + build contract; built and live (web/).
Learning/Phase3/revision.md = Phase 3 concept cards (calibration → ONNX → serving DB → API → Docker → AWS → NGINX → Vercel).
NLP.md = every NLP / ML concept and metric used in the project, where and how (Shivam asked 2026-10-05).

**Postgres:** container `triageiq-postgres`, database `triageiq`, user `triageiq`, `localhost:5433`,
password in `.env`. CSV mounted read-only at `/import/complaints.csv`. **Division of labour:** I do
infrastructure, dependencies and DDL. **Load queries (`sql/02_load/`): one file at a time, only when
Shivam says go** — I write it, he reviews and understands it, then the next. Features (window/CTE
work) come after all tables are loaded.

---

## 5. Phase status

| Phase | Scope | Status |
|---|---|---|
| 0.1 | Docker + Compose, pgvector Postgres 16 | **Complete** — running on port 5433 |
| 0.2 | Source dataset + profiling | **Complete** — v2 training set built |
| 0.3 | **Schema + DDL** | **DDL done** — all tables created, 12/12 constraint tests pass |
| 0.4 | Bulk load | **Complete** — fact 4,826,564 · narrative 1,639,068 · events 14,479,692 · validated 21/21 · 6 indexes (company×issue lookup 0.19 ms, index-only); 31,378 company×issue pairs (F1) |
| 0.5 | Label as a SQL view | **Complete** — `v_label`: 4,826,564 rows · paid 60,952 · untimely 2,785 · 19 unknown → 0 · base rate 1.26% (F1); reads the partial index, 1 s |
| 1 | Layered point-in-time pipeline | **Complete** — `sql/03_features/` 01–08: `mv_features` (4,826,564 rows × 48 cols since v4, ~80 s build at 40) → `v_model_input` (23 inputs since v4; the shipped model reads the first **19**, see `sql/03_features/feature_dictionary.md`); recount tests + leak test pass (`sql/tests/`) |
| 2 | fusion model, ablation, evaluation | **Complete** — DistilBERT fusion v3 (3 × ~481 s on one T4): within-company **0.8224** (0.815–0.830), within-strata 0.9492, PR 0.4395, riskiest 10% catches 92.6% (FACTS.md); text-only 0.8036; v4 (+amount features) no better → v3 kept; error analysis (trap 14). Results: `training/results/`. pgvector similarity search never built (optional) |
| 3 | calibration, ONNX, serving DB, FastAPI, Docker, AWS, NGINX, website | **Complete (live 2026-10-05)** — Days 1–5 + frontend + blog in §12; concept cards `Learning/Phase3/revision.md`. Deferred (optional, §0): read-only DB role, CI (3.8), monitoring (3.9), screen recording |

---

## 6. Phase 0.3 — schema decisions (full reasoning: `Context/schema_explanation.md`)

| # | decision |
|---|---|
| D1 | Narrative in its own **extension** table `complaint_narrative` (1,639,068 rows) — not a dimension |
| D2 | Outcome / label lives in `complaint_events`, never on the fact — leakage needs a deliberate join |
| D3 | `date_received` on the fact (point-in-time anchor) **and** in the event log |
| D4 | **Surrogate** integer keys on every dimension; id mapping assigned once, never regenerated |
| D5 | `dim_company (company_id, company_name, first_seen_in_window)` — thin; no counts or rates |
| D6 | Hierarchies = two tables; child **UNIQUE (parent_id, child_name)** — names repeat across parents |
| D7 | Issue and Product are **independent** — 55% of issues span several products |
| D8 | Renamed products map to **one canonical** `product_id` |
| D9 | NULL child → `'(not specified)'` member, never a NULL FK (inner joins drop NULLs silently) |
| D10 | Keep **all** F1 rows; never store a pre-computed ratio in place of rows |
| D11 | Both `product_id` and `sub_product_id` (and issue pair) on the fact — hot path |
| D12 | Split/renamed products **route by sub-product** (bottom-up), keyed on (raw_product, raw_sub_product) — 14 raw → 11 canonical, new taxonomy names |
| D13 | Crosswalk is a **table** (`product_crosswalk`, 12 rules seeded in DDL) |
| D14 | `date_received` is `DATE` — source has no time of day |
| D15 | `responded` event has **no date** — CFPB never records it; the 60-day lag is an assumption |
| D16 | Composite FKs make the database reject a sub-product under the wrong product |
| D17 | Renamed **issue** → one canonical `issue_id` via `issue_crosswalk` (1 rule, 337,252 F1 rows rerouted); fact keeps `raw_issue` |

**Still open (none block the load):**
1. **`Submitted via`** — **CLOSED (2026-09-30): not a model input.** 1 value (Web) among complaints with
   text (F2) — constant on every training row. Stays in the fact table; left out of `v_base`.
2. **`Tags`** — kept as a nullable fact column. 94.49% null in F1, 87.82% in F3. **NULL = "no tag", not
   missing** (a form checkbox, known at intake). Payout rate F1: no tag 1.06% · Servicemember 2.43% ·
   **Older American 10.08%** · both 8.54%. Holds within product (credit card 26.41% vs 13.81%; credit
   reports 0.74% vs 0.04%) — not just mix. **DECIDED (2026-09-29): encode it — as two 0/1 flags**
   (`is_older_american`, `is_servicemember`) computed in `02_base`, not a 4-way one-hot: the value is a
   list, so 'both' must share what each flag learns. No schema change — a view reads the column.
3. **NULL outcomes** — 19 in F1. **DECIDED (Shivam, 2026-09-29): label 0**, like untimely. 8 have text;
   1 is in the shipped training set (test) and already has y = 0. The events table still stores NULL
   (raw truth); only the label view maps it to 0. Every complaint now has a label: 4,826,564 rows.
4. **`Untimely response`** — 2,785 in F1: the company **never answered** — no final outcome exists, no public
   response. (Different from *late*: 18,374 late answers, 15,589 still closed normally, 637 with money.)
   Almost all tiny companies: <10 complaints → 13.27% untimely; 1k+ complaints → 21 of 4.6M.
   **DECIDED (Shivam, 2026-09-28): label 0** — no money left the company, and the target is company cost.
   (My proposal to exclude was rejected.)

*Resolved:* crosswalk location → D13 (a table); `Date sent to company` → loaded straight from the CSV.

---

## 7. The v2 pivot — history, settled, do not re-litigate

v1 assumed a **synthetic** customer/transaction world. **Dead** — CFPB has no consumer identity.

**Decisions locked in:**
1. **No synthetic data.** 100% real, public, verifiable.
2. **Entity = company / issue**, not customer. Point-in-time window features survive.
3. **Label = `Closed with monetary relief`** — company cost, not consumer harm (no severity label
   exists; documented as a limitation).
4. **All products, no rule tier.** Text ranks complaints *within* low-payout products too.
5. **Serving the company.** Consumer view comes free via precedent retrieval — same model.
6. **`any_relief` rejected** — metadata-dominated, would make the transformer decorative.

*The exploration measurements behind these (Product-alone AUC 0.957, Company-alone 0.976, etc.)
were taken on various subsets and are **historical**. Current numbers: `FACTS.md` only.*

v1 archive: `Context/old_context/TriageIQ_v1_archive.md`. **Do not delete.**

---

## 8. Traps — active

1. **Window frames — three patterns, verified on PostgreSQL 18.4** (`TriageIQ.md` §1.4a).
   `date_received` is day-granularity; 43.46% of company-days hold >1 complaint. The **default frame
   includes the current row's own label**; without a tiebreak results change between runs; adding
   `complaint_id` to `ORDER BY` is **rejected** by `RANGE` interval frames; `ROWS` is not a time window.
   - outcome rates → `ORDER BY date_received RANGE BETWEEN UNBOUNDED PRECEDING AND '60 days' PRECEDING`
   - volume counts → `ORDER BY date_received RANGE BETWEEN '90 days' PRECEDING AND '1 day' PRECEDING`
   - sequence (LAG, rank) → `ORDER BY date_received, complaint_id`
2. **Two-tier rule.** Features computed over F1 (4,826,564); training on F3. The 3.2M no-narrative
   rows hold 42% of all payout outcomes.
3. **Target encoding must never be fitted on case-control-resampled data** — it cost the metadata
   branch 0.032 AUC. Entity rates come from F1 in Postgres.
4. **Post-intake columns are never features:** `Date sent to company`, `Company response to
   consumer`, `Timely response?`, `Company public response`.
5. **Three evaluation frames, three claims** — pooled 0.9713 / within-strata 0.9457 / within-company
   0.8115 (fusion, v3). Report the honest one; explain the gap.
6. **Recalibration is mandatory** — v3: keep-fraction 0.0892 → logit offset **−2.4165** (read it from the
   manifest, never type it; v2's −2.2572 is dead). It fixes sampling only — see trap 13 for drift.
7. **721 narratives straddle splits** (3,120 rows, 1.03%, 3 positives) — dup filter isn't group-aware.
   **FIXED in v3 (2026-10-01):** one text, one split (trap 12).
8. **Undeclared dependency:** scikit-learn is required but not in `pyproject.toml`.
9. **A global average is a credit-reporting average.** Credit reporting is 83.2% of F1 rows but 3.3% of
   payouts, so the global rate (1.26%, F1) is mostly its number. Smooth entity rates toward the
   **product** rate, not the global one. **K measured (2026-10-01, `05a_choose_k.sql`, tuning period
   Oct–Dec 2023, F1):** company × issue, history < 200 — AUC K=0 0.8690 · K=5 0.8901 · K=50 0.8754 ·
   K=500 0.8495; Brier best at K=5. Flat from 2 to 7. **DECIDED (Shivam, 2026-10-01): K = 5 — with a
   grain of salt:** (a) measured on each rate ALONE, not inside the fusion model — re-check K in the
   Phase 2 ablation; (b) one quarter of tuning data (Oct–Dec 2023); (c) differences are small overall
   (0.9777 vs 0.9769 at K=50, all complaints) — the gain is mainly for small-history pairs. K lives in
   ONE place, table `feature_params` — changing it is one UPDATE, never an edit to the view.
10. **Sub-issue changed meaning in Aug 2023.** All 122,207 missing sub-issues (F1) are structural, none
    skipped: 45 issues never have one (79,614); 3 depend on product — none under Payday (5,334); and
    **4 mortgage/payment issues got sub-issues only with the Aug-2023 form change (37,259)**. For those 4,
    `(not specified)` means "filed before Aug 2023": train (<2023-10) mostly sees it, test (2024) almost
    never. A sub-issue feature there encodes the date — use issue-level features; decide in Phase 1.
    **Bigger: Aug-2023 also RENAMED an issue.** 'Problem with a credit reporting company's investigation…'
    → 'Problem with a company's investigation…' on 2023-08-25 — same products, same 5 sub-issues:
    **893,566 complaints (18.5% of F1)**. Unfixed, its company×issue history resets 5 weeks before val.
    Plus 4 before-only and 14 after-only small issues (≤ 3,980 each). **Not a leak** (known at receipt) —
    a history reset + shift. **Fixed by D17** (issue crosswalk, 2026-09-28). The small ones and the 4
    sub-issue cases are still open → smoothing toward the parent rate, Phase 1.

12. **Training set v3 (2026-10-01) — rules chosen by me on Shivam's delegation; revisit after training.**
    Start 2022-04-01 (warm-up) · ≥20 real words (one rule replaces min-words + ≤30% blanks) · one text, one
    split (first appearance; later copies dropped, never moved) · cap 25 copies (v2 code DROPPED whole 26+
    clusters — contradicted its own docs) · train all payouts + 3× · val 80k · test 150k · seed 42.
    Result: train 62,940 (15,735 payouts) · val 80,000 (3.50%) · test 150,000 (2.31%) · **offset −2.4165**
    (keep-fraction 0.0892 — v2's −2.2572 no longer applies). Rule 4 dropped 106,942 complaints / 9 payouts,
    the cap 195,545 / 14: repeated templates almost never pay (unique texts 3.84% vs copies ≤0.05%).
    Idea for later: an as-of 'identical text seen before' count is a legit, likely strong feature.
    Heavy queries: `SET max_parallel_workers_per_gather = 0` — parallel hashes overflow the 1 GB shm_size.
14. **Why within-company stops near 0.82 — error analysis (2026-10-02, `training/error_analysis.py` →
    `training/results/error_analysis_v3.md`).** Missed payouts are mostly *surprise* payouts on non-money
    complaints (credit-report disputes, login problems; company×issue rate ≈ 0.001–0.05) — they read like
    thousands that never pay (likely goodwill credits). False alarms read exactly like refund cases (named
    fee, amount, unauthorized charge) — the company just didn't pay; the reason isn't in the text. ⇒ The
    ceiling looks like unobservable company decisions, not a weak reader; 0.9 is unlikely on average.
    **Dollar amounts:** 19.2% of F2 narratives mention `{$…}`; they pay 7.10% vs 0.99% and hold 63% of
    payouts (22,306 / 35,375). An earlier "0 amounts" check was a shell-quoting bug (`$` eaten by bash).
    Candidate SQL features: has_amount, log max amount — and a recent (last 6–12 months) company×issue rate.
13. **Calibration drifts (2026-10-02).** After the −2.4165 offset the full run predicts 3.11% on 2024 vs
    2.31% actual. The offset corrects the case-control sampling, NOT the year-on-year fall in payouts.
    Ranking is unaffected; before showing probabilities (API, expected cost), recalibrate on the most
    recent labelled data. **DONE 2026-10-03 (Platt, §12 Day 1)** — the API must apply `calibration.json`. **Kaggle UI:** a committed run's log page may stop updating — check the
    Output tab / metrics.json before assuming a stall (it fooled us on 2026-10-02).
    **Draft vs saved runs:** a draft (interactive) session keeps `/kaggle/working` only while it is on —
    download from the Output panel before stopping, or use Save Version → Save & Run All.
11. **Sort memory.** Default `work_mem` 4MB made window sorts over 4.8M rows spill to disk (28 GB temp,
    11+ min). Set `ALTER DATABASE triageiq SET work_mem = '256MB'` (in `12_indexes.sql`): same check 16 s.
    Docker VM has 8 GB — don't raise much further; a query can hold several sorts at once.
---

## 9. Phase 2 plan — decided

- **Encoder: DistilBERT**, revisit only if measurement says so. Cheaper upgrade path if needed:
  DistilRoBERTa (same speed, better pretraining) → DeBERTa-v3-base (strongest at 512).
- **`max_length=512`** (DistilBERT's hard ceiling). Length correlates with the label — positive rate
  peaks at 512–1k tokens (39.6%) then declines; 512 reads ~78% of the median doc in that band.
  **Measured 2026-09-28** (DistilBERT tokenizer, 20,000-narrative random sample of F2): 1.24 tokens/word
  median (p90 1.51; XXXX redactions push it up: 1.28 vs 1.17) → 512 tokens ≈ **410 words** (≈ 340 worst
  10%). Fit fully: 72.7% @256 · **91.8% @512** · 98.1% @1024. Tokens p50 149, p99 1,341.
- **512 is the baseline — no 256 run (Shivam, 2026-09-28).** His concern: long complaints matter
  (8.2% of the sample exceed 512, median 729 tokens). Order of attack:
  1. Collapse CFPB redactions (`XX/XX/XXXX`, `XXXX` runs → one special token each) — keeps meaning;
     70.3% of F2 narratives contain XXXX; alone it makes 18.2% of the long ones fit. *Proposed.*
     Cost today: `XXXX` = 2 tokens, a redacted date = 6; **15.5% of all tokens are blanks** (20k F2
     sample); 20.2% of complaints are ≥20% blanks, 8.6% ≥40%. Collapse to typed markers ([DATE],
     [REDACTED]) — don't delete: where and how many blanks appear is itself signal.
     **Where:** one SQL function `clean_narrative(text)` — the training export AND the API call it, so
     train and serve can't drift. **Already in place:** training drops complaints with ≥30% blank words
     (manifest: 1,569,044 → 1,444,945). Open (decide in `04_training_set`): that filter hides heavy-blank
     complaints from training though they still arrive in production — after collapsing, loosen it to
     drop only texts with almost no real words left.
     **Measured 2026-10-01 (all F2):** blanks ≥30% = 126,596 complaints (7.72%) but only 129 payouts
     (0.08–0.13% rate). After collapsing (5,000-sample each side): median 238 → 109 tokens, over 512:
     12.1% → 3.6% — heavy-blank texts get SHORTER, so loosening the filter can't push real words out.
     **Catch:** as plain text `[REDACTED]` = 5 word-pieces vs `XXXX` = 2 → register the markers as
     **special tokens** (`add_special_tokens` + resize embeddings), or collapsing lengthens typical texts.
  2. Head + tail truncation for the long ones.
  3. **Slice evaluation:** score the model separately on complaints cut at 512 vs those that fit —
     this is how we learn whether length matters (replaces the 256 ablation).
     **TF-IDF reference (v3 test, `test.ipynb` 6b):** 12,602 cut (8.4%), paying 3.41% vs 2.21%. AUC fits /
     cut — text 0.9603 / 0.9542 · features 0.9656 / 0.9495 · fusion 0.9722 / 0.9614. **Long complaints are
     harder for every model, even features-only (which reads no text).** So a DistilBERT gap alone proves
     nothing: truncation hurts only if DistilBERT's cut-slice AUC falls clearly below TF-IDF fusion's 0.9614.
     Smoke (10k, noisy): DistilBERT 0.9721 / 0.9529.
     **Full run verdict (2026-10-02): DistilBERT cut-slice 0.9614 = TF-IDF's 0.9614; fits 0.9741 vs 0.9722.**
     The 512 cut costs nothing relative to a read-everything model → **512 stays; head+tail is low priority.**
  4. If the cut slice underperforms: chunk + pool with the same DistilBERT (median 2, p90 3 chunks).
  5. Last resort: a 1024+ model (jina-embeddings-v2-small, ~33M, 8,192 ctx).
  **Steps 4–5 NOT NEEDED (2026-10-02):** the read-every-word TF-IDF fusion ties DistilBERT on the cut
  slice (0.9614 = 0.9614) — no evidence of signal beyond piece 512; the cut slice is 8.4% of test
  complaints / 12.4% of payouts (430 / 3,471); the ceiling is company decisions (trap 14), not length.
  **Rejected, measured:** lemmatization — even an impossible best case (every `##` piece removed)
  fits only 22.0% of the long ones, and it feeds DistilBERT unnatural text; stopword removal — fits
  55.5% but deletes not / no / never / nothing / cannot.
- **MODEL CHOSEN (2026-10-02): DistilBERT fusion — text + the 19 SQL inputs.** Best on every score (within-
  company 0.8224, riskiest 10% catches 92.6% vs 88.8% text-only ≈ 130 more payouts in the 2024 test).
  Shivam: the text-vs-fusion debate is settled → **DeBERTa text-only run skipped** (optional; the error
  analysis says the missing information isn't in the text). Fallback for CPU serving: TF-IDF fusion
  (0.8115, 91.7%) if DistilBERT is too slow on Cloud Run — measured in Phase 3: fast enough (p50 ~0.2 s on EC2), no fallback. Before showing any
  probability: recalibrate on recent data (trap 13). Next improvement: dollar-amount + recent (6–12 mo)
  company×issue rate as SQL features → training set v4 → one fusion rerun (~30 min).
- **v4 features (2026-10-02): 23 inputs = 19 + `has_amount`, `log_max_amount` (01_schema/08) + recent
  company×issue / company rates (04/05, 365 days ending 60 back).** Export `triageiq_training_v4` = same
  rows/labels as v3. SQL-only standalone: within-company **0.7779 (0.769–0.787)** vs v3 0.7537 (0.744–0.763);
  the **amount** clue carries it (LR: 0.7500 → 0.7710), recent rates add nothing (0.7490). Notebook switch
  `DATA` (default v4). Open: the text model already reads `{$…}`, so the fusion gain may be smaller —
  **retrain DistilBERT fusion on v4 to find out.**
  **RESULT: v4 fusion within-company 0.8195 (0.811–0.827) vs v3 0.8224 (0.815–0.830); PR 0.4426 vs 0.4395;
  top-10% 92.9% vs 92.6% — same within noise.** DistilBERT already reads the amounts in the text.
  **DECISION: keep the v3 fusion (19 inputs)** — equal score, simpler to serve.
- **Class imbalance — plan agreed 2026-10-02 (after the text-only run).** Payouts are ~2–3% of complaints.
  Today: **undersampling** (all 15,735 train payouts + 3 non-payouts each = 62,940; ~480k eligible
  non-payouts thrown away) — chosen to save GPU, not because it is best. Options: (1) undersample [now];
  (2) **weighted loss on ALL rows** — keep every complaint, weight payouts ~×34 (529,022 / 15,735);
  (3) SMOTE/oversampling — **rejected for text** (can't blend two complaints into a valid new one; fine
  for numeric data like transactions); (4) focal loss — a variant of (2). Steps:
  a. **Half-data check** — `TRAIN_FRAC=0.5` (notebook cell 1), ~15 min. If halving drops within-company
     clearly (outside the ± range), more data helps → b. If not → skip, spend hours on features/DeBERTa.
     **RESULT (2026-10-02): fusion at 50% → within-company 0.8166** (pooled AUC 0.9722, PR 0.4280, strata
     0.9465) vs 0.8224 (0.815–0.830) at 100% — **inside the range: halving cost ~0.006.** ⇒ more data is
     unlikely to help; **b is deprioritised.** Source: Kaggle *draft* session read off screen — outputs were
     not saved (draft files vanish when the session stops; only Save Version keeps them), so no results
     file / range for it. Rerun as a saved version only if it must enter FACTS.md.
  b. Training set **v4**: all eligible train non-payouts + weighted loss (~3.5 h on one T4, measured
     481 s/epoch per 63k rows). Probability correction becomes −ln(weight) instead of ln(keep-fraction).
  c. Final model also trains on Oct–Dec 2023 (+~3k real payouts), fixed epochs (no val left to stop on).
  Expected gain: modest (+0.005–0.02) — error analysis says the misses are mostly information the text
  lacks. Every run now reports a 95% range (`evaluate_fusion.py`): fusion 0.8224 (0.815–0.830).
- **Training speed:** `group_by_length=True` is the big win (2.23× fewer tokens; dynamic padding
  alone does *nothing* on this data), `fp16` on T4, freeze encoder epoch 1, early stop on val PR-AUC,
  ≤3 epochs, develop on a 10k subset.

---

## 10. Guardrails

- Every feature computed **as-of complaint receipt**, never as-of today.
- Window frames **exclude the current row** — the label lives in the same database.
- Smoothing priors must **also** be as-of date.
- Split is **temporal**, never random.
- Preprocessing fit on train only, persisted, reused verbatim at inference.
- The API accepts **identifiers + text, never features**. State the trade-off.
- Aggregate-then-join. **Check row count after every join.**
- Smell test: within-strata AUC materially above ~0.95 → hunt for the leak.

---

## 11. Optional, cuttable — expected-cost ranking (Phase 2.6)

`expected_cost = P(monetary relief) × claimed_amount` (amount regex-extracted, product-median
fallback). A $200k claim at P=0.012 → $2,400; a $10 claim at P=0.285 → $3. Recovers magnitude
without a severity label. Amount is *claimed*, not verified; ~81% of complaints need the fallback.

---

## 12. Phase 3 — deployment plan (agreed 2026-10-02)

**Goal:** a public link for the resume (Swagger UI, ground rule 2) that anyone can open for **a few months**.
Shivam has **$140 AWS credits** (account ready 2026-10-03; EC2 likely — confirm on Day 5); long-term hosting is NOT needed. Replaces the spec's Cloud Run plan
(TriageIQ.md §3.3) — AWS only because the credits are free. Shivam is setting up AWS himself (with Claude
Code) and will report back; check his account's credit type / expiry date before relying on it.

**Chosen: option A — one small Lightsail or EC2 server (~2 GB RAM, ~$10–12/month est.), Docker Compose,
two containers:** `postgres` (small serving DB) + `api` (FastAPI + model on CPU). Budget alarm at
$10 / $50 / $90. (Rejected for now: 4 GB server ~$20–24/mo — less runway; Lambda — cold starts, more work.)

**What ships — NOT the 16 GB local DB (measured 2026-10-05) (it stays on the Mac as the rebuild source):**
| serving table | built by SQL from the full DB | purpose |
|---|---|---|
| dims (company, product, sub-product, issue, state names) | copy | pick from lists, not ids |
| **entity snapshot as of 2024-12-31** — company×issue / company / issue / product rates, shares, trends, quiet days | same formulas as `sql/03_features/`, evaluated at the data's end | score a NEW typed-in complaint |
| the 150,000 test complaints (2024): 19 inputs + clean text + true outcome | from `v_model_input` / `v_training_export` | demo "real complaint → prediction vs what happened" |
≈ 300 MB. Features stay SQL-only (ground rule 4); AWS holds results. **Limit to state openly:** the data ends
2024-12-31, so new complaints are scored "as of end-2024" — live updates need a refresh job (later).

**Model:** DistilBERT fusion v3 (`training/outputs/fusion_distilbert_full/model.pt`, 19 inputs). Before
deploy: (1) **recalibrate** on the most recent labelled data (trap 13: 3.11% predicted vs 2.31% actual);
(2) export to **ONNX** (same model, runs on ONNX Runtime — no PyTorch in the image) and optionally
**int8** (~130–140 MB est., not ~70: the 23M-number vocabulary table usually stays 32-bit). **Ship rule:**
score all 150k test complaints with both; int8 ships only if within-company stays inside the PyTorch
model's range and top-10% recall holds; else ship fp32 ONNX. Also report senior/template flips + CPU latency; TF-IDF fusion is the
fallback if CPU is too slow. Endpoints (spec §3.1): `POST /predict` (ids + text, never features),
`GET /complaint/{id}` (demo), `GET /health`, `GET /model-info`.

**Never commit AWS keys / `.env`.** Cost is a hard limit: free credits only (ground rule 3).

**TIMELINE (Shivam, 2026-10-03, hard): local demo in 4 days, deployed in the 2–3 days after; exams at the
end of October need 10 full days (~Oct 21–31).** So: **build first, learn just-in-time** (I write the code,
he reviews and explains it back; a short concept lesson before each day's work). Courses (Andrew Ng's
*Machine Learning in Production*, freeCodeCamp API course, Nana Docker) and the DMLS book move to **after
deploy (~Oct 10–20) and after exams.** Day 1 = Sat 2026-10-03 (shift if he starts later).
| day | build (Learning/Phase3 step) | done when |
|---|---|---|
| 1 | 3.1 recalibrate (fit Jan–Jun 2024, check Jul–Dec, refit on all 2024) + threshold · 3.2 ONNX fp32 + parity on 1k + CPU latency | avg predicted ≈ actual on H2-2024; ONNX = PyTorch to ~1e-5; p50/p95 known |
| 2 | 3.3 serving DB (`sql/06_serving/`): dims + snapshot as of 2024-12-31 + 150k test complaints + `clean_narrative()`; skew test; `pg_dump` | skew test passes; dump size known |
| 3 | 3.4 FastAPI (`api/`): /predict, /complaint/{id}, /health, /model-info on the Mac | Swagger works; /complaint score = offline score |
| 4 | 3.5 Docker: api Dockerfile + compose (postgres from dump + api) → **localhost demo from scratch** | `docker compose up` on a clean volume → demo works |
| 5 | 3.6 AWS: budget alarms, server (~2 GB + swap), security group, SSH, copy model + dump, compose up | public IP opens /docs from another network |
| 6 | 3.7 NGINX + read-only DB role + input limits + rate limit; README live link; screen recording | 5432 closed from outside; recording saved |
| 7 (buffer) | minimal 3.8 (ruff + pytest + build) and 3.9 (prediction_log + one SQL drift query) — else after exams | — |
**Day 1 DONE (2026-10-03):** `training/serving/calibrate.py` → Platt on z = logit − 2.4165 (fit on Jul–Dec 2024): a 0.7025,
b −0.7621; checks fit Jan–Jun → Jul–Dec: predicted 2.22% vs actual 2.10% (offset only 2.91%); fit Jul–Sep → Oct–Dec
2.04% vs 2.13%; isotonic rejected (its ties lower within-company 0.8296 → 0.8290). Senior if p ≥ 0.0367 = riskiest
10% of 2024 test: catches 92.6% of payouts, 21.4% pay vs 0.19% in template; within-company unchanged 0.8224.
`training/serving/export_onnx.py` → bundle `training/outputs/serving_v3/` (model.onnx 266 MB, tokenizer.json,
preprocessing.json, calibration.json): tokenizer 1000/1000 identical, ONNX vs PyTorch max |Δlogit| 1e-5, vs Kaggle T4
fp16 max |Δp| 0.00036, rank corr 0.999999; M4 2 threads p50 20 ms / p95 81 ms (≤128 tok 11 ms, 256–512 54 ms).
Results: `training/results/calibration_v3.json`, `onnx_v3.json`. int8 deferred: decide on Day 5 from server latency.
**Day 2 DONE (2026-10-03):** `sql/06_serving/` 01–03 + `sql/tests/06_serving_check.sql`. Schema `serving` = scoreboard
(`snap_*`, as of 2025-01-01, built by `build_serving_snapshot(date)` in ~12 s) + `serving.model_input(...)` (19 inputs,
~0.06 ms) + lists + `serving.clean_narrative()` (copied via pg_get_functiondef) + `serving.test_complaints` (150k).
**Skew test: 19/19 inputs identical (max diff 0) on 20,773 complaints of 3 past days.** Quiet-days rule: the 2nd+
complaint of a company on the same day has gap 0 in training; a live complaint has no same-day predecessor.
187 MB in Postgres → `training/outputs/serving_v3/serving_db.dump` 57 MB; restore into an empty DB verified.
**Day 3 DONE (2026-10-04):** `api/` — FastAPI: `POST /predict` (names + text → SQL inputs via serving.model_input,
text cleaned by serving.clean_narrative, ONNX, Platt, route), `GET /complaint/random|{id}` (2024 demo vs what happened),
`GET /options/{products|issues|states|companies}`, `/health`, `/model-info` (reads `model_card.json`, generated by
`training/serving/model_card.py` from the results files). Clean venv `.venv-api` (no PyTorch, 210 MB; pinned
`api/requirements.txt`); run `sh api/run_local.sh` → localhost:8000/docs. **Check: 300 random 2024 complaints via the
API vs the offline score: max |Δp| 0.00017, same route 100%; latency p50 28 ms / p95 94 ms (Mac, incl. DB).**
Inputs validated (≥20 words like training; unknown company → no-history + note; bad names → 422 with choices).
**Day 4 DONE (2026-10-05) — LOCAL DEMO COMPLETE:** `deploy/` — `api.Dockerfile` (multi-stage, python:3.11-slim,
non-root `app`, model baked in, uvicorn 0.0.0.0:8000, 1 worker) · `docker-compose.yml` (project `triageiq-serving`:
`db` = postgres:16-alpine, NO published port, restores `serving_db.dump` on first start via `initdb/01_restore.sh`,
TCP healthcheck so the api waits for the restore; `api` → db by service name) · root `.dockerignore` (allow-list:
api/ + 5 model files) · `deploy/.env` (git-ignored, random password; `.env.example` committed). Run:
`docker compose -f deploy/docker-compose.yml up -d --build` → localhost:8000/docs. From an empty volume: db restored
150,000 demo complaints in ~3 s, api healthy ~6 s later. Image content ≈ 650 MB (base ~120 + packages 257 + model 267;
Docker Desktop shows 1.07 GB incl. its compressed copy). **Checks: 300 complaints container vs offline max |Δp| 0.00024,
same route 100%.** Container on the Mac (Linux VM): p50 70 ms / p95 247 ms; RAM api 455 MiB + db 45 MiB (≈ 0.5 GB →
fits a 2 GB server; int8 not needed for memory). The Mac's Docker builds ARM64 (aarch64) images — Day 5: pick an ARM
(Graviton) server, or build on the server itself.
**Day 5 PREP (2026-10-05):** `deploy/nginx/default.conf` (reverse proxy api:8000 → port 80; 5 req/s per IP, burst 20,
429 over the limit; body ≤ 64 KB; tested locally: 60 rapid calls → 19 × 200 / 41 × 429) as compose service `nginx`
under profile `public` (server only); the api is now published on 127.0.0.1:8000 only. `deploy/server_setup.sh`
(swap 2 GB, Docker via get.docker.com, deploy/.env with a random password made ON the server) · `deploy/push.sh <ip>`
(rsync of the ~330 MB needed files — macOS's openrsync: no --info flag — then setup + `compose --profile public up
-d --build`). AWS plan: region **ap-south-1** (Shivam is in IST), Ubuntu 24.04 ARM64, **t4g.small** (matches the
Mac's ARM images), 20 GB gp3 encrypted, IMDSv2, CPU credits *standard* (no surplus charges), Elastic IP, security
group: 80 from anywhere, 22 from Shivam's IP only, no 5432/8000. Budgets: monthly cap + cumulative alerts at
$10/$50/$90 of the $140 credits. AWS CLI 2.37.8 installed, NOT signed in yet (`aws login` = Shivam's step).
**Frontend (Shivam, 2026-10-05, revised same day):** JS static site on **Vercel**, brief in `frontend.md` (Part A for
Claude Design, Part B build contract). `vercel.json` rewrites the API paths to the EC2 server; Vercel adds a secret
header and NGINX will accept only requests carrying it, and rate-limit on Vercel's `x-real-ip` (the visitor) instead
of the connecting IP (which would be Vercel's). OPEN: Vercel docs show only https rewrite targets — test http first;
if refused, add HTTPS on the server (Let's Encrypt on an sslip.io name).
**BUILT (2026-10-05, Shivam: "choose whatever design, elegant, not AI slop" — no Claude Design):** `web/` = static
site, plain HTML/CSS/JS, no build step: `index.html` (score form + assessment: big serif %, route badge, log-scale risk
bar with the senior line, "what the model saw" ledger), `real.html` (random 2024 complaint as a case file: CFPB blanks
as black redaction bars, model call vs what happened, stamp Caught / Missed / False alarm / Right call), `how.html`
(3 steps, live stats from /model-info, limits, stack), `styles.css` (paper/ink/oxblood, Source Serif 4 + IBM Plex
Sans/Mono, hairline rules, no gradients/shadows/emoji), `app.js`, `vercel.json` (cleanUrls + rewrites of /predict,
/complaint/*, /options/*, /model-info, /health, /docs, /openapi.json → http://3.106.107.237; rewrite caching off),
`dev_server.py` (local stand-in for Vercel: `API=http://3.106.107.237 python3 web/dev_server.py` → localhost:3000).
Verified locally against the LIVE API with headless-Chrome screenshots (desktop + 390 px mobile + error state).
**DEPLOYED (2026-10-05) — PUBLIC LINK: https://triageiq-mu.vercel.app** (`triageiq.vercel.app` was taken). Vercel account
`bajaj30`, project `triageiq` (CLI linked in `web/.vercel/`, git-ignored). Deploy the site: `cd web && npx vercel deploy
--prod --yes`. Vercel → **plain-http** destination WORKS (verified; no HTTPS needed on the server). `vercel.json` now uses
**`routes`** (not rewrites — `routes` can't mix with rewrites/cleanUrls/headers): API paths → http://3.106.107.237 with a
`request.headers` transform adding `x-origin-secret` = Vercel env **ORIGIN_SECRET** (sensitive, production); then
`handle: filesystem`; `/real`, `/how` → .html. Query strings are forwarded. **The server is locked to Vercel:**
`deploy/nginx/default.conf.template` (nginx image envsubst) — `map $http_x_origin_secret` → 403 without the secret;
rate limit keyed on Vercel's `x-real-ip` ONLY when the secret matches (requests reach NGINX from many Vercel IPs).
Secret = 48 chars, generated ON the server into deploy/.env by `server_setup.sh`, piped to Vercel (`vercel env add …
--value "$V" --sensitive --yes`; stdin piping via npx did NOT work), never printed/committed. Checks: through Vercel all
pages + API 200, /predict 31.7%; direct to the IP 403 (also with a guessed header); 40 rapid calls via Vercel → 22 × 200
/ 18 × 429. **Trap (cost ~3 min downtime):** a 48-char map key overflows NGINX's default `map_hash_bucket_size` 64 →
`[emerg] could not build map_hash` → container restart loop; fixed with `map_hash_bucket_size 128;`. Local test had used
a short secret — test with production-like values. `web/dev_server.py` needs `ORIGIN_SECRET=…` to reach the live server.
**Day 5 DONE (2026-10-05) — server live at 3.106.107.237 (now reachable ONLY through Vercel — see the frontend note).** AWS account = **Free plan**,
$140 credits, plan expires **2027-04-02**; member of an AWS Organization whose SCP **denies EC2 in ap-south-1** → deployed
in **ap-southeast-2 (Sydney)**. CLI: `export AWS_PROFILE=triageiq` (signed in via `aws login`, role
AccountFullAccessRole; default region ap-southeast-2). Prices (Pricing API): t4g.small $0.0212/h ($15.48/mo) + gp3
$0.096/GB-mo × 20 ($1.92) + public IPv4 $0.005/h ($3.65) ≈ **$21.05/month** → ≈ $124 by the plan's end.
**Resources (all tagged project=triageiq) — delete ALL at teardown, then RELEASE the Elastic IP (it bills while idle):**
instance `i-0c9bd599b5b3d3a29` (t4g.small, Ubuntu 24.04.5 ARM64, 20 GB gp3 encrypted, IMDSv2, CPU credits **standard**)
· security group `sg-065abe94829601344` `triageiq-web` (80 from anywhere v4+v6; 22 from Shivam's IP only — his ISP IP may
change → re-authorize) · Elastic IP `3.106.107.237` (`eipalloc-03a976f274a89c888`) · key pair `triageiq-key`
(private key `~/.ssh/triageiq-key.pem`, never in the repo) · budget `triageiq-guard` ($25/mo, actual + forecast > 100%,
same email as the pre-existing "My Monthly Cost Budget" $65). Deploy/update: `sh deploy/push.sh 3.106.107.237`.
**Checks from outside:** health/docs 200; the 3 Postman examples identical (31.7% / 0% / 40.5%); ports 5432 and 8000
closed; 200 real complaints via the public link vs offline max |Δp| 0.00017, same route 100%.
**CPU credits (trap):** T4g in *standard* mode starts with 0 credits → throttled to the 20% baseline during setup and the
200-call test (CPUCreditBalance ≈ 0, CPU 19.8%, steal 32%). Earns 24 credits/h when idle (max 576). Kept *standard*
on purpose: it can never bill surplus credits (*unlimited* would). Re-measure latency once credits have built up.
**Mac vs server — all measured. Server column re-measured 2026-10-06 at full speed** (CPU 100% busy, 0% steal under
load → not throttled; the first Day-5 numbers were taken with 0 CPU credits and are kept only as the "out of breath"
story). The server's 2 Graviton2 cores are ~10× slower than the M4 at this model. Used in README, blog_log, NLP.md:
| number | Mac value | how measured | on the server |
|---|---|---|---|
| full local DB (stays home) | 16 GB | `pg_database_size`, 2026-10-05 | — (never ships) |
| serving schema / dump file | 187 MB / 57 MB | `pg_total_relation_size` / file size | restored DB 188 MB |
| feature lookup `serving.model_input` | 0.06 ms | EXPLAIN ANALYZE (in-DB, no network) | 0.6 ms p50 / 2.9 ms p95 (round trip from the api container, 50 calls) |
| model alone (ONNX, 2 threads) | p50 20 ms / p95 81 ms | `onnx_v3.json`, 300 complaints | **p50 218 ms / p95 803 ms** (300 random 2024 complaints, inside the api container) |
| API end-to-end `/complaint/{id}` (DB + model) | p50 28 ms / p95 94 ms | 300 calls, 2026-10-04 | **p50 210 ms / p95 807 ms** server-side (300 calls on the server, 2026-10-06). Throttled Day-5 run: p50 501 / p95 3,590; from India incl. the trip to Sydney p50 1,212 / p95 4,231 (200 calls) |
| API `/predict` (example complaint) | ~41 ms | one call | **p50 82 ms / p95 86 ms** server-side (20 calls, short text; 2026-10-06) |
| rebuilding all training features (`mv_features`) | ~80 s | Phase 1, at 40 columns — not re-timed at 48 | — (never ships) |
| API in Docker on the Mac `/complaint/{id}` | p50 70 ms / p95 247 ms | 300 calls, 2026-10-05 | (= the row above) |
| RAM: api container / db container | 455 MiB / 45 MiB | `docker stats` after 300 calls | api 546 MiB · db 52 MiB · nginx 3 MiB after 14 h live (2026-10-06); ~850 MiB still available → int8 NOT needed for memory |
**Later (after deploy / after exams) — v5 "recency" retrain, Shivam's idea 2026-10-03:** the raw CSV runs to
2026-08-27 (raw file, complaints with text): 2025-H1 695,184 (1.08% pay) · 2025-H2 526,871 (1.46%) · 2026-H1 109,324
(6.54%) · 2026-Jul/Aug 3,707 (9.50%, 49.73% still 'In progress'). **2026 is unusable** — outcomes not in yet and
narratives are published late, so the few public 2026 texts are a biased early subset. **2025 is usable.** Plan:
extend the window through 2025 (check 2025 taxonomy renames), time-decay sample weights (needs its own probability
correction), test on 2025-H2, champion-vs-challenger vs v3 → the step-3.9 'drift → retrain' story. Evidence it isn't
urgent: v3's ranking did not decay in 2024 (within-company Jul–Dec 0.8296 vs full year 0.8224); the drift is in the
base rate, which calibration on the newest data already handles. ~2 days of work → not in the 4-day plan.
**Cut / deferred:** int8 (ship fp32 ONNX unless time), CI/CD and the drift report (buffer day or later), HTTPS/domain
(optional), courses + book (after deploy).
