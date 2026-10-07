# TriageIQ — every NLP and ML concept we used

Every NLP and machine-learning idea the project touched, in the order we met it. Each entry has three parts:
- **what** it is, in a line or two;
- **in TriageIQ:** how we used it, with the number;
- **where:** the file.

Groups behind the numbers: *all complaints* = 4,826,564 (2022–24) · *complaints with text* = 1,639,068 ·
*the 2024 test set* = 150,000 complaints, 3,471 payouts. Source of every number: `Context/FACTS.md` and the files
named. Metric definitions: `Context/metrics.md`.

---

## 1. Understanding the text (EDA)

**Corpus profiling at scale (streaming).** A 9.2 GB CSV can't be loaded at once.
- **In TriageIQ:** read in 250k-row chunks, kept only the needed columns, cached as Parquet (columnar, compressed).
- **Where:** `Data/EDA.ipynb`.

**Real or machine-written? A five-test gate.** Synthetic (LLM-made) text would make a language model
pointless, so we tested authenticity with numbers:

| test | what it measures | result | suspicious if |
|---|---|---|---|
| Exact-duplicate rate | share of texts identical to another (MD5 hash of normalised text) | 19.8% raw → 0.3% after the copy cap (early sample) | — |
| Sentence-length coefficient of variation | std ÷ mean of sentence lengths. People write unevenly; LLMs are regular | 0.94 | < 0.35 |
| **Type-token ratio (TTR)** | distinct words ÷ total words: vocabulary variety | 0.065 raw, 0.081 after removing copy-paste floods | < 0.08 |
| Top repeated opener | share of texts starting with the same 5 words | 3.7%: legal wording from the Fair Credit Reporting Act that people copy from forums, evidence of humans | > 5% |
| Unfilled placeholders | `{name}`-style template slots | 0 real ones | > 0 |

- **Heaps' law:** longer texts reuse more words, so TTR only compares fairly at equal length. We always measured
  it on exactly 100,000 words.
- **Two lessons:**
  - The "placeholders" flagged at first were CFPB's `{XXXX}` amount redactions. The regex `\{[a-z_]+\}` ran
    case-insensitive and so matched capital X. Print the matches, don't trust a counter.
  - 40-character "words" were URLs, not corrupted text.
- **Where:** `Data/EDA.ipynb`, `Data/docs/data_profile.md`.

**Duplicate anatomy and near-template text.**
- 41% of raw narratives sit in a duplicate group, and the largest group has 30,110 copies of one template.
- Unique texts pay out 3.84%; texts seen 2+ times, 0.05% or less (complaints with text, from April 2022).
- **Text hashing** (`text_hash` = md5 of lower-cased, trimmed, whitespace-collapsed text) finds exact copies cheaply.
- **Where:** `sql/01_schema/07_narrative_columns.sql`.

**Redaction analysis.**
- The CFPB hides private details as `XXXX`, and dates as `XX/XX/XXXX`.
- 70.3% of complaints with text contain XXXX.
- 15.5% of all word-pieces are blanks (a 20,000-complaint sample).

**Length analysis.**
- 1.24 word-pieces per word (median), so 512 pieces ≈ 410 words.
- 91.8% of complaints fit in 512 pieces (20k sample, complaints with text). In the 2024 test set, 8.4% get cut.

---

## 2. Cleaning and preparing text

**Normalisation.** Lower-case, trim, collapse whitespace before hashing, so trivial differences don't hide copies.

**Typed placeholder tokens instead of deletion.**
- Each run of blanks becomes **one** marker, `[DATE]` or `[REDACTED]`.
- This keeps the information that something was hidden there, while freeing room in the 512-piece window.
- It's one SQL function, `clean_narrative()`, used by both the training export and the live API, so there's no
  train/serve skew.
- **Where:** `sql/04_training_set/01_clean_narrative.sql`.

**Regex information extraction.**
- Dollar amounts appear as `{$35.00}`; a function reads the largest one.
- 19.2% of complaints with text mention an amount. They pay 7.10% vs 0.99% and hold 63% of payouts.
- **Where:** `sql/01_schema/08_narrative_amounts.sql`.

**Real-word count as a quality filter.**
- `n_real_words` = alphabetic words minus `XX…` blanks.
- Training keeps complaints with ≥ 20 real words, and the API enforces the same rule.

**Copy cap and group-aware splitting.**
- At most 25 copies of any text.
- A text belongs to the split where it *first* appeared, so identical complaints can't be in both training and
  test (*group leakage*).
- **Where:** `sql/04_training_set/02_training_set.sql`.

**Considered and rejected — measured first:**
- **Lemmatisation** ("charged" → "charge"): even a perfect one fits only 22% of the long complaints, and it hands a
  transformer unnatural text.
- **Stopword removal:** it fits 55%, but it deletes *not, no, never*. "I did **not** authorise this" becomes
  "authorise".
- Both are tools for word-counting models, not for transformers.

---

## 3. Tokenisation (how the model reads)

**Subword tokenisation (WordPiece).**
- Text is split into pieces from a fixed vocabulary of 30,522. Rare words become several pieces:
  "chargeback" → `charge ##back`, "overdraft" → `over ##dra ##ft`.
- A CFPB blank `XXXX` costs 2 pieces (`xx ##xx`) and a hidden date 6, hence the markers in section 2.
- DistilBERT-*uncased* also lower-cases and strips accents first.

**Special tokens.**
- `[CLS]` (start; its output summarises the text), `[SEP]` (end), `[PAD]` (filler).
- **Added tokens:** `[DATE]` and `[REDACTED]` were registered as new special tokens, and the embedding table was
  resized (30,522 → 30,524).
- As plain text, `[REDACTED]` would cost 5 pieces.

**Maximum length and truncation.** DistilBERT reads at most 512 pieces; we keep the first 510 plus `[CLS]`/`[SEP]`
("head" truncation).
- **Head+tail** truncation (keep the start and the end) was built but not needed. The cut complaints scored the same
  as with a read-everything model.

**Attention mask.** Marks real pieces (1) vs padding (0), so the model ignores filler.

**Dynamic padding vs length-grouped batching.**
- Padding each batch to its longest text gave **no** speed-up on this data (1.00×), because almost every batch holds
  one long complaint.
- Grouping similar lengths together cut padded pieces **2.23×**.

**Tokeniser parity.** The serving tokeniser (Rust `tokenizers` library) gave the same pieces as training's
(`transformers`) on 1,000 of 1,000 test complaints. **Where:** `training/serving/export_onnx.py`.

---

## 4. The classic baseline: TF-IDF + logistic regression

**Bag of words.** A text becomes word counts; order is ignored.

**TF-IDF.** Term frequency × inverse document frequency: a word counts more the more it appears in *this*
complaint, and less the more complaints use it. Our settings:
- unigrams **and bigrams** (`ngram_range=(1,2)`, so "not authorised" is one feature);
- at most 80,000 features;
- a word must appear in ≥ 5 complaints;
- **sublinear TF** (1 + log count, so 10 mentions ≠ 10× the weight);
- accents stripped.

**Logistic regression.** A weighted sum of features squeezed into a probability by the sigmoid. Linear,
interpretable, fast (liblinear solver).

**One-hot encoding + standardisation** for the 19 track-record inputs: one 0/1 column per category value; numbers
rescaled to mean 0, std 1 (fitted on training rows only).

**Early fusion.** TF-IDF columns and the track-record columns glued side by side into one sparse matrix, one model.

**Result** (within-company AUC, 2024 test): text 0.7956 · track record 0.7500 · **fusion 0.8115**, the bar the
transformer had to beat.
**Where:** `training/test.ipynb`.

---

## 5. The transformer: DistilBERT

**Self-attention.** Every word-piece looks at every other piece to decide what it means in context ("charge" in
"charge my card" vs "in charge of").

**The transformer encoder.** Stacked layers of attention + feed-forward networks, producing a 768-number vector per
piece.

**Pre-training with masked language modelling.** BERT learned English by guessing hidden words in huge amounts of
text, before ever seeing a complaint.

**Knowledge distillation.** DistilBERT is a smaller *student* (6 layers, about 66M parameters) trained to mimic
BERT. Its authors report it is 40% smaller and 60% faster, keeping about 97% of BERT's language understanding.
We chose it for the free T4 GPU and CPU serving.

**Transfer learning / fine-tuning.** We started from the pre-trained weights and trained further on our task, so
the model reused general English knowledge.

**`[CLS]` pooling.** The vector at the `[CLS]` position represents the whole complaint; it feeds our layers.

**Considered, not used:**
- **DeBERTa-v3** (stronger at 512): skipped, because error analysis says the missing information isn't in the text.
- **Long-context models** (ModernBERT, a jina 8k model) and **chunk-and-pool:** not needed. On complaints cut at
  512, DistilBERT tied a read-every-word model (0.9614 = 0.9614).

---

## 6. The fusion network (text + track record)

**Multimodal, intermediate fusion.** Two branches, one head:
- **text branch:** `[CLS]` 768 → 128;
- **table branch:** the track record → 64;
- the two are joined (128 + 64 = 192) → 64 → 1 score (logit).

**Entity embeddings.** Each category value gets a small learned vector instead of a one-hot column: product 4
numbers, sub-product 8, issue 12, state 8. Similar categories end up close together.

**Feature scaling.** `log1p` for the heavy-tailed "days since last complaint", then standardisation with
**training-only** mean and std. These are saved (`preprocessing.json`) and reused unchanged when serving.

**Layers.** Linear layers, **GELU** activations (a smooth ReLU), **dropout** 0.1 (randomly silences units while
training, so the model can't lean on any one).

**Where:** `training/fusion_distilbert.ipynb`.

---

## 7. Training mechanics

**Loss: binary cross-entropy** on the logit (`BCEWithLogitsLoss`). It punishes confident wrong answers hardest.

**Optimiser: AdamW** with **weight decay** 0.01, which shrinks weights a little each step to prevent over-fitting.

**Discriminative learning rates.** 2e-5 for the pre-trained encoder (gentle), 1e-3 for the new layers.

**Learning-rate schedule.** Linear **warm-up** over 5% of steps, then linear decay.

**Freezing, then unfreezing** (*gradual unfreezing*, ULMFiT). Epoch 1 trains only the new layers while the encoder
is frozen, so random gradients can't damage pre-trained knowledge. Epochs 2–3 train everything.

**Gradient clipping** (norm 1.0) stops a single bad batch from taking a huge step.

**Mixed precision (fp16) + gradient scaler** on the T4 GPU: half the memory, faster maths, with the scaler stopping
tiny gradients from rounding to zero. The T4 has no bf16.

**Gradient accumulation:** built for DeBERTa (half batch × 2 steps); not used in the final run.

**Early stopping** on validation **PR-AUC**, at most 3 epochs, patience 1.

**Reproducibility:** seed 42 everywhere; the dataset file is verified by sha256 against a manifest before training.

**Compute:** about 481 s per epoch on one free Kaggle T4.

## 7.5 Explaining a prediction — what an attribution does and does not say

**Attribution.** A score is a result of many inputs. An attribution shares that result among the words and
track-record values the model read; it describes the model's behavior, not what caused a company to pay.

**Shapley values.** A cooperative-game idea for sharing credit: measure what each input adds across different
orders in which inputs are introduced. Contributions are in the model's raw-logit units and add back to the
score difference from the chosen background. TriageIQ groups each category ID as one player and samples a
fixed number of feature orders.

**Background.** The examples used as the comparison point change the answer to "what mattered?" TriageIQ's
planned background is a fixed, reproducible sample of 2024 serving/demo complaints, using only their intake
text and features; no outcomes are read. The current complaint is never compared with an artificial average
category ID.

**Sentence masking.** The planned text explanation removes a sentence or a redaction marker and re-scores the
remaining words. Removed text can make an unusual sentence, so this is a test of model sensitivity, not a
guarantee that the consumer's wording was the real-world cause.

**Faithfulness checks.** Additivity checks whether contributions reconstruct the same model score.
Deletion/insertion checks compare removing or restoring highly attributed sentences with random ones.
Stability checks whether attributions persist when the background sample or random seed changes. These
checks are required before publishing explanations.

**Status:** implementation is opt-in and disabled until the split ONNX graphs pass parity against the deployed
graph and the server's latency and memory are measured. The offline scripts cover SHAP, LIME, Integrated
Gradients, additivity, deletion/insertion and stability; no explanation-quality results have been recorded yet.

---

## 8. Data design for ML

**Temporal split.** Train up to Sep 2023, validate Oct–Dec 2023, test on 2024: test on the future, never a random
sample.

**Data leakage, and how each kind was blocked:**
- **target leakage:** the outcome lives in a separate event table, and post-intake columns are never features;
- **future leakage:** features are point-in-time;
- **group leakage:** the duplicate-text rule;
- **preprocessing leakage:** scalers are fitted on train only.

**Class imbalance.** Payouts are about 2–3% of complaints. The options:
- **undersampling (used):** every payout + 3 non-payouts each, giving 62,940 training rows, 25% payouts;
- **weighted loss on all rows:** about 34× weight on payouts;
- **SMOTE:** rejected for text, since you can't blend two complaints into a real one;
- **focal loss.**

**Case-control sampling + prior correction.** Keeping 8.92% of non-payouts inflates the odds of a payout; adding
ln(0.0892) = **−2.4165** to every logit undoes it exactly.

**Learning-curve test ("halve it before you double it").** Trained on half the data: 0.8166 vs 0.8224, inside the
noise. So more data wouldn't help, and that saved about 3.5 GPU-hours.

---

## 9. Feature engineering (the SQL track record)

**Point-in-time features.** Each complaint's company / issue / product history *as of the day it arrived*, using
window functions. Outcomes count only after a 60-day lag.

**Target encoding.** A category (a company) is replaced by its average outcome. It's powerful, and it leaks easily:
fitted on the resampled training set it cost 0.032 AUC, so we compute it over all 4.8M complaints instead.

**Smoothing / shrinkage (empirical Bayes).** (payouts + K × prior) ÷ (complaints + K). Small samples lean on the
product's rate. **K = 5 was measured:** for company × issue with < 200 history, AUC was 0.8690 at K = 0 vs 0.8901 at
K = 5.

**Cold-start flags.** "No history" ≠ "never pays"; an explicit 0/1 flag says which it is.

**Multi-hot flags.** "Older American" and "Servicemember" can both be ticked, so they're two independent flags,
not one category.

**Ratios and log-ratios instead of raw counts.** Shares of national volume; trend = ln((last 90 days + 1) ÷ (the 90
before + 1)).

**Covariate shift ("clock" features).** Counts that grow with the calendar teach *when*, not *what*. We removed
them, and a test confirmed it: 19 inputs 0.7500 vs 19 + 16 removed 0.7499.

**Feature selection by meaning, then by test.** 48 stored columns, 19 model inputs.

**Where:** `sql/03_features/`, `sql/03_features/feature_dictionary.md`.

---

## 10. Evaluation — every metric

| metric | what it answers | our value (2024 test, DistilBERT fusion) | why |
|---|---|---|---|
| **ROC-AUC, within company** (headline) | inside one company's queue, how often a payout is ranked above a non-payout | **0.8224** | what a deployed bank experiences |
| ROC-AUC, within product × issue | the same, holding product and issue fixed | 0.9492 | how well the case itself is read |
| ROC-AUC, pooled | any two complaints | 0.9731 | context only; inflated by easy between-group differences |
| **PR-AUC** (average precision) | how clean the top of the list is | 0.4395 | focuses on the rare class; also picks the best epoch |
| **Recall at top-k** | share of payouts in the riskiest k% | top 10% → **92.6%** | the business sentence |
| **Bootstrap 95% range** | is a difference real or luck? | 0.815–0.830 | 1,000 resamples inside each company |
| **Slice AUC** | do complaints cut at 512 suffer? | cut 0.9614 vs fits 0.9741 | equal to a read-everything model |
| Single-feature AUC | signal in one input alone | used to screen features | |
| **Calibration**: mean predicted vs actual | do "3%" predictions pay about 3%? | 3.11% vs 2.31% before the fix | probabilities shown to people |
| **Reliability table** (10 equal-count buckets) | calibration bucket by bucket | riskiest bucket 28.6% → 21.2% predicted vs 19.7% actual | |
| **Brier score** | mean squared error of probabilities | 0.01474 (Platt fitted on Jan–Jun 2024, scored on Jul–Dec) | also chose K |
| **Log loss** | penalty for confident mistakes | 0.04939 (same check) | |
| **ECE** (expected calibration error) | average gap, predicted vs actual | 0.00172 (same check) | |
| **Spearman rank correlation** | did the ONNX export keep the ranking? | 0.999999 vs the GPU model | serving parity |

**Deliberately not used:**
- **Accuracy:** "never pays" scores 97.7% on 2024 (the *accuracy paradox*).
- **F1 / F2:** they need a fixed cut-off. F2 is the candidate if a desk ever sets one from its capacity.

**Ablation study.** Each part removed in turn, within-company:

| model | score |
|---|---|
| track record only, small network | 0.7537 |
| text only, DistilBERT | 0.8036 |
| TF-IDF fusion | 0.8115 |
| **DistilBERT fusion** | **0.8224** |

**Error analysis.** Reading the worst mistakes:
- the misses are "surprise" payouts on complaint types that almost never pay;
- the false alarms read exactly like refund cases.

Either way, the deciding fact is a company decision that isn't in the text.

**Information overlap.** The dollar-amount feature lifted the track-record-only model (0.7537 → 0.7779) but not
the fusion (0.8195 vs 0.8224): DistilBERT already reads "$760" in the text.

---

## 11. Calibration and decisions

**Prior correction** fixes the sampling only.

**Platt scaling** sigmoid(a·z + b) fixes over-confidence (a = 0.70) and drift (b = −0.76). It was fitted on the
newest data (Jul–Dec 2024) and checked on later months.

**Intercept shift** (b only) was compared too. It fixes the drift but not the over-confidence.

**Temperature scaling** (a only) is the usual neural-network variant. It wasn't needed, because Platt covers it.

**Isotonic regression** was rejected: its ties lowered the ranking score (0.8296 → 0.8290).

**Monotone transforms never change AUC,** because the order is kept.

**Threshold from capacity, not 0.5.** The riskiest 10% go to seniors: p ≥ 3.67%. In 2024, 21.4% of that group paid,
vs 0.19% in the template group.

**Drift types met:**
- **label / prior shift:** payouts 2.74% → 1.71% (complaints with text, 2022 → 2024), handled by recalibration;
- **covariate shift:** the clock features;
- **concept drift:** a company changing its policy, which shows up in error analysis.

**Where:** `training/serving/calibrate.py`.

---

## 12. Serving ML

**Model export to ONNX.** The network's recipe plus its weights in one file; it runs on ONNX Runtime with no
PyTorch. It matches PyTorch to 1e-5.

**Quantisation (int8).** Store weights in 8 bits. Considered, not needed: the server has RAM to spare
(api ~550 MiB of ~1.8 GB) and 0.2 s is fast enough for triage.

**Train/serve skew test.** The live track record was rebuilt for 3 past days. All 19 inputs matched training
exactly, over 20,773 complaints.

**Latency.** p50 / p95, the typical and the slow requests. Model alone: 20 / 81 ms on the M4 Mac, 218 / 803 ms
on the 2-core cloud server (300 real 2024 complaints). The feature lookup is under 1 ms, so the model is
nearly the whole wait.

**Where:** `training/serving/`, `sql/06_serving/`, `api/`.

---

## 13. Planned, not built

**Embeddings and similar-complaint retrieval** (pgvector): show the 5 most similar past complaints and how they
ended. It's in the original design, not yet built.

**A recency-weighted retrain on 2025 data:** time-decay sample weights, a champion vs challenger comparison.
