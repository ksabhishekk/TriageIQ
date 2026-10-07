# TriageIQ — Blog log

Raw material for the project blog, written for **readers with little tech background**: plain words,
everyday comparisons, and the reason behind every choice. Technical names appear only as small
*(tech: …)* tags, so the blog can keep or drop them. New entries go at the bottom of their section.

**How numbers are described.** Every number says which group of complaints it comes from:
- **all complaints** — the 4,826,564 complaints from 2022–2024
- **complaints with a story** — the 1,639,068 where the customer's own written account is published
- **the training set** — the 292,940 complaints used to train and test the AI (v3; an earlier version had 301,460)

*(Internally these are called F1 / F2 / F3 — never use those names in the blog; readers will think of
the "F1-score".)*

**How scores are described.** "Score out of 100" = take one complaint that ended with a payout and one
that didn't; how often does the model rank the payout one higher? 50 is a coin flip, 100 is perfect.
*(tech: ROC-AUC × 100)*

---

## 1. The story so far

1. **Getting to know the data.** The file was 9.2 GB — too big to open at once — so we read it in
   slices. We proved the complaints were written by real people, not a machine, and built a first
   training set.
2. **The big change of plan.** The original design invented customers with purchase histories. The
   public data has no customers at all — every complaint is anonymous — so the plan switched to tracking
   *companies* and how they handle each kind of problem.
3. **The honest-score moment.** The first scores looked amazing, but mostly because the model knew
   *which product* or *which company* a complaint was about. We changed how we test.
4. **Designing the database.** 17 decisions, each tied to one idea you could explain on a whiteboard.
5. **Loading the data.** 17.4 million raw complaints copied in, then sorted into linked tables. ✅ Done.
6. **The track records.** Every company's history, as it stood on the day each complaint arrived, built in the
   database for all 4.8 million complaints and checked by hand-recounts and a "no peeking" test. ✅ Done.
7. **The AI reader.** DistilBERT trained on a free GPU, combined with the track record: inside one company it
   puts a paid complaint above an unpaid one 82 times in 100 (2024 exam). ✅ Done.
8. **Putting it online.** Honest percentages, a small cloud server, a website anyone can open:
   https://triageiq-mu.vercel.app. ✅ Done (5 October 2026).
9. **Writing it up.** The essay "Finding the two percent": https://bajaj30.github.io/TriageIQ/. ✅ Done.

---

## 2. Ideas we threw away — and why

1. **Inventing customers.** The complaints are anonymous, so there was no customer to attach a history
   to — and invented data can't be checked by anyone. → *Real data only; history is tracked per company
   and per kind of problem.*
2. **"Balancing" the data by capping each product at 16,000 complaints.** It quietly more than tripled
   the payout rate, from 2.16% to 7.5% (complaints with a story) — like judging how often people win the
   lottery by only asking people outside the lottery office. → *Keep the natural mix. When we need more
   payout examples, we sample in a way we can mathematically undo later.* (tech: case-control sampling)
3. **Deleting every duplicate complaint.** Real people copy-paste the same templates from online forums —
   that's genuine behaviour, not an error. But one template appeared 30,110 times. → *Keep at most 25
   copies of any text* (duplicates fell from 19.8% to 0.3%).
4. **A simple rule: "products that rarely pay out get a template reply".** Student-loan complaints pay
   out only about 1 in 90 times (all complaints), so the rule would send a template to someone whose
   $200,000 loan was never paid out. And the written words still sorted complaints well *inside* such
   products (early tests: mortgages 81, vehicle loans 77 out of 100). → *Score every complaint.*
5. **Predicting "any relief" — money or not.** The company's history alone predicted it so well that the
   AI reader would have been decoration. → *Predict money relief only.*
6. **Predicting how badly the customer was hurt.** No such label exists anywhere, so there's nothing to
   check the model against. → *Predict the company's cost instead — optionally multiplied by the amount
   the customer claims.*
7. **Throwing away the 3.2 million complaints with no written story.** They hold 25,577 payouts — 42% of
   all payouts (all complaints). Without them every company's track record is wrong, by a different amount
   for each company. A single summary number can't replace them either, because a track record is
   different on every date. → *Keep every complaint.*
8. **Moving the complaint date out of the main table because thousands of complaints share a date.** The
   table's rule is "one row per complaint" — that's about rows, not about values repeating. Company names
   repeat too. → *The date stays, and is also copied into the event history.* (tech: grain)
9. **Treating the complaint text like a lookup list.** Each text belongs to exactly one complaint — it's
   an attachment, not a list. → *A side table that holds the text.* (tech: extension table, not a dimension)
10. **Assuming each sub-category belongs to one category.** False: the same sub-category name appears
    under several categories (29 sub-issue names sit under 2 or more issues; 87.5% of all complaints carry
    a sub-product name that's shared). → *A sub-category is identified by its name AND its parent.*
11. **Treating "issue" as a sub-category of "product".** 51 of 93 issues show up under several products.
    → *Two separate lists.*
12. **Matching renamed products by name.** One old product ("Credit card or prepaid card") split into two
    new ones — a name alone can't say which. → *Route each complaint by its sub-product; 14 product names
    become 11.*
13. **Leaving "no sub-issue" blank.** Blanks make 122,207 complaints silently vanish whenever tables are
    combined. → *An explicit "(not specified)" entry.*
14. **Storing the complaint date with a time of day.** The source only records the day; a time would
    invent a fake "midnight". → *Day only.*
15. **Trusting the database's default "look back" setting.** When many complaints share a day, the
    default quietly includes the complaint's *own* outcome in its own history — the answer key leaks into
    the exam. → *Three explicit look-back rules, chosen by what each number means.* (tech: window frames)
16. **One tie-breaking rule for every look-back.** The database refuses it for date-range look-backs.
    → *Pick the rule per feature.*
17. **Working out company payout rates from the training sample.** The sample is 25% payouts, reality
    is about 3.4% — the rates came out distorted and the history-only score lost about 3 points out of 100.
    → *Compute rates in the database, over all 4.8 million complaints.*
18. **Switching now to an AI reader that handles very long texts** (ModernBERT and similar). It's 2.3×
    bigger, its speed trick needs newer GPUs than the free ones, and payouts peak at medium-length
    complaints then *fall*. → *DistilBERT, reading up to 512 word-pieces. For the 8% of complaints that
    are longer: keep the beginning + the end, and check separately how well the model does on them.
    Only if they suffer: read them in 2–3 pieces, and as a last resort, a bigger model.*
19. **The textbook training speed-up** (pad each batch only to its longest text). Measured: no gain at
    all (1.00×) — almost every batch contains one very long complaint. → *Group similar-length complaints
    together: 2.23× less wasted work.*
20. **Google Cloud's $300 free credits for GPUs.** Free-trial accounts have historically been blocked from
    GPUs, and upgrading allows real charges (not re-checked). → *Kaggle's free GPU.*
21. **Using the database already installed on the laptop.** It clashed with the new one, and the project
    must rebuild from a single file. → *A fresh database in Docker.*
22. **Fixing the renamed issue later, inside the analysis.** "Issue" would then mean two different things
    in two places. Nothing depended on it yet, so fixing it at the source took minutes. → *A translation
    table in the database.*
23. **Leaving out complaints the company never answered.** The assistant suggested it, since their final
    outcome is unknown. Rejected: the question is "did this cost the company money?" — and for those
    2,785 complaints no money was paid. → *Count them as "no payout"; only the 19 truly blank outcomes
    are left out.*
24. **Shrinking long complaints with classic text cleanup** — reducing words to their dictionary form
    ("charged" → "charge") or deleting common words. We measured before trying: even a *perfect*
    dictionary-form tool would squeeze only 22% of the long complaints under the limit, and it hands the
    AI unnatural sentences ("I was charge twice"). Deleting common words fits 55% — but the list of
    "common words" includes *not, no, never, nothing*: "I did **not** authorise this charge" becomes
    "authorise charge". → *Only collapse the CFPB's XXXX privacy blanks (meaning survives), keep the
    beginning + end, and read the longest ones in pieces if needed.* (tech: lemmatization, stopword removal)

---

## 3. Surprises

1. **One model, three honest-sounding scores.** On 150,000 complaints from 2024 (the training set's
   test part): **96** when comparing any two complaints, **93** when both are about the same product and
   problem, **80** when both come from the same company. The easy test rewards knowing that credit-card
   complaints pay out more than credit-report ones — true, but useless to a bank sorting its *own* inbox.
2. **Which matters more — the words or the history? It depends who's asking.** Same product and problem:
   history wins (91 vs 89). Inside one company: the words win (79 vs 75). Together they beat both (93, 80).
3. **The answer was almost decided before anyone read the complaint.** In early tests, knowing only the
   product scored 96; knowing only the company scored 98. That's exactly why we test *inside* groups.
4. **The complaints aren't where the money is.** Credit reports: 83% of all complaints, 3% of payouts.
   Credit cards + bank accounts: 6% of complaints, 76% of payouts (all complaints).
5. **Complaints the AI can't read still matter.** 42% of all payouts sit in complaints with no written story.
6. **Payouts are getting rarer every year.** 2.7% → 2.5% → 1.7% (complaints with a story, 2022 → 2024).
   Over the whole public database it's 9.09% in 2012 down to 0.48% in 2025. The final exam is harder than
   the lessons — a test on the future shows this; a random test would hide it.
7. **The regulator changed its own categories halfway through.** In August 2023 the CFPB renamed and
   split products. Loaded naively, credit-report history restarts from zero weeks before our test period.
   1,334,958 complaints were re-labelled (all complaints).
8. **Ties everywhere.** On 43% of the days a company received complaints, it received more than one —
   once, 4,245 in a single day. "The complaints before this one" means nothing without a rule.
9. **Copy-paste was proof of real people.** The most repeated opening line turned out to be legal wording
   (from the Fair Credit Reporting Act) that consumers copy from credit-repair forums — not a machine template.
10. **The database doesn't make the AI faster — it makes its job easier.** Training time is the same. But
    without the database's company histories, the AI would have to guess the company from the text and
    memorise the payout habits of 4,946 companies from just 71,460 examples.
11. **"Reproducible" was a claim, not a fact.** The headline numbers came from scripts nobody saved, and
    the project had no version history for weeks. Re-running on the real training set dropped the
    history-only score from 94 to 91.
12. **Same company, different capital letters.** 4 companies appeared twice ("ATM OPS Inc" / "ATM OPS
    INC") and were merged.
13. **"Missing" wasn't really missing.** All 122,207 complaints with no sub-issue (all complaints) have a
    reason: 45 issues never have one, 3 lack it only for payday loans, and 4 mortgage/payment issues only
    got sub-issues when the form changed in August 2023. For those 4, a blank secretly means "filed
    before August 2023" — a model would learn the *date*, not the problem.
14. **The time window we picked had a trap inside it.** We picked 2022–2024 to balance *how much* data we
    had against *how recent* it was (payout rates keep falling, so old years look different) — not because
    it had the most payouts; 2025 has more. But the CFPB changed its complaint form in late August 2023,
    right in the middle. Products were renamed and split — and, found only *after* loading the main table,
    an **issue** was renamed too: "…a credit reporting company's investigation…" became "…a company's
    investigation…". Same day, same products, same sub-issues: 893,566 complaints under the two names
    (18.5% of all complaints). **It wasn't a leak** — both names are known the moment a complaint arrives.
    It was a *history reset*: the most common issue's track record would have restarted from zero five
    weeks before testing began. **The fix was a small design change** — one translation rule, one extra
    column for the original name — then a full rebuild (337,252 complaints re-labelled). It took minutes,
    only because nothing depended on the old layout yet.
15. **Some companies simply ignore the regulator.** 2,785 complaints are marked "Untimely response"
    (all complaints): the company never answered, so no final outcome exists at all. It's almost entirely
    tiny companies — those with fewer than 10 complaints ignored 13.27% of them; companies with over 1,000
    complaints ignored 21 out of 4.6 million. 77 companies with at least 5 complaints never answered a
    single one. "Late" is different: 18,374 answers came late, and 15,589 of those still closed normally
    (637 even with money paid).
16. **The AI's 512-piece reading limit is roomier than it sounds.** It reads in word-pieces, not words —
    but the typical complaint needs only 1.24 pieces per word, so 512 pieces ≈ 410 words. 92% of
    complaints fit completely; only 8% get cut (random sample of 20,000 complaints with a story). The
    CFPB's "XXXX" privacy blanks cost extra pieces: 1.28 per word with them, 1.17 without.
17. **One in six things the AI reads is a privacy blank.** The CFPB hides names, dates and account
    numbers as "XXXX" — and the AI splits each one into 2 pieces (a hidden date costs 6). Across a random
    20,000 complaints with a story, 15.5% of everything the AI would read is blanks; for 8.6% of
    complaints it's at least 40%. The blanks don't confuse it much — it learns to skip them — but they
    crowd real words out of its 512-piece window. Fix: one short marker per blank. Not deleting them,
    because *where* the blanks are is a clue too: a formal dispute full of hidden account numbers reads
    differently from a short angry story.

18. **"Mostly empty" doesn't mean useless.** The tags column is blank for 94.49% of all complaints —
    it looks like junk. But blank means "no tag", not "missing": consumers tick a box if they're a
    servicemember or an older American (62+). Older Americans' complaints end in a payout 10.08% of the
    time vs 1.06% untagged — and it isn't just which products they complain about: on credit cards it's
    26.41% vs 13.81%. Always ask what a blank *means* before throwing a column away.
19. **The AI studies only 71,460 complaints — but the history never shrinks.**
    *Why we shrank:* free GPU time is limited, and payouts are rare (2.16% of complaints with a story).
    Training on all 1.6 million would take far too long, and the AI would see almost nothing but "no
    payout". So we kept **every** payout and 3 non-payouts for each — 71,460 complaints, a quarter of
    them payouts. *(tech: case-control sampling)*
    *What it costs:* that's thin, company by company. In the training set's learning part, only 410 of
    1,627 companies have even one payout, only 70 have ten or more, and the top 10 companies hold 69.6%
    of all payouts. From that, the AI could never learn each company's habits by itself.
    *How the history stays safe:* every company's track record is built in the database from **all 4.8
    million complaints — before any sampling happens.** Each training complaint then carries a summary
    card about its company, written from the full history. Sampling decides how many complaints the AI
    *reads*; it never changes how much the history *knows*. *(tech: the two-tier rule — features over the
    full population, training on a sample)*
    *Proof it matters:* early on, the company rates were computed from the sample itself. The sample is a
    quarter payouts instead of about 3%, so every rate came out inflated — and the history-only score lost
    about 3 points out of 100 (idea #17 above).
20. **An explanation is not the reason.** We can test which words and track-record values moved the model's
    score, but that does not tell us why a company chose to refund someone. Hiding a sentence and rescoring
    is also an imperfect experiment: the remaining story may sound unlike anything the model saw in training.
    The explanation stays off until the split model is proven to give the same score and the tests show that
    its highlighted parts matter more than random ones.
---

## 4. Classroom ideas that turned out to matter

- **Word variety** *(tech: type-token ratio)* — how many *different* words a text uses. Machine-written text
  tends to reuse the same words. Twist: the raw data scored 0.065, *below* our own 0.08 "suspicious" line;
  it rose to 0.081 after removing copy-paste floods. The "real people" verdict rested on the other tests.
- **Longer texts reuse more words** *(tech: Heaps' law)* — so word variety only compares fairly at the same
  length. We always measured on exactly 100,000 words.
- **How uneven sentence lengths are** *(tech: coefficient of variation)* — people write unevenly (0.94);
  machine text is suspiciously regular (below 0.35).
- **Accuracy lies about rare events.** A model that always says "no payout" is right almost 99% of the
  time (all complaints) — and useless. We measure how well it *finds* the rare payouts instead, and test on
  data with the real-world rarity. *(tech: PR-AUC, not accuracy)*
- **Missing a payout is worse than a wasted review.** So catching payouts ("recall") comes first. A score
  that weights catching twice as much as precision is a candidate for setting the final cut-off — not used
  yet. *(tech: F2-score)*
- **Sampling you can undo** — keep every payout plus 3 non-payouts for each, then correct the predictions by
  a known amount afterwards. *(tech: case-control sampling; correction −2.2572 on the log-odds scale, from
  keeping 10.46% of non-payouts)*
- **A group average can hide what happens inside the group** *(tech: between- vs within-group variation, a
  cousin of Simpson's paradox)* — the 96 / 93 / 80 scores.
- **No peeking at the future** *(tech: data leakage)* — use only what was known the day a complaint arrived.
- **Test on the future, not a random sample** *(tech: temporal split)* — learn on 2022–Sep 2023, tune on
  late 2023, final exam on 2024.
- **A leak and a shift are different problems** — a leak uses information you wouldn't have yet; a shift
  means the future simply looks different from the past. The 2023 form change was a shift, fixed by
  translating old names into new ones.
- **Don't trust a small sample's average** *(tech: smoothing / shrinkage, empirical Bayes)* — a company with
  3 complaints and 1 payout isn't really a "33% payer"; that's luck. The fix is one line:

  > **smoothed rate = (payouts + K × product rate) / (complaints + K)**

  Read it as: *before looking at a company, pretend it already has K imaginary complaints that paid out at
  its product's usual rate — then add its real ones.* A company with 3 complaints is mostly imaginary
  complaints, so it leans on its product's rate. A company with 5,000 real complaints drowns the imaginary
  ones out and speaks for itself. K sets how many imaginary complaints: small K trusts small samples, big K
  ignores them. (The default is the product's rate, not the overall rate — the overall rate is mostly
  credit reports, which almost never pay.)

  | 3 complaints, 1 paid, a credit card company (product rate ≈ 15%) | smoothed rate |
  |---|---|
  | K = 0 — trust the 3 complaints completely | 33.3% |
  | K = 5 | 21.9% |
  | K = 50 | 16.0% |
  | K = 200 — ignore the company's own history | 15.3% |

  **We measured K instead of guessing it.** On complaints from Oct–Dec 2023 (never the 2024 final exam),
  we tried 12 values and asked: how well does the smoothed rate alone separate complaints that paid from
  those that didn't? For companies with little history, raw rates scored 87 out of 100; smoothing with K = 5
  scored 89; K = 50 (our first guess) 88; K = 500 only 85. Anything from 2 to 7 was about equally good —
  the big win is smoothing *at all*. We picked **K = 5**: just 5 imaginary complaints are enough.

  One puzzle left: on the very first days of 2022, *no* outcome is known yet — not even a product's rate.
  So the default for the default is last year's overall payout rate (2.86% in 2021), which was public
  before our data starts — it can't give anything away. After 60 days every product has its own rate.
- **A translation dictionary for renamed categories** *(tech: crosswalk / mapping table)* — the same tool
  governments use when medical or industry codes get renumbered. A two-column list "old name → new name";
  every complaint looks itself up in it: found → take the new name, not found → keep its own.
- **Database design basics** *(tech: dimensional modelling)* — decide what one row means, keep lists apart
  from events, give every item a stable id number.
- **"Unknown" is not "no" — usually** *(tech: NULL)* — the database keeps blanks as blanks. For the label we
  made a deliberate exception: 19 complaints with no recorded outcome count as "no payout" — 19 out of
  4.8 million can't move anything, and it keeps the rule simple: every complaint gets an answer.
- **Let the database refuse nonsense** *(tech: foreign keys, CHECK constraints)* — it rejects, say, a
  mortgage sub-product filed under credit cards.
- **Look-back windows** *(tech: window functions)* — "how often did this company pay in everything *before*
  today?"
- **Run it twice, get the same result** *(tech: idempotency)* — every load can be safely re-run.
- **Names that change over time** *(tech: crosswalk / concordance table; slowly changing dimension, type 1)* —
  when the regulator renamed categories mid-way, we kept a small translation table (old name → today's
  name) so every complaint shows today's name and each company's history stays in one piece. The
  original name is kept alongside, so nothing is lost.
- **When one box can hold two answers** *(tech: one-hot vs multi-hot encoding)* — the tags column can say
  "Older American", "Servicemember", or both. Treating "both" as a separate, unrelated category would
  hide that it *is* an older American. Two yes/no switches — one per fact — let "both" simply flip both.
- **Four ways to handle a rare answer** *(tech: class imbalance)* — when only 2–3 in 100 complaints pay
  (or 3 in 100 bank accounts are mule accounts), a model can score 97% by always saying "no". The fixes:
  (1) throw away most "no" examples (what we did first — fast, but wasteful); (2) keep everything and
  make each missed "yes" cost ~34× more *(tech: weighted loss)*; (3) invent extra "yes" examples by
  blending real ones *(tech: SMOTE)* — great for numbers like transactions, useless for text, because
  half of one complaint glued to half of another isn't a complaint; (4) make the model focus on what it
  gets wrong *(tech: focal loss)*.
- **Is the database worth it, or just use a better AI?** — first answer: the same AI reading only the
  story scored 80 out of 100 inside a company; reading the story *plus* the database's track records, 82.
  The ± ranges don't overlap, so it's not luck. On the other tests the gap is bigger (92 vs 95). The
  track records add something the text alone can't. (Still to test: a much bigger AI, text only.)
- **Halve it before you double it** — before spending 3.5 GPU-hours on 8× more data, train once on *half*
  the data (15 minutes). If losing half hurts, gaining more will probably help; if it doesn't, it won't.
- **Why the AI gets 19 clues, not 40 — tested** — we fed a simple model the 19 chosen clues, then the 19
  plus the 16 left out. The extra 16 added exactly nothing. They are mostly *clocks*: a company's total
  complaint count was ~108,000 on average in the training years and ~588,000 in 2024. A model trained on
  the first number has never seen the second — it would be reading the calendar, not the complaint.
- **The verdict on "database + AI" vs "AI alone"** — reading the riskiest 10% of 2024 complaints, the AI
  alone catches 88.8% of payouts; the AI plus the database's track records catches 92.6% — about 130 more
  paying complaints found for the same amount of reading. The combination is the model we keep.
- **One new clue beat 480,000 extra examples** — instead of more rows, we added two new *clues*: whether
  the complaint mentions a dollar amount (and how big), and how a company has behaved in just the last
  year. Tested on the database clues alone, inside a company: 75.4 → 77.8, and the ± ranges don't
  overlap. Testing each clue separately showed the **amount** did all of it; the "last 12 months" clue
  added nothing. Oddly, *small* amounts pay most: under $100 → 11.7% payouts, over $10,000 → 4.1% —
  a $35 fee is easy to refund, a $20,000 loan dispute isn't.
- **…but the AI had already read it** — retrained with the amount clue, the full AI scored the same (82.0
  vs 82.2, well inside the ± range). It makes sense: the clue helped the database-only model because that
  model can't read; the AI reads "$760" in the complaint itself. A clue only helps if the model doesn't
  already have it. We kept the simpler 19-clue version.
- **Why we didn't feed the AI 8 times more data — and how 15 minutes proved it** — the AI trained on
  62,940 complaints: every one of the 15,735 that paid, plus 3 that didn't for each. About 480,000 more
  "didn't pay" complaints were left on the table to save GPU time. The textbook way to use them all is a
  *weighted* lesson: keep every complaint, but make each missed payout count about 34 times more than a
  false alarm, so the rare payouts don't get drowned out *(tech: weighted loss)*. That would have taken
  ~3.5 GPU-hours. Before spending them, we ran the cheap test in reverse: train on **half** the data
  (15 minutes). If losing half the examples hurts a lot, gaining more should help a lot. It didn't: the
  score went from 82.2 to 81.7 inside a company — well within the ± range, i.e. about the size of luck.
  **Why more data couldn't help much:** (1) the extra complaints are all "didn't pay" — there are no
  new payouts to learn from, ever; (2) the AI's mistakes come from information that isn't in the
  complaint at all (the company's internal decision), and more copies of the same kind of complaint
  can't supply it. More of the same isn't more information. Fifteen minutes of testing saved three and
  a half hours of training — and pointed us at what is actually missing: new *clues*, not more *rows*.
- **Why a longer-reading AI wouldn't help either** — DistilBERT reads the first 512 word-pieces (~410
  words) and skips the rest; 8.4% of 2024 complaints are longer. If the deciding words were hiding in
  those tails, a model that reads *every* word would beat it on exactly those complaints. It doesn't —
  both score 96.1 on them. So a bigger, slower long-context model would buy nothing here.
- **Every score gets a ± range** — the honest score is averaged over just 24 companies, so we re-draw the
  test complaints 1,000 times and report where the score lands 95% of the time: 0.8224 is really
  "somewhere between 0.815 and 0.830". Two models whose ranges overlap may differ only by luck.
- **AI readers have a length limit** — DistilBERT reads at most 512 word-pieces.
- **Old text-cleanup tricks can hurt modern AI readers** *(tech: lemmatization, stopword removal)* —
  they were made for models that just count words. A modern reader understands "charged" vs "charge"
  and needs "not"; stripping them saves little and loses meaning.
- **Long tails** — a few extreme cases (a 30,110-copy template, a few very long complaints) can break
  methods built for typical cases.
- **Older free GPUs can't use the newest number format** *(tech: fp16 vs bf16)* — so we use the older one.
- **Making the percentages honest** *(tech: calibration, Platt scaling)* — a good weather forecaster is right
  about *how often*: on days they say "30% rain", it rains about 3 days in 10. Our AI said 3.1% of 2024 complaints
  would pay; 2.3% did. Two numbers fix it — one squeezes the AI's over-confident extremes, one shifts everything
  down for the falling payout rate. Fitted on January–June 2024 and checked on July–December, the corrected
  forecast said 2.22% against a real 2.10% (uncorrected: 2.91%). The order of complaints never changes, so every
  ranking score stays exactly the same. A fancier, free-form correction did fit slightly better — but it lumps
  complaints into ties and quietly lowered the ranking score, so the simple two-number fix won. The rule for the
  desk: the riskiest 10% of complaints go to a senior; in 2024 that group paid out 21% of the time, the rest 0.2%.
  **Why not tune it on the late-2023 practice data?** Payouts kept falling: October–December 2023 paid 3.50%,
  2024 paid 2.31%. A correction tuned on 2023 would still forecast too high for 2024 — and forecasting too high
  is exactly the problem we were fixing. So the correction is tuned on the newest data available (the second
  half of 2024), and checked on months that came *after* the ones it was tuned on.
- **A scoreboard instead of a replay** *(tech: online features, train/serve skew test)* — to train, the AI needed
  every company's track record as it stood on the day of *each* past complaint: the database replays 4.8 million
  complaints to get that. A live complaint needs only *today's* record. So the live system keeps a small
  scoreboard — for each company and kind of problem, how many complaints and payouts so far — and a calculator
  that applies the same formulas. Two ways of computing the same numbers is dangerous: if they disagree even
  slightly, the AI gets inputs it never learned on, and nothing warns you. So we rebuilt the scoreboard as of three
  real past days — including the busiest day in the data — and compared all 19 numbers for 20,773 complaints
  with what training had used: identical, to the last digit. Everything the live system needs fits in one
  57 MB file; the 16 GB database stays at home.
- **A waiter between you and the kitchen** *(tech: a FastAPI web service)* — you type a company, the kind of
  problem and the complaint in your own words; the service looks up the company's track record, cleans the text
  exactly the way the training text was cleaned, asks the AI, and answers "31.7% chance of a payout — send it
  to a senior analyst" in about 40 thousandths of a second. It also shows real 2024 complaints next to what the
  company actually did — including the ones the AI got wrong (a credit-card complaint it gave 2%, which was
  paid). The honest check: 300 real complaints scored through the service matched the scores from the original
  test to within 0.02 of a percentage point, and every one went to the same desk.
- **Packing it into shipping containers** *(tech: Docker, Docker Compose)* — "it works on my laptop" isn't a
  deployment. So the whole service was packed into two sealed boxes: one holds the code, the AI model and exactly
  the 9 software packages it needs; the other is a small database that, the very first time it starts, fills
  itself from the 57 MB file. One command starts both, in the right order — the service waits until the database
  actually answers. From empty to ready took about ten seconds, uses about half a gigabyte of memory, and gave
  the same answers as before on 300 real complaints. The database box has no door to the outside world: only the
  service box next to it can talk to it.
- **Live on the internet** *(tech: AWS EC2, NGINX reverse proxy)* — the two boxes now run on a small rented computer
  in Sydney (the closest region this AWS account is allowed to use), for about $21 a month paid from free credits. A
  receptionist program sits at the only open door: it passes visitors to the service, turns away anyone sending more
  than a few requests a second, and keeps the database completely out of reach. Checked from outside: 200 real 2024
  complaints got the same scores as in the original test, and every one went to the same desk.
- **The server that started out of breath** *(tech: burstable instances, CPU credits)* — the cheapest cloud servers
  run at full speed only while they have "credits" saved up, and a brand-new one starts with none. Installing
  everything and a 200-complaint test ran it at a fifth of its speed — half a second per complaint instead of about
  a fifth of a second (measured at full speed the next day). Credits refill whenever it's quiet, which for a demo link is almost always. We kept the setting that can
  never cost extra, instead of the one that is always fast but bills for heavy use.
- **A front door people actually want to open** *(tech: static site on Vercel, reverse-proxy rewrites)* — the API
  page is fine for engineers; a recruiter needs a website. It's styled like a case file — paper, ink, one dark-red
  accent, hidden details shown as real black redaction bars — and it shows the model's misses next to its hits. The
  site lives on Vercel; when you press "Score it", Vercel quietly carries the request to the server in Sydney and
  brings back the answer, so the browser only ever talks to one secure address.
- **A password the server checks on every knock** *(tech: origin secret header, rate limiting by real visitor IP)* —
  the server now ignores everyone except Vercel: Vercel attaches a long secret to every request it forwards, and
  anything without it is turned away. That also lets the server count requests per real visitor (Vercel says who
  they are) instead of per Vercel machine. One embarrassing minute: the secret was 48 characters long, and the
  receptionist's lookup table only had room for shorter keys — it refused to start, and the site was down for about
  three minutes until one setting was raised. The test had used a short practice secret; real values break things
  that practice values don't.
- **The newest complaints can't teach yet** — the public file runs to August 2026, so why not train on the
  freshest data? Because the newest complaints aren't finished: half of those from July–August 2026 have no
  outcome yet, and the customers' written stories are published weeks or months later — only about 3% of 2026
  complaints have one so far. The few already public are an unusual early group (6.5–9.5% paid, against the
  usual 1–2%). Learning from them would be like judging a school year from the first three exams handed back.
  2025 is complete enough to use — that's the next retrain. *(tech: label lag, publication lag, selection bias)*
- **The same AI, without the AI toolkit** *(tech: ONNX export)* — the trained model was saved as a single file
  that a small, fast engine can run on an ordinary processor, no graphics card or training software needed.
  Checked on 1,000 real complaints: identical answers to 5 decimal places. One complaint takes about 20
  thousandths of a second on a laptop (a short one 11, the longest 54).

---

## 5. How the database does the heavy lifting

- **Everything lives in one database** (PostgreSQL, in Docker) — it even survived the laptop's Docker
  stopping overnight.
- **Copy the raw data first, clean it second** — all 17,355,295 complaints went in untouched, like
  photocopying documents before marking them up.
- **Saved filters** *(tech: views)* — "only 2022–2024" and "with names translated" are written once and
  reused everywhere, so no step can quietly use a different definition.
- **Rules kept as data, not buried in code** — two small translation tables (12 product rules, 1 issue
  rule) re-label 1,334,958 and 337,252 complaints. Anyone can open them and check.
- **Every step proves itself** — one file per table, and each file ends by checking its own numbers.
- **Run twice, nothing doubles** — every load can be repeated safely.
- **Shrink first, then match** — 4.8 million rows boil down to 293 unique pairs before any matching.
- **Fail loudly, never quietly** — if a complaint's category can't be found, the whole load stops with an
  error instead of silently dropping it. 4,826,564 complaints went in in 81 seconds.
- **Read once, write three times** — each complaint becomes 3 history rows (received, sent, answered) in
  a single pass: 14,479,692 rows in under 2 minutes.
- **"Nothing happened" can still use up ticket numbers** — re-running the history load added 0 rows but
  used up 14.5 million id numbers, like pulling a deli ticket and then leaving the queue. Harmless — and
  a reason never to rely on an id number meaning anything.
- **Change the design early** — the issue fix meant rebuilding every table, which took minutes. Later, the
  same change would break everything built on top.
- **The database refuses bad data** — 12 out of 12 deliberately wrong inserts were rejected.
- **The answer key lives in exactly one place** *(tech: the label as a view)* — "did it pay?" is one saved
  question, not a column copied into many tables. Every later step asks it the same way, so the answer
  can never mean two different things. It returns all 4.8 million answers in about a second.
- **An index is a book's index** — instead of reading all 4.8 million complaints to find one company's
  history on one problem, the database looks it up: 797 complaints found in 0.19 milliseconds, without
  opening the main table at all. Building all 6 indexes took 11 seconds.
- **The answer sheet lives in one place** — the numbers the final exam checks against come from one file,
  produced by a *separate* program that re-does the whole load in a different language (Python instead of
  SQL). Both arrive at the same 10 load numbers independently — much stronger evidence than one program
  checking itself. And the docs and the checks read the same file, so they can never disagree.
- **The database was doing its homework on the floor** — the first volume check ran for over 11 minutes.
  The reason: by default Postgres gets only 4 MB of memory to sort with, so each sort of 4.8 million
  complaints spilled onto the disk — 28 GB of scratch files. Giving it 256 MB brought the same check down
  to 16 seconds. Same answers, same code — one setting. *(tech: work_mem)*
- **Two methods, one answer** — each volume number was recounted a second, completely different way
  (plain counting instead of window functions) for 22 complaints, including both ends of the busiest
  company-day in the data (4,245 complaints). All 22 matched.
- **Proving there's no peeking** — we secretly flipped the answers of 3,495 complaints from one company's
  busiest day (paid ↔ not paid), then recomputed that company's track record. Complaints on the same day
  and up to 59 days later: not a single digit moved. On day 60, the rate jumped from 0.0034% to 0.34% —
  exactly when those outcomes are allowed to be known. Then we undid the flip. A test that can also
  *fail* is the only kind worth trusting. *(tech: leakage test inside a rolled-back transaction)*
- **The track record works before the AI reads a word** — complaints whose company had paid out over 15%
  of the time on that kind of problem really paid 28.74% of the time; where it had paid up to 1%, only
  0.03% did (all complaints).
- **Share, not size** — complaint volume grew so fast that an average company's 90-day count rose 5.5×
  from 2022 to 2024. A model trained on 2022–23 would think every 2024 company is unusually busy. So we
  also measure each company's *share* of all complaints — that rose only 1.5×. Like comparing a shop's
  share of the market instead of its raw sales in a growing economy.
- **Building 40 clues, using 19** — the finished table stores 40 facts about every complaint, but the model
  reads only 19. The other 21 aren't wrong — they're *clocks* ("how far into the data are we?") or raw
  counts that swell as complaint volume grows. A model trained on 2022–23 would read them as "this is
  2024" rather than "this complaint is risky". Choosing what *not* to feed a model is half the job. The
  list of 19 lives in one saved query, so training and the live service can never disagree about it.
- **Copy-paste complaints almost never get paid** — 849,718 complaints with a unique story paid out 3.84%
  of the time; complaints whose exact text appears 2 or more times paid out 0.05% or less (from April
  2022, complaints with a story). They're mostly form letters disputing credit reports. When we stopped
  the same text from appearing in both the lessons and the final exam, 106,942 exam-side copies went —
  and only 9 of them had been paid. The old exam had been padded with easy, repeated letters.
- **The database's track records beat the old shortcut** — with the old training set, the company
  history was worked out from the training sample itself; with the new one it comes from the database,
  built from all 4.8 million complaints. On the fair test (same product + problem) the history-only
  score went from 91 to 93 out of 100, and history + words from 93 to 95; inside one company, 80 → 81.
  (Different test sets, so read it as "at least as good", not a precise gain.) The three weakest clues
  (trends, quiet days) added almost nothing — 0.1 points at most.
- **Long complaints are harder for everyone** — we feared that cutting long complaints at 512 word-pieces
  hides the important part. So we checked a model that reads *every* word, and one that reads *no* words
  at all (track records only). Both scored lower on long complaints too (8.4% of the 2024 test set,
  which pay out more often: 3.41% vs 2.21%). Long complaints are simply harder — so the AI can only be
  blamed for the cut if it falls clearly behind the read-everything model on exactly those complaints.
- **The AI reader passed the final exam** — on 150,000 complaints from 2024 it had never seen, the
  fine-tuned model (DistilBERT reading the story + the 19 database clues) scored **82 out of 100** inside a
  single company, against 81 for the simple word-counting model. If a senior analyst reads only the
  riskiest 10% of complaints, they catch **93%** of all the complaints that end in a payout. It trained
  in about 30 minutes on a free GPU.
- **The cut at 512 word-pieces turned out harmless** — on the long complaints, the AI scored exactly
  what the read-every-word model scored (both 96). Long complaints are just harder, for everyone.
- **Right about *who*, too high about *how many*** — the AI ranks complaints well, but its percentages
  come out a third too high on 2024 (3.1% predicted vs 2.3% real). It learned from 2022–23, when payouts
  were more common; they keep getting rarer. Ranking is what routing needs, but any probability shown to
  a person must first be re-tuned on recent data.
- **Reading the AI's worst mistakes** — we read the complaints the AI ranked most wrongly inside their own
  company. The *missed payouts* were mostly complaints about credit reports or logging in — the kind that
  almost never pay — where the company paid anyway, probably as a goodwill gesture. The *false alarms*
  read exactly like refund cases: a named fee, an amount, an unauthorised charge — and the company simply
  said no. In both cases the deciding fact lives inside the company, not in the complaint. That is why
  a bigger AI can't simply push the score to 90: the missing information isn't in the text.
- **Money talk is a strong clue** — one in five complaints with a story mentions a dollar amount; they
  end in a payout 7.1% of the time, against 1.0% for the rest, and hold 63% of all payouts. (An earlier
  check had said "no amounts at all" — a typo-level bug in how the search was typed. Always look at a
  few real rows before trusting a zero.)
- **One final exam for the whole load** — a single query checks 21 things at once (every table's size,
  one row per complaint, exactly 3 history rows each, the payout count against the raw file, both
  translation tables) and prints PASS or FAIL for each, in 27 seconds. And we tested the tester: feed it
  one wrong expected number and it flags FAIL. A check that can't fail proves nothing.
- **Why a database at all?** One source of truth: training and the live service read the same numbers
  from the same place. The heavy work on 4.8 million complaints stays in the database; the GPU only sees
  the 292,940-complaint training set.
- **Coming next:** company track records built with look-back windows, the "did it pay?" answer as a saved
  query, and a repeatable way to draw the training sample.

---

## 6. Mistakes and fixes — ours and the AI assistant's

- **Numbers without their group.** Figures from different groups (all years vs 2022–24, a sample vs the
  real training set) got mixed up in the documents, forcing a full correction round. Fix: one generated
  file is now the only source of numbers.
- **The AI assistant said it had done things it hadn't** (a file "saved", a fix "done"). Rule since then:
  never claim something without checking.
- **My own wrong reasons, caught early:** "id numbers are easier to read than names" (backwards — they're
  harder; the real reasons are speed, size and stability); calling the text a "lookup list"; thinking
  repeated dates broke the table; "each sub-category has one parent" (false for 87.5% of complaints).
- **Measuring bugs that looked like data problems:** a search pattern counted the CFPB's "XXXX" privacy
  blanks as template placeholders; 40-character "words" turned out to be web links; one number was measured
  on the wrong group.
- **Tests that assumed fixed id numbers failed** — the database never re-uses an id, even after a failed
  insert. Look things up by name.
- **No version history for weeks**, while the rules said "reproducible".

---

## 7. Loose ends

- Word variety sat below our own threshold on the raw data — explain it properly, or drop that test.
- ~~Smoothing: pull small companies toward their product's rate~~ — done: toward the product's rate, K = 5.
- ~~721 complaint texts appear in more than one of learn / tune / exam~~ — fixed: one text, one split.
- ~~Where to draw the "send to a senior" line~~ — decided: the riskiest 10% (a 3.67% chance or more).
- We *assume* outcomes are known within 60 days — the CFPB never records when a company answered.
- "Payouts peak at medium length (39.6%)" — the group behind that number wasn't recorded; re-check it
  before publishing.

---

## 8. Pictures to build for the web page

> 🔔 **Reminder:** when building the blog web page, turn each diagram into a proper visual —
> hover or click to see what each table or step holds, how big it is, and *why* it exists.

**BUILT (2026-10-05): https://bajaj30.github.io/TriageIQ/** — the essay "Finding the two percent" in `docs/`, made
from this log; the snowflake is interactive there. The table below is what was reused.

**Source of truth: [`README.md`](../README.md).** Copy the diagrams from there when building the page —
never keep a second copy here; two copies drift apart. What to reuse, by README section:

| README section | visual |
|---|---|
| 1. The problem | the routing decision · "needle in a haystack" pie · complaints vs payouts pies |
| 2. The idea | two readers, one score |
| 3. Playing fair | no peeking at the future · learn → tune → final exam |
| 4. The data | 17.4M → 301k funnel · the category translation example |
| **5. Under the hood** | **the snowflake** · the journey from raw file to score · the seven track-record steps |
| 6. Results so far | the three tests (bar chart) · words vs track record |
| 7. What changes | without vs with TriageIQ |
| 8. Where the project is | the roadmap |

---

## 9. How we keep score — every measure, in plain words

The question is never "will this complaint pay — yes or no?" It's **"which complaints should a senior read
first?"** So almost every score is about **ordering**, not about being right or wrong one complaint at a time.
(Precise definitions and the code behind each: `Context/metrics.md`.)

- **The main score — "inside one company"** *(tech: within-company ROC-AUC)*. Take two complaints sent to the
  **same** company, one that ended in a payout and one that didn't. How often does the AI put the payout
  first? 50 = coin flip, 100 = perfect. It's the honest score because a bank only ever sorts its *own*
  complaints. Our best: **82**.
- **Two easier versions of the same score** — compare any two complaints at all (*pooled*: 97 — inflated,
  because "credit cards pay more than credit reports" is an easy point), or two complaints about the same
  product and problem (95). We show all three and trust the lowest.
- **The ± range** *(tech: bootstrap 95% interval)* — the main score rests on just 24 companies, so we
  re-draw the test complaints 1,000 times and report where the score lands 95% of the time. If two
  models' ranges don't overlap, the difference is real; if they overlap a lot, it may be luck.
- **"Read the riskiest 10%, catch 93%"** *(tech: recall at top-k)* — the sentence a manager cares about:
  if seniors only read the 10% of complaints the AI ranks highest, they see 93% of the complaints that
  end up costing money.
- **How clean the top of the list is** *(tech: PR-AUC)* — as you go down the AI's list, how many of the
  flagged complaints really paid? This one punishes false alarms, which the main score is lenient about
  when payouts are rare. It's also how we pick the best training round.
- **Are its percentages honest?** *(tech: calibration)* — when the AI says "3%", do about 3% pay? Ours
  says 3.1% on average for 2024, but only 2.3% paid: right about *who*, a bit high about *how many*,
  because payouts keep getting rarer. Fixable later by re-tuning on recent data.
- **Long vs short complaints** *(tech: slice AUC)* — the same main score, measured separately on complaints
  the AI had to cut short and those it read in full. That's how we learned the cut doesn't hurt.
- **How close the percentages are to reality** *(tech: Brier score)* — used once, to tune how much to trust
  small companies' track records.
- **What we refuse to use: accuracy.** Saying "no payout" to every 2024 complaint is 97.7% accurate — and
  completely useless. A yes/no score like F2 *(recall counted twice as much as precision)* comes later,
  when a real desk chooses where to draw its "send to a senior" line.
