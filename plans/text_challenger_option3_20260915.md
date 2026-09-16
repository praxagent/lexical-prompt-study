# Option 3: stronger prompt text with retained prefix hashes

Date: 2026-09-15. New retrospective exploratory analysis of the already unsealed
A142/A145/A149/A150 calibration corpus. No confirmation panel, new generation,
model download, API request, GPU job, or change to an old fit is included.

The question is whether stronger learned prompt text plus the retained fixed
prefix hash representation predicts the final observed classifier endpoint as
well as an internal readout using the same prompt and token-eight checkpoint. This is a
classifier-proxy prediction experiment, not human accuracy or a deployable
defense. Existing A150 results motivate the comparison; it is not confirmation.

## Population and immutable inputs

Use exactly the pinned A139 calibration topology and A142 original acquisition
manifest: 8,880 observations from 60 request cores, 4,320 in each scaffold
placement and 240 unplaced observations. The fixed A145 row order and five
request-core folds carry forward unchanged. Each fold has 1,776 original rows
and 12 cores. Neither unplaced rows nor any sealed confirmation material enter
fitting. The A149 loader independently verifies the original and continued
scoring chain, original-EOS carryforward, and exactly three unknown final cells.

At token eight, include only original observations with that readout available.
This includes both original-EOS and continuation-selected cohorts; it never
conditions eligibility on survival to token 128. State exclusions explicitly.
Keep placements separate throughout fitting and evaluation.

The full learned prompt-plus-prefix proposal was narrowed before any fit:
original A142 restricted generations are absent locally for all 8,528 eligible
rows. Do not select on later continuation availability to recover a subset.
Exact prompts remain in the pinned calibration topology. The pinned A150 numeric
feature export retains original-regime token-eight vectors for the full cohort.
Privately process prompt text; never open raw generations or generated token IDs.
Verify each prompt SHA against the original receipt. Join each numeric prefix
vector by original receipt SHA and token-eight prefix SHA to the internal readout.
The fixed prefix representation is the original 256-dimensional signed byte
3/4/5-gram FNV-1a hash, L2 normalized, decoded from exactly the original first
eight tokens with special tokens removed. No later token becomes a predictor.
This is a lossy fixed prefix representation, not a learned prefix vocabulary;
the internal representation may retain distinctions removed by that transform.
Do not inspect raw prompts, feature vocabulary or row identifiers in agent
context. The private program emits aggregate metrics and hashes only.

Pin the topology, A145 candidate/folds, A146 receipt manifest, acquisition and
scoring summaries, A148 scoring ZIP, A149 source/freeze, original tokenizer
provenance in the verified numeric export, its extractor/recovery sources, and
A150 preparation receipts. Preserve all four horizon labels,
observed lengths and censoring flags without relabeling. A capped negative is a
finite-prefix negative, not proof of a safe completed response.

## Fixed models and fitting

Fit against known final-observed 1,024-token classifier labels. Use exactly five
comparators: training prevalence, stronger prompt-only text, stronger
prompt text plus the retained token-eight prefix hash, J-lens at token eight
(31 margins), and prompt text plus the prefix hash plus those 31 margins. Text models also include the
six original prompt structural metrics, available from the same prompt.

Stronger prompt text uses word 1–2 grams and character 3–5 grams. TF–IDF
vocabularies and document frequencies are fit only on each training set, with
`min_df=2`, sublinear term frequency and L2 normalization. Fixed vocabulary caps
are 20,000 prompt character features and 10,000 prompt word features. Case and
punctuation are retained. The 256 prefix coordinates retain their original
normalization; they are not learned again or individually standardized. No
hyperparameter search, outcome-ranked token selection, pretrained embedding or
extra semantic lexicon. This strengthens the prompt lexical challenger; it does
not complete comparison against a strong pretrained semantic representation or
a learned prefix vocabulary. All models use identical eligible test rows.

Use unweighted L2 logistic regression, `C=1`, `liblinear`, tolerance `1e-6`,
maximum 1,000 iterations and random seed 20260915. Unweighted fitting preserves
the probability target for log loss and Brier score. Training-only standardize
the numeric internal and structural features. Both classes must occur in training;
convergence warnings make that fit unavailable. No solver fallback or threshold
selection. These are fresh matched probability comparisons, not reinterpretations
of old class-balanced A145/A150 coefficients.

## Splits and bounded first pilot

Primary comparison: the original five held-out request-core folds. All variants
of a core remain together. Structural stress comparisons additionally exclude
one complete observed structure/control family from training and test only that
family on held-out cores. The three placed families are attack-block-mask,
structural-sham and harmless-structured-wrapper. These are structure/control
families within this corpus, not independently sampled attack families. General
transfer to unseen attack families remains unidentifiable.

The first bounded pilot is outer fold zero, separately in both placements, for
ordinary core transfer and held-out structural-sham transfer: four fixed cells,
five models each. The complete prospective matrix additionally contains all
five folds and each of the three held-out structure/control families. Each cell
writes an immutable result and may be resumed by verifying its hashes; never
overwrite or select a favorable cell after looking at results. The initial pilot
does not estimate full-five-fold performance. Any expansion must retain this
matrix and disclose previously inspected outcomes.

Report log loss, Brier score, ROC AUC and average precision, plus positive/known/
unknown counts and censoring, for the final endpoint and descriptive evaluations
against the other fixed horizon labels. Show all intents and direct/safe frames
separately. Rows are repeated measurements, not independent sampling units;
report request-core counts. No row-wise significance claims or bootstrap CI is
made by this initial pilot. A later pooled evaluation needs paired core-level
uncertainty that acknowledges fitted-model variability.

## Execution and outputs

Prepare and hash an inventory before fitting. It records eligible counts,
grouped split availability, input digests, aggregate predictor hashes and exact
runtime/library versions, without raw text or labels at item level. Freeze this
source, tests and protocol in the authorized code backup before any real fit.
Run only through an inventory hash binding with an explicit cell selection.
Use the project Python environment, with compatible CPU numerical dependencies
installed and exact versions recorded before the inventory freeze. Limit BLAS,
OpenMP and numerical threads to at most four, disable tokenizer parallelism,
and keep GPU visibility empty. The monotonic cell deadline stops starting new
fits; run each command under an external hard process timeout as well, because
a fit already executing native code is not interrupted by that deadline.
Maximum document and vocabulary sizes further bound work. Failures emit fixed
error classes rather than raw text.
Vocabulary and model coefficients are not published. Numeric prediction
artifacts, if retained, stay private. Source code and invented tests may be
backed up under the existing authorization; scientific result release is separate.
