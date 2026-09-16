# Frozen prompt-semantic augmentation

Date: 2026-09-15. Prospective specification for a new retrospective exploratory
analysis of the already unsealed calibration corpus. Existing lexical protocols,
fits, inputs and source files remain unchanged. No confirmation panel, new target
generation, response reconstruction, API judging or hyperparameter search enters
this extension. Preparing this source does not itself execute extraction or fits.

## Question, population and comparisons

Does a fixed pretrained representation of the original prompt improve the
lexical-plus-retained-prefix comparator, and does J-lens still add predictive
information after that augmentation? This predicts the existing final-observed
1,024-token classifier endpoint. It does not establish semantic ground truth,
harm prevention, deployable detection or performance against every text model.

Reuse the original text_challenger.py loader, immutable input/receipt chain,
eligible token-eight population, placement separation, five request-core folds,
three held-out structure/control families, known-training-label exclusions and
test denominators. All 40 cells remain: two placements times five folds times
ordinary core transfer plus three structure/control-family exclusions. No row
selection depends on continuation availability or model outcomes. Earlier
inspected lexical results make this exploratory even though these new choices
are frozen before semantic outcomes. Missing raw response-prefix strings remain
missing; the 256-dimensional token-eight hash is not a semantic-prefix embedding.

Add exactly two classifiers per cell, hence 80 new fits in the complete matrix:

1. Existing word/character prompt TF-IDF, retained prefix hash and six structural
   features, concatenated with the frozen 384-dimensional prompt embedding.
2. Those same features plus the 31 token-eight J-lens margins.

Compare against the existing lexical-plus-prefix and corresponding J-lens-added
fits on exactly matched cells and rows. Preserve J-lens-only and prevalence
results as context; do not refit/select the old results to favor this extension.
Fit-budget matching means one prescribed supervised fit per representation and
cell, not equal feature dimensionality or equal historical pretraining compute.

## Frozen encoder and full prompt coverage

Use BAAI/bge-small-en-v1.5 at revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, offline local safetensors, with exact
SHA256 bindings for config.json, model.safetensors, tokenizer.json,
tokenizer_config.json, special_tokens_map.json and vocab.txt. Pin the extraction
snapshot directory to exactly these six files (cache symlinks to their hashed
bytes are permitted); reject extra loader inputs such as added_tokens.json or
adapter_config.json rather than loading unbound tokenizer/model behavior. Pin the
Python executable, Python/torch/transformers/tokenizers/numpy versions and source
hashes. No remote code, adapter, fine-tuning, model selection, retrieval instruction
or quantization. Use CPU float32, eager attention, evaluation mode, no gradients,
deterministic algorithms, seed 20260915, four torch threads and batch size four.

Tokenize the exact original prompt with no added special tokens and no truncation.
Split its complete token sequence into contiguous nonoverlapping chunks of at
most 510 content tokens; add the model's two special tokens to each chunk. Extract
the final-layer CLS vector for each chunk, L2 normalize each chunk vector, take
the content-token-count-weighted mean, then L2 normalize that mean. Empty input,
empty token sequences, unexpected shape, zero vectors or nonfinite values fail;
there is no fallback embedding. Record complete token and chunk counts. Do not
decode/rewrite chunks or silently discard a tail that might contain the request.
This deterministic document pooling loses some cross-chunk relations and order;
it is a specified prompt representation, not a full semantic understanding claim.

The official model documentation specifies CLS pooling, normalization, 384
dimensions and a 512-token limit. This extension's long-document pooling is our
frozen construction, not a claimed official long-context capability.
Sources: [model card](https://huggingface.co/BAAI/bge-small-en-v1.5),
[pinned revision](https://huggingface.co/BAAI/bge-small-en-v1.5/commit/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a).

## Separate extraction and fitting artifacts

The project Python's `prepare` action verifies the frozen retained-data chain and
writes two new immutable private artifacts: a prompt-only packet, sorted and
deduplicated by exact prompt SHA256, and a separate preparation inventory. The
packet contains only prompt strings, their hashes, extraction contract and source
hashes. Labels, outcome metadata, response text, prefix features and internal
readouts are not fields of this packet and never reach the encoder.

The extraction Python consumes only that packet and a frozen encoder-config JSON.
Its config schema is `prompt-semantic-encoder-v1`, with exactly `schema_version`,
`encoding` (the module's ENCODING constant), `model_path` (absolute pinned snapshot
directory), `model_files_sha256`, `runtime` (from encoding_runtime()) and
`source_pins` (from extraction_sources()). Config hashes bind its raw file bytes.

Before full extraction, benchmark exactly the first 32 unique prompt SHAs in
sorted order, without inspecting labels or selecting examples by outcome. Report
only counts, token/chunk aggregates and elapsed time. Its unchanged receipts may
be reused by the complete extraction. Preserve every prompt; no throughput or
embedding-quality result permits outcome-driven sampling or encoder changes.

Per-prompt owner-only immutable receipts bind prompt, packet and encoder-config
hashes, complete chunk lengths and the normalized 384-dimensional vector. A cache
index binds each receipt hash and its immutable run header. The fitting runtime
re-reads and validates the complete receipt set, exact prompt population, source
and config bindings; a benchmark or partial cache cannot be used for fitting.
The extraction runtime may differ from the fitting runtime. The latter uses the
existing project's pinned numpy/scipy/scikit-learn stack and gets a separate
new fit inventory. Existing lexical inventory/source pins are never overwritten.

## Fitting and reporting

Keep the original fold-only TF-IDF vocabularies/document frequencies and
StandardScaler fits for structural and J-lens features. Semantic embeddings and
retained prefix hashes keep their fixed L2 normalization; they are not individually
standardized or assigned tuned block weights. A frozen, per-document encoder may
run before cross-validation because it does not learn from this corpus, labels,
test distribution or other documents. All learned corpus transforms and classifier
parameters remain training-fold-only, including structural-family exclusions.

Reuse unweighted L2 logistic regression, C=1, liblinear, tolerance 1e-6, at most
1,000 iterations and random seed 20260915. Convergence failure or exceeded
fit-start deadline makes that fit unavailable; no solver or threshold fallback.
No embeddings, vocabulary, coefficients or item-level predictions enter public
artifacts. Metrics retain log loss, Brier, ROC AUC, average precision, horizon,
intent, unknown-label, censoring and request-core descriptions from the existing
helper. First compare log loss, with Brier corroboration; ranking metrics are
descriptive. Do not treat repeated rows or the 40 overlapping cells as independent
samples or declare significance from win counts. A pooled uncertainty analysis
requires its own specified treatment of core clustering and fitted-model variation.

## Exact CLI and execution bounds

All commands run with `CUDA_VISIBLE_DEVICES=''`, `TOKENIZERS_PARALLELISM=false`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and each of OMP_NUM_THREADS,
OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS and
VECLIB_MAXIMUM_THREADS set to an integer from one through four. Use an external
hard process timeout for extraction and each cell; the fit-start deadline cannot
interrupt native code already executing. Root freezes actual timeout/headroom
after the fixed label-blind benchmark and before complete extraction. Bound
documents to the existing 100,000-character maximum. No GPU is needed.

Here `PROJECT_PY` is the existing project Python; `ENCODER_PY` is the separately
pinned CPU extraction Python. Every output and receipt root must be outside the
Git worktree. Hash arguments are SHA256 of exact file bytes.

```sh
PROJECT_PY -m lexical_prompt_study.text_challenger_semantic prepare \
  --packet-output PROMPTS.json --output PREPARATION.json
ENCODER_PY -m lexical_prompt_study.text_challenger_semantic encode \
  --packet PROMPTS.json --packet-sha256 PROMPTS_SHA \
  --encoder-config ENCODER.json --encoder-config-sha256 ENCODER_SHA \
  --cache-root CACHE_DIRECTORY --benchmark --output BENCHMARK.json
ENCODER_PY -m lexical_prompt_study.text_challenger_semantic encode \
  --packet PROMPTS.json --packet-sha256 PROMPTS_SHA \
  --encoder-config ENCODER.json --encoder-config-sha256 ENCODER_SHA \
  --cache-root CACHE_DIRECTORY --output CACHE.json
PROJECT_PY -m lexical_prompt_study.text_challenger_semantic inventory \
  --preparation PREPARATION.json --preparation-sha256 PREPARATION_SHA \
  --cache CACHE.json --cache-sha256 CACHE_SHA --output INVENTORY.json
PROJECT_PY -m lexical_prompt_study.text_challenger_semantic run \
  --preparation PREPARATION.json --preparation-sha256 PREPARATION_SHA \
  --cache CACHE.json --cache-sha256 CACHE_SHA \
  --inventory INVENTORY.json --inventory-sha256 INVENTORY_SHA \
  --placement scaffold_before_request --fold 0 --max-seconds 900 \
  --output CELL.json
```

The final command is repeated for the frozen 40-cell matrix: both placements,
folds 0–4, and either no `--held-family` or exactly one of attack_block_mask,
structural_sham, harmless_structured_wrapper. Each immutable cell contains just
the two new models and aggregate metrics. Errors emit a fixed rejection status,
never raw text. No download or inference is performed on import or preparation.
