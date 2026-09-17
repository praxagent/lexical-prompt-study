# A170: prospective numerical check of a shared JSON prefix

This is a new numerical-method question using three fixed, harmless synthetic
contexts. It does not continue or repair the A169 answer task. A169 remains
closed at 132 of 288 planned candidate evaluations; no A169 private input,
response, score, native freeze, run header or parent plan is an A170 input.
Public A169 runtime helpers and the pinned CPU reference loader may be reused
as implementation dependencies. No A170 outcome is asserted by this protocol.

The question is whether the same three prefix-token probabilities change across
fresh CPU FP32 causal forwards when the supplied suffix and total input length
change, and whether equalizing total length changes the observed differences.
Two identical minimum-prefix forwards bracket each context's candidate forwards
to describe within-context repeat drift. These are numerical diagnostics; they
do not establish a cause of differences, answer-task competence, generation
accuracy, population behavior or a mechanism.

## Fixed inputs and native-token preparation

Use exactly three contexts, in repeat-count order `0`, `24`, `72`. Each has the
system message `Return the JSON object requested by the user.` and one user
message constructed exactly as follows:

```python
content = ("The field is quiet. " * repeat).rstrip() + '\n\nReturn {"answer":"s00"}.'
```

Apply `rstrip()` only to the repeated sentence, never to the complete content.
The repeat-zero user message therefore begins with two newline characters.
Use the pinned tokenizer's native chat template, generation prompt, date setting
and special-token rules. No input is drawn from an earlier study. Freeze the
messages, rendered native text and native prompt token arrays before loading the
model; preserve them privately.

For every context, supply exactly four compact ASCII canonical assistant JSON
objects, in value order `s00`, `s01`, `A`, `B`: `{"answer":"s00"}`,
`{"answer":"s01"}`, `{"answer":"A"}`, and `{"answer":"B"}`. These fixed
values distinguish tokenized suffixes, not correct and incorrect answer classes.
Append exactly the native intended assistant EOS token to each target.

Prepare the native prompt plus each complete JSON string in one tokenization.
Require exact prompt-token prefix invariance and exact decoded continuation
round-trip with special tokens retained and tokenization cleanup disabled.
Separately render the completed assistant message with the native chat template.
Its native token array and the encoding of its full rendered text must both
equal the prompt plus the JSON continuation plus exactly one intended EOS token.
Require that EOS to equal the tokenizer's native EOS ID and belong to the frozen
generation EOS list. Reject any premature EOS and any context-limit violation.
Do not assume that independently encoded string pieces concatenate.

Require the longest common continuation-token prefix across all four paths to
contain exactly three tokens, and freeze these as the measured prefix. Require
all four paths to share these tokens and require the same three
token IDs and boundary in all three contexts. Their decoded text must be a
nonempty pure prefix of `{"answer":"`, and must round-trip jointly with the
native prompt. A valid native three-token boundary may decode to only part of
that JSON syntax; it need not end after the opening answer-value quote. Do not
move the boundary to a preferred textual position or choose it from results.
Audit the common boundary against the complete continuation arrays before any
model call. Every full target must continue beyond the measured prefix.

Record JSON, EOS and total target lengths privately. Require equal total target
lengths within the `s00`/`s01` pair and within the `A`/`B` pair, and require the
two pairs to have unequal total target lengths. Freeze the lengths and pad
counts; stop preparation if these guards fail rather than change values.

## Exact 30-forward schedule and causal alignment

For each of the three contexts, execute these ten forwards in this exact order:

1. One minimum-prefix reference, named `prefix_pre`.
2. Four variable-length full-target forwards, in `s00`, `s01`, `A`, `B` order.
3. Four fixed-length full-target forwards, in the same value order.
4. One minimum-prefix reference, named `prefix_post`.

This is exactly 30 planned forwards in one fresh process with one model load.
Every forward is a fresh batch of one with `use_cache=False`; no shared KV cache,
warmup model call, free generation, sampling, extra candidate, retry, resume or
adaptive evaluation is permitted. Numerical thresholds do not stop or extend
the schedule. Recoverable evaluation failures become missing planned outcomes
and receive no retry. Signals and the wall deadline stop the process, preserving
the remaining planned measurements as missing.

Let `P` be the native prompt-token array, `T` a complete JSON-plus-EOS target,
`q = 3` the fixed prefix count, and `M` the largest of the four full target lengths
within that context. The three predictor positions in the complete causal
sequence are always `len(P) - 1`, `len(P)` and `len(P) + 1`. They predict the
three shared target tokens in order. Native prompt tokens are conditioning.

For each minimum-prefix reference, input `P + prefix[:q - 1]`, with an all-one
attention mask, and request `logits_to_keep=q`. This is the minimum input needed
to predict all three prefix tokens. All three returned rows are the intended
predictors; the third prefix target need not be an input token.

For each variable-length forward, input `P + T`, with an all-one attention mask,
and request `logits_to_keep=len(T) + 1`. The returned rows begin at the same
`len(P) - 1` predictor position. Read only the first three rows. The remaining
rows, including the final EOS input's next-token prediction, are outside this
measurement and are discarded without answer or EOS scoring.

For each fixed-length forward, right-pad `T` to `M` using
`tokenizer.pad_token_id`, or the audited native EOS ID if the pad ID is absent.
Input `P + T + padding`, use attention-mask ones for every prompt and real target
token and zeros for all right-padding tokens, and request
`logits_to_keep=M + 1`. Read only the first three returned rows, which must again
begin at `len(P) - 1`. Record the pad ID, target length, pad count, complete mask,
input length and requested logit-row count privately. All four fixed-length
forwards within a context must have exactly the same total input length.

## Runtime and readout

Use the original-weight, unquantized model and the same pinned original CPU
configuration and corrected CPU reference loader used by the public A169
implementation. The reference-script SHA-256 is
`f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`.
The original configuration SHA-256 is
`1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745`.
Validate configuration, source, tokenizer/chat-template, EOS and model-file
bindings before execution. The retained configuration describes an older NF4
instrument; the effective reference loader uses original weights, CPU SDPA,
FP32 model arithmetic, four CPU threads and the unchanged pinned seed. Require
at least 48 GiB of available RAM immediately before model loading. Use offline
local files only, with no GPU, external model API, download or spending.

Require the expected returned logit shape and finite FP32 logits for all three
measured full-vocabulary rows. For each row, save the chosen shared-prefix
token's FP32 logit and compute the full-vocabulary logsumexp in float64 from
those FP32 logits. Save that normalizer and their difference as the token log
probability. Sum the three log probabilities with accurate Python summation.
Save only the three-prefix-token readout; no answer-continuation, full JSON,
joint-path, EOS or length-normalized probability is computed or reported.
Reject nonfinite normalizers, token probabilities or prefix sums. Missing
measurements remain missing and are never replaced with zero.

## Prespecified numerical summaries

Keep detailed numeric records private. For each context and each candidate
method (`variable_full` and `fixed_full`), report the range (maximum minus
minimum across the four candidates) of the three-token prefix sum. Describe
whether that range is at most `0.0001` nats, absolute, with no relative tolerance.
There are exactly six such groups. This is a descriptive reuse of the A169
prefix tolerance, not an eligibility gate, accuracy measure or task-repair rule.
An incomplete group has a null range and null tolerance result, with coverage
shown explicitly; do not substitute an observed-candidate subset.

For each complete group, also report the per-token range across its four
candidates for chosen logits, float64 log normalizers and token log
probabilities. Preserve all six ordered pair comparisons from the frozen value
order (`s00` versus `s01`, `s00` versus `A`, `s00` versus `B`, `s01` versus `A`,
`s01` versus `B`, and `A` versus `B`), with each signed difference defined as
the first named value minus the second. Retain the per-token differences for
chosen logits, log normalizers and token log probabilities, and the difference
of the prefix sums. A pair requires both planned measurements; otherwise its
differences are null.

For every candidate forward, record the same signed per-token differences and
prefix-sum difference against that context's `prefix_pre` measurement (candidate
minus `prefix_pre`). Record `prefix_post` minus `prefix_pre` in the same form as the fixed
repeat-drift diagnostic. Preserve each planned comparison and its missingness.
Also retain each candidate's signed `fixed_full` minus `variable_full` differences
in the same per-token and prefix-sum form, requiring both measurements.
Do not average across prompt lengths to conceal a missing context. There are no
answer scores, candidate preference ranks, accuracy surrogates, confidence
intervals, population inference, adaptive tolerance, causal attribution or
mechanism claims. Comparing variable and fixed input shapes can describe this
instrument's measured behavior only; it cannot identify the underlying source
of a difference or reopen A169.

The variable-to-fixed comparison changes physical input geometry and, for short
targets, introduces right-padding and zero-valued attention-mask entries. It may
therefore change mask handling or kernel code paths as well as length; it does
not isolate length as a cause. Minimum-prefix references also change context
and output-projection geometry relative to complete-target forwards. Equal-
length candidate pairs test substitution of future suffix content within their
method, while cross-pair comparisons combine suffix content and native target
length differences. Equality of the measured chosen logits, log normalizers or
prefix probabilities does not prove bitwise equality of the complete vocabulary
logit vectors, which are not retained as the measurement.

## Freezing, durable evidence and export

Compile the new plan directly from `config_sha256` and `protocol_sha256`, binding
the current source and pinned runtime dependencies. Do not embed or require any
parent study plan or run artifact. Before execution, freeze source, tests,
protocol, config/plan bindings and all native input variants and readout
positions. Preserve the original virtualenv interpreter path.

Use an absolute private output directory outside the source checkout, private
file permissions, exclusive process flock, immutable one-shot claim before
loading, and a process-local consumption guard. Record each sequential attempt
durably before its forward and each result durably after it. Bind attempts to
the exact plan, source, config, runtime, native input, target, attention mask,
padding and readout geometry. Reject a consumed run; do not evade a one-shot
claim by retrying under another directory. Preserve interrupted attempts and
all 30 planned outcome slots.

The internal wall deadline is 1,800 seconds including preflight and loading.
The external limit is 1,800 seconds with a 60-second kill grace. SIGTERM,
SIGINT, SIGHUP and SIGALRM raise `BaseException` and bypass recoverable evaluation
errors. Record the actual received signal number and receiving process PID when
available; a Python `KeyboardInterrupt` without a handler receipt has a null
signal number. No generic `BaseException` is treated as a recoverable error.
Retain a bound completion/interruption/failure receipt and explicit
attempted, completed, failed and missing counts. Neither numerical results nor
private strings, token arrays, per-item identifiers or hashes are printed to
stdout; the CLI prints only safe execution status and coverage.

Export under a shared lock without loading the model. Reconstruct native inputs
and all three input constructions; verify exact source/config/protocol/runtime
lineage, attempt ordering, masks, padding, predictor alignment, expected row
counts, result bindings and completion coverage. Recompute token probabilities
from saved chosen logits minus saved log normalizers, use accurate summation to
recompute prefix sums, and recompute every range, tolerance flag and signed
comparison. Reject changed stored summaries, unknown or orphan results, gaps in
attempt order and invalid completion receipts. Replay verifies construction and
saved arithmetic; it does not recreate or independently verify model logits.

API: `compile_plan(config_sha256, protocol_sha256)`, `prepare_inputs`, `run_plan`
and `export_run`. Preparation, execution and export use keyword-only bound
paths/hashes: `plan_path`, `plan_sha256`, `config_path`, `config_sha256`,
`protocol_path` and `reference_script`; execution/export also take
`output_root`. CLI: `python -m lexical_prompt_study.prefix_numerics --mode run`
(or `export`) with `--plan`, `--plan-sha256`, `--config`, `--config-sha256`,
`--protocol`, `--reference-script` and `--output-root`. Raw construction,
measurements and detailed numerical summaries remain private return values and
private artifacts. Public release is limited to separately reviewed aggregate
numbers and whole-artifact provenance.

## Qualification before any model execution

Use meaningful model-free tests for exact synthetic bytes (including repeat-zero
leading newlines), fixed counts/order, plan drift rejection, native prompt and
completed-assistant EOS audits, a partial-JSON native prefix, shared-prefix and
pair-length validation, and right-padding masks. Qualify all three predictor
alignments against full-logit tiny-model controls; include unequal target
lengths so a misplaced shifted row is observable and inspect the forwarded
attention mask exactly. Check
float64 readout and accurate summation against an independent arithmetic
calculation, and use synthetic differing logits to expose signed-difference or
range mistakes. Exercise the six-group tolerance boundary, incomplete groups,
reference missingness, nonfinite rejection, one-shot/lock enforcement,
deadline/signal interruption, no-retry behavior and durable export replay.
Tamper controls must reject altered geometry, token readout, numeric summaries,
receipts and schedule order. These controls qualify implementation behavior;
this prospective protocol records no test pass or experimental outcome.

Use the pinned interpreter without resolving its path:
`/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python`.
Import Torch and the Transformers tiny-model classes before appending the
repository's `.venv/lib/python3.12/site-packages` for pytest, so optional model
dependencies are discovered in the pinned runtime rather than from incompatible
Python 3.12 binary packages. Tiny models use randomly initialized local weights
and make no pretrained-weight or target-model calls.
