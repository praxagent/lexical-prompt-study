# A166: bounded CPU state diagnostic

Prospective instrument, conditional on the separate A165 receipt/native-envelope
posthoc audit finding no concrete implementation bug. No A166 calls have occurred.
This is a descriptive diagnostic motivated by the already-observed A165 outcomes;
it does not reopen A164's failed gate or use its unused held-out worlds.

Before execution, freeze the A166 source/tests/protocol, complete private plan,
unchanged A165 run header and its exact config, model/config/tokenizer/source
closure, and all 24 native rendered inputs. The compiler validates the frozen
A165 plan and selects the first four distinct development worlds in the original
A164 base-plan order. Retain both selectors in their original trial order. Each
of these eight entries becomes one consecutive triplet: unchanged no-scaffold
baseline, unchanged A165 inert scaffold BEFORE the request, identical baseline.
Pre/post trial identities are new and distinct, while their actual messages and
native input token arrays must be identical. There is no prompt, model, renderer,
precision, placement, seed, order, outcome-dependent selection or held-out search.

Run all 24 scheduled calls in one fresh CPU FP32 process with a single model load.
Reuse the exact reference loader `run_cpu_reference_audit.py`, SHA-256
`f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`,
and the A165 config. The inherited config describes the original NF4 instrument;
this run explicitly binds the same original unquantized FP32 CPU reference loader
and effective CPU SDPA settings as A165. Use four CPU threads, unchanged seed,
greedy single-beam generation, 64 new tokens and at least 48 GiB available memory
before loading. No GPU, API, model downloads, manual state reset between calls,
retries, resumed attempts, repeated process in the same output directory, or
extension after the 24 calls. Pre-baseline competence is a descriptive prerequisite
for interpretation, not an adaptive execution gate. Task errors do not alter the
schedule; infrastructure errors are recorded as unresolved, never retried.

A process flock, immutable one-shot claim before model loading, and a process-local
consumption guard prevent duplicate calls. Private attempts bind sequence order,
source/plan/config/model/native-input hashes and EOS IDs. Completed native token
arrays, decoded responses and strict scores remain private. A SIGTERM/SIGINT/SIGHUP
raises BaseException so it cannot be swallowed by per-trial error recovery. Launch
under an external hard 1,800-second timeout with a short kill grace; an internal
process-wide 1,800-second alarm additionally covers preflight, loading and calls.
On interruption, preserve the pending attempt and all missing scheduled cells;
never rerun it. These bounds allow a partial diagnostic, not an automatic extension.

Numeric export and aggregation acquire the read lock and replay every native
prompt, response-token decode, termination and strict score using the source-pinned
A164 native validator through explicit A166 wire adaptation. Reject source/input
changes, unknown/duplicate/out-of-order trials, orphan results, mismatched attempts,
changed scores and stale stored exports. Aggregate only after receipt replay.
Private token-hash comparisons stay outside stdout; public-safe output contains
only counts, bounded differences and artifact/source hashes, with no world/trial
identifiers, prompts, answers, tokens or per-pair response hashes.

Report all 24 planned cells, with eight pre, eight inert and eight post cells;
completed, failed, missing and interrupted coverage; strict/cap/parser-category
counts; complete pre-baseline competence; eight planned pre/post pairs; strict
improvements/deteriorations and exact generated-token identity counts. Missing
or infrastructure-failed binary outcomes have [0,1] bounds. The post-minus-pre
strict difference always uses all eight planned pairs, and is a point estimate
only when all pairs are observed. Never silently restrict it to complete pairs.

The predefined diagnostic decisions are:

- Any observed correct-pre to wrong-post transition flags investigation of
  possible persistent state, even if another triplet is missing. This is a warning,
  not an attribution; incompleteness and other limitations remain explicit.
- All eight pre and post baselines strictly correct, all eight pairs token-identical,
  and all eight inert responses strict failures reduces concern about persistent
  state carrying into the immediate following baseline in this fixed schedule.
  It does not prove absence of state effects or explain the scaffold failures.
- Otherwise, including unestablished pre-baseline competence, nonreproduced inert
  failure, token differences without strict deterioration, or missing cells without
  observed deterioration, the diagnostic is inconclusive.

No confidence intervals, significance tests, equivalence/population claims,
quantization-only causal explanation, mechanism proof or automatic calls beyond this fixed 24-call diagnostic.
End this branch after the scheduled run and its receipt-backed interpretation.

The compiler API is `compile_plan(a165_plan, a165_run_header=header,
protocol_sha256=...)`; all inputs must be hash-bound private files. The runner CLI
is `python -m lexical_prompt_study.instruction_selection_cpu_state --mode run`
with `--plan`, `--plan-sha256`, `--config`, `--config-sha256`, `--protocol`,
`--reference-script`, and an absolute private `--output-root` outside the source
checkout. `--mode export` performs tokenizer/receipt replay only, with no model.
`prepare_inputs(...)` exposes the same bound native preparation before any model
load; its returned prompt/token arrays are private freeze artifacts. Preserve the
original virtualenv interpreter path when launching frozen source.
