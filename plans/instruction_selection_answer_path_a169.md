# A169: fixed teacher-forced canonical answer paths

Prospective final diagnostic branch on already-observed development contexts.
No held-out tasks, new worlds, wording, materials or response-based selection.
An unqualified or uninformative result ends this task-repair branch. This assay
conditions on supplied canonical JSON syntax; it does not rescue free generation
or provide direct causal evidence about a mechanism.

Compile from the bound A167 plan/header and its original nested A165 plan/header.
Use the same first four A167 development worlds and both original selectors.
Retain eight unique original baselines (collapse A167's repeated placements),
16 original A165 full contexts, 16 sham, 16 inert, and all 16 A167 prose contexts.
The original system-only selector, original A/B displayed labels, world values,
query order, task rendering, material bytes and placements remain unchanged.
These 72 contexts are exact parent inputs, not edited follow-ups to errors.

Execute all eight baselines first, in original world/selector order. Then visit
world, selector, and before/after placement in their original order. Each of
these 16 groups contains all four material contexts. Index the following fixed
orders by (zero-based world + selector + placement index) modulo four:

- full, sham, prose, inert;
- sham, inert, full, prose;
- inert, prose, sham, full;
- prose, full, inert, sham.

Each order occurs four times; every material occupies every position four times.
Full precedes sham eight times, and prose precedes inert eight times. This fixed
counterbalancing does not establish absence of order or state effects.

For every context, evaluate exactly four paths, always in this order: selected
value, other value, selected displayed label, other displayed label. Each path
is the compact, ASCII JSON object `{"answer":"VALUE"}` plus exactly the intended
native assistant EOS token. There is no free generation, sampled path, extra
candidate, retry, resume, or adaptive schedule. The maximum is 288 individual
teacher-forced candidate evaluations in one fresh process and one model load.

Before any model loading, prepare each entire native prompt plus canonical JSON
as one tokenization. Require exact native prompt-token prefix invariance and
exact decoded continuation round-trip. Independently render each completed
native assistant message; its token array and full-text encoding must both
exactly equal the native prompt plus canonical continuation plus the tokenizer's
single EOS token, which must belong to the frozen generation EOS list. Reject
premature EOS. Do not assume that independently encoded string pieces concatenate.
The longest common continuation-token prefix of all four paths must be nonempty,
decode to a pure prefix of the shared syntax `{"answer":"`, and round-trip jointly
with the native prompt. Its exact token boundary must be identical in all 72
contexts and frozen in the private native inputs. Closing syntax and EOS remain
in each answer continuation. Preparation reconstructs the complete A167 native
freeze and matching receipts, verifies its header, and verifies all 72 selected
contexts' original parent messages and native arrays. Root independently audits
this boundary and freezes source, tests, protocol, plan and all 288 paths.

Use the same original-weight, unquantized CPU FP32 model, pinned original A165
configuration and corrected CPU reference loader, SHA-256
`f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`.
The inherited config describes the old NF4 instrument; effective loading uses
original weights, CPU SDPA, four threads and the unchanged seed. Require 48 GiB
available RAM before loading, offline local files, no GPU/API/download/spending.
Use one fresh full causal forward per candidate with no shared KV cache. Request
the last target-count-plus-one logits; rows except the final row predict every
canonical JSON/EOS target beginning at native-prompt-length minus one. The final
EOS input's next-token logit is outside the scored path. Model forward arithmetic
is FP32; use float64 logsumexp of its FP32 logits for readout normalization and
Python accurate summation. Save each chosen-token logit, full-vocabulary
logsumexp and resulting log probability privately, along with raw prefix,
continuation, joint and EOS sums and token counts. No length normalization.

The split is raw joint log P(canonical JSON plus EOS | original native prompt)
equals JSON-prefix log probability plus remaining-answer log probability given
that prefix. Native prompt tokens are conditioning, not scored targets. Audit
that the maximum minus minimum shared-prefix sum across four candidates is at
most 0.0001 nats, absolute, with no relative tolerance. Failed or missing prefix
audits make that entire context unresolved; retain its available raw scores.
Reject nonfinite logits, normalizers or scores; never replace a missing score
with zero. Prefix probability describes only this canonical spelling, not the
probability of all valid output formats. Four-path rank is not full-vocabulary
choice probability. Different answer/label path lengths are part of this path
definition and may favor shorter labels.

After exactly 32 baseline candidate evaluations, require all eight complete,
prefix-audited baseline selected-value JOINT scores to strictly outrank all
three alternatives. Ties or any unresolved baseline fail. If unqualified, stop
before the remaining 256 evaluations and preserve all planned missing outcomes.
Do not relax the gate, choose another boundary or tune another task from results.
Recoverable candidate exceptions during an otherwise authorized schedule become
missing outcomes and receive no retry; process signals bypass candidate recovery.

Use an absolute private output directory outside the source checkout, exclusive
process flock, immutable one-shot claim before loading, process-local consumption
guard, and durable sequential candidate attempts/results. Source/plan/config/
model/native bindings are replayed. Internal deadline: 7,200 seconds including
preflight/loading; external deadline: 7,200 seconds with 60-second kill grace.
SIGTERM, SIGINT, SIGHUP and SIGALRM raise BaseException. Preserve interrupted
attempts and every unresolved planned path. No interim effects or private strings,
responses, token arrays, identifiers or per-item token hashes go to stdout.

Export under shared lock without loading the model. Reconstruct native inputs;
verify attempt order, gate compliance and source lineage; recompute token log
probabilities from saved chosen logits minus logsumexp; recompute split scores;
reject changed summaries, orphan results, unknown attempts and invalid completion
receipts. This arithmetic replay does not recreate model logits; root additionally
qualifies causal alignment against full-logit tiny-model controls and independently
verifies the frozen private construction and run receipts.

Primary: for each placement separately, full minus sham in the selected-value
answer-continuation log-probability margin against the best of the three competing
paths. Pair the two selectors within each of the four fixed worlds, average their
differences, then average the four world means equally. Each placement retains
eight planned pairs. Secondary: analogous prose minus inert; selected-value minus
other-value and selected-value minus selected-label continuation margins by
material/placement, with the same two-selector/four-world averaging. Also report
the shared canonical JSON-prefix log probability by material/placement, always
using the selected-value path as the fixed reference after the four-prefix audit.
Retain raw candidate joint/prefix/continuation/EOS scores and lengths privately,
joint and continuation margins, selected-path preference counts and full coverage.
Selected-path preference is a descriptive rank among these four supplied paths,
not generation accuracy. No accuracy surrogate, CI, population inference or
mechanism proof. Unresolved continuous contrasts are null with unbounded missing
outcomes; do not drop pairs or average only observed worlds. Report each fixed
world's mean, preserving nulls, and identify completion/audit guards explicitly.

API: `compile_plan(a167_plan, a167_run_header, protocol_sha256)`, `prepare_inputs`,
`teacher_force`, `run_plan`, `export_run`. CLI: `python -m
lexical_prompt_study.instruction_selection_answer_path --mode run` (or `export`)
with `--plan`, `--plan-sha256`, `--config`, `--config-sha256`, `--protocol`,
`--reference-script`, and `--output-root`. Preserve the original virtualenv
interpreter path. Raw construction and score objects remain private return values;
only reviewed aggregate numbers and whole-artifact provenance are releasable.
