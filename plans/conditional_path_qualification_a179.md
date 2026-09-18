# A179: fresh-control conditional answer-path qualification

This is a prospective 32-forward instrument qualification, not free generation,
a scaffold experiment or a continuation of A169. A178 closed the strict-generation
lookup branch with clean 7/8 and all context arms 0/48, dominated by format
failures. Supplying a canonical candidate answer changes the measurement
question. A179 asks whether one fixed teacher-forced readout is numerically
consistent and follows the authoritative selector on fresh clean controls.
It neither rescues any generation score nor measures hidden knowledge.

A169 remains interrupted, incomplete and permanently closed. Its larger
shared-prefix inconsistency remains unexplained; A170's limited synthetic
checks do not establish that every answer-path context is numerically qualified.
No historical context, response, token array, material or held-out input is used.

## Exact controls and instruction

There are exactly two fixed public value pairs: `(s96, s97)` and
`(s98, s99)`. Each is used in normal and swapped A/B assignment. These four
worlds are transformations of two pairs, not four independent pairs or eight
independent replicates.

| World | Pair | Assignment | A value | B value | Selector context order |
|---|---:|---|---|---|---|
| w00 | 0 | normal | s96 | s97 | A, B |
| w01 | 0 | swapped | s97 | s96 | B, A |
| w02 | 1 | normal | s98 | s99 | B, A |
| w03 | 1 | swapped | s99 | s98 | A, B |

The user message is the entire mapping as compact JSON with double quotes,
`separators=(',', ':')`, and A followed by B in every world. Presentation order
is always AB; selector execution order is a separate property. There is no
heading, scaffold, prose, reminder or trailing newline. Within each world,
selector switches leave user bytes identical. Swapping assignment exchanges
which label refers to each fixed candidate value, without changing candidate
identity. There is no vocabulary search or substitution on qualification failure.

Use exactly these three single-line ASCII templates in order:

```text
Select the entry labeled {selector} from the user's mapping.
```

```text
Return the mapped value of the entry labeled {selector} as exactly one JSON object with the single key "answer" and a string value.
```

```text
Do not transform the value or include other fields, explanations, or Markdown fences.
```

Replace both selector slots with the active uppercase label and join sentences
with exactly one ASCII space, without leading/trailing whitespace. For both
selectors the system message must equal A177
`system_instruction('of_explicit_label', selector)` byte for byte.

## Candidate identities and fixed 32-forward schedule

Each context has exactly two canonical candidates. `value_1` always denotes
the first literal value in its pair, and `value_2` the second, regardless of
world assignment or selector. Each candidate is compact JSON with exactly the
key `answer` and that candidate value as its string, followed by the intended
native assistant EOS. The selected candidate is determined by mapping and
selector; do not reorder or rename candidates by correctness.

Execute worlds and their two contexts in the displayed order. Within every
context of an even-indexed world, execute candidate order
`value_1, value_2, value_2, value_1`; within odd-indexed worlds execute
`value_2, value_1, value_1, value_2`. Assign repeat index 0 at the first and 1
at the second occurrence of each candidate. Assign a unique global sequence
index 0 through 31. All four evaluations are distinct fresh forwards, even
when their input bytes are identical. No cache, batching, warmup or reuse of a
saved score replaces a scheduled repeat. There are eight contexts and 32
forwards. Run all 32 regardless of interim numerical or functional outcomes.

## Native construction and fixed syntax prefix

Freeze all native constructions before target forwards. Let P be the native
prompt ending at the assistant-generation boundary. Tokenize the complete
rendered prompt plus each canonical JSON candidate jointly; require its prompt
prefix to equal P exactly. The candidate continuation must decode exactly to
the canonical JSON. Complete native assistant rendering must equal P plus
that continuation plus exactly the intended native EOS. Require rendered-text
and token round-trips, frozen template/EOS bindings, no earlier EOS in a
candidate, token-vocabulary bounds, and P plus the actual complete candidate
path fitting the bound model context limit. No padding or truncation is allowed.

Fix `q=3` before tokenization. The first three candidate token IDs must be
identical across both candidates and all eight contexts. Their decoded text
must be nonempty and a prefix of the pure JSON opening `{"answer":"`;
it need not equal that entire opening. Require joint prompt-plus-prefix
round-trip. The two complete candidate paths, including EOS, must have equal
token lengths within each context and must extend beyond q. Do not replace q
with the longest common prefix: any common tokens beyond q, including a value
stem, belong to the scored suffix. Record all lengths. Failed native
qualification stops before target calls and does not authorize changing values,
prefix, padding or candidate definitions.

## Exact teacher-forced arithmetic

Reuse A169's pinned `teacher_force` implementation directly. For each evaluation,
feed the complete P+T sequence, where T is the candidate JSON continuation plus
native EOS. Use a fresh batch-one causal forward with all-one attention mask,
`use_cache=False` and `logits_to_keep=len(T)+1`. Score the first target token
from the last prompt position and each later target from its preceding token.
Discard the final logit row produced at the input EOS, which predicts outside
the path. Do not use that row to score EOS itself.

Convert used logits to FP32, then compute chosen logits and full-vocabulary
log-sum-exp normalizers in float64. Each token log probability is chosen logit
minus normalizer, in nats. All used logits and recorded numerical values must
be finite. Retain chosen logits, normalizers, token scores and their arithmetic
privately for independent replay; do not print arrays.

The primary candidate score is the sum after the first q tokens, **including
EOS**. The shared syntax-prefix sum, complete-path sum and EOS log probability
are separate diagnostics. Record token counts and the EOS contribution;
do not normalize by token count, exclude EOS from the primary suffix, score
only the answer token, or infer a candidate score from generated text.

## Per-context qualification and missingness

Every context requires all four evaluations. Numerical validity requires native
construction success, finite bound score arithmetic, shared-prefix-sum range
across all four forwards at most `1e-4` nats, and continuation-score repeat drift
for each candidate at most `1e-4` nats. These limits are inclusive. A larger
range or drift fails numerical qualification; it is not repaired, rounded into
acceptance or excluded from the cohort.

For a complete numerically valid context, let S0/S1 be the selected candidate's
two suffix scores and U0/U1 the other candidate's. Report:

- mean margin: `(S0 + S1)/2 - (U0 + U1)/2`;
- worst-repeat separation: `min(S0,S1) - max(U0,U1)`.

Functional qualification requires worst-repeat separation **strictly greater
than `1e-3` nats**. Equality at that boundary fails. Mean margin and worst-repeat
separation remain reportable when numerical validity passes but functional
qualification fails. Exact mean ties, exact worst-repeat ties, negative/reversed
order and positive but insufficient separation remain distinguishable. Preserve
the threshold-boundary case explicitly; do not call a small positive margin
qualified merely because its sign is correct.
The `1e-3` separation is an engineering acceptance criterion, not a statistical
threshold or a validated bound on numerical error.

If any required evaluation is missing or infrastructure-failed, qualification
flags are unknown/null where their inputs are unavailable. If completed
numerical checks fail, numerical qualification is false and the functional
readout is unavailable. Missing or numerically failed context margins are null
with unbounded uncertainty, not zero, bounded binary outcomes or a silently
dropped context. No complete-case mean substitutes for the planned eight-context
result. Preserve attempted, completed, failed, interrupted and unattempted slots.

The prefix guard requires all four evaluations; each candidate repeat guard
requires its own two. `numerical_valid` is false if any available required guard
fails, true only when all four evaluations are complete and every guard passes,
and null otherwise. Thus a known repeat failure can disqualify an incomplete
context without imputing its missing result. Functional flags and margins are
null unless numerical validity is true. Overall `qualified` uses the same
conjunction rule: false if any context has a known numerical or functional
failure, true only if all eight qualify, and null otherwise. Execution
`status=complete/incomplete` remains separate from qualification.

Retain separate `mean_tie`, `worst_tie`, `mean_reversed`, `worst_reversed`,
`positive_low_separation` and `threshold_boundary` flags. The first two mean
exact zero; reversed means a negative corresponding margin; low separation
means `0 < worst_margin_nats <= 1e-3`; boundary means exact equality to `1e-3`.
Flags may overlap. The exclusive `functional_status` is `unresolved`, `pass`,
`worst_reversed`, `worst_tie`, `threshold_boundary` or
`positive_low_separation`, using that margin-based distinction after numerical
qualification. Each available margin's `*_bounds_nats` has lower and upper
equal to the margin; otherwise both are null and `margins_unbounded` is true.

Overall qualification succeeds only if all eight contexts are complete,
numerically valid and functionally qualified. Report separate coverage,
numerical-validity, functional-qualification, tie/reversal/low-separation and
boundary-failure counts. Do not describe the eight context checks or two repeats
as independent inferential samples. There are no confidence intervals,
population estimates, scaffold effects, generative accuracy scores or causal
patching endpoints.

## Decisions and fixed stop rules

Failure ends this frozen calibration. Do not revise vocabulary, q, candidate
strings, tolerances, margins or numerical construction to turn it into a pass.
No interim gate truncates the scheduled remaining evaluations after a
recoverable per-forward failure or observed numerical/functional failure.
Stop after 32 attempts, the deadline, an interruption or terminal failure;
retain all remaining slots as missing and never retry or resume.

Passing supports only separately designing a prospective scaffold readout
experiment. It does not establish an internal selection mechanism, hidden
correctness, unconditional answer preference, restored A178 performance,
broader model reliability or eligibility for activation patching. No scaffold
readout is collected in A179. The strict-generation branch and A169 remain
closed regardless of this outcome.

## Runtime, binding and private evidence

Use the original local weights and corrected CPU FP32 reference, CPU SDPA,
four CPU threads, unchanged seed, batch one and one fresh model load. No
generation, target warmup, sampling, cache, GPU, API, download, rental, paid
judge or new spending is authorized. The historical config's NF4 description
does not change the effective original-weight CPU FP32 implementation.

Bind original configuration SHA-256
`1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745`,
corrected reference-script SHA-256
`f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`,
A177 helper SHA-256
`7a10f0e4c361f35b0c56b92c6921bd89ef7cec2f7b8deefdb99912dacaeb394f`,
and A169 readout helper SHA-256
`1b63ddf15eae5866b444cdb94293b80dcf7bd6e580978a23873c0167cb40e69e`.
Preserve the interpreter path without resolving it:
`/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python`.

Freeze reviewed protocol, source, tests, plan, dependencies and native inputs
before target calls. Respect shared jobs and require at least 48 GiB available
RAM immediately before loading. Use the established reservation and exclusive
process lock. Enforce a 3,600-second internal/external deadline including
preflight and loading, with 60-second kill grace. SIGTERM, SIGINT, SIGHUP and
SIGALRM are terminal, bypass recoverable handling and retain actual signal
number/name and PID. Direct KeyboardInterrupt records null signal number/name;
generic BaseException is not a recoverable forward failure.

Use a fresh absolute private run directory under
`/data2/PRAX/lexical-prompt-study-data/runs/a179/`. Preserve private permissions,
immutable startup/one-shot receipts and process consumption guards. Publish
each attempt before its forward and each result durably afterward, including
zero-call startup failures. Changing output directory cannot resume a consumed
process or run. Export under lock without model forwards and independently
reconstruct native paths, causal alignment, score arithmetic, repeated-score
checks, selected/other assignment, margins, threshold flags and all coverage.
Private prompts, token arrays, numerical arrays and per-item identities stay
outside Git and console output. Root owns review, code-only backup/push/freeze,
execution, independent verification and archive.

Terminal replay checks saved chosen logits minus saved normalizers, token sums
and downstream arithmetic. It does not independently recompute full-vocabulary
log-sum-exp from those saved scalars: full-vocabulary vectors are not retained
and no new model forward is allowed. The analytical and tiny untrained Torch
qualification checks the readout implementation separately; neither check makes
the terminal replay an independent regeneration of model logits or normalizers.

## Public synthetic qualification

Use public synthetic tokenizers and scores only; never read a historical
private context or load a real target tokenizer/model. Verify literal
instruction inheritance, two-pair/four-world/eight-context design, assignment
swaps, candidate identity, repeated-evaluation schedule and all 32 slots.
Analytical logits and a tiny untrained Torch model must independently verify
causal shift, EOS inclusion, final-row exclusion, float64 normalization and
fresh cache-free calls. Test q=3 syntax/round-trip/geometry guards, unequal
candidate lengths, nonfinite arithmetic, exact threshold boundaries, ties,
reversals, low positive separation, repeated-score and shared-prefix drift,
and missingness without imputation. Exercise full schedule despite failed
checks, one-shot semantics, signals/deadline, locks, durable publication,
tamper rejection, replay and privacy. Qualification never reads the closed
A169/A178 contexts or collects scaffold outcomes.
