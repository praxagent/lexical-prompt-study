# A197: instruction rule × supplied-answer cue interaction

This prospective finite experiment asks whether sensitivity to a supplied answer depends on an instruction to copy versus recompute the arithmetic. It uses eight fresh cores and full conditional candidate-path scores. The intervention is the exact pair of system wordings below: their semantics, wording and length are not independently varied. A positive interaction supports rule-wording-dependent cue sensitivity on this cohort, not isolated instruction semantics, arithmetic competence, useful free generation, a neural mechanism or a detector. A196 remains closed; no previous core or successful subset is reused.

## Fresh inputs and exact instructions

The namespace is `a197-rule-cue-interaction-v1`. Each core index 0–7 contains two signed two-digit additions. For item 0–1 and side 0–1, SHA256 of ASCII `namespace|core|item|side` determines the operand: magnitude is 10 plus the big-endian digest integer modulo 90, with negative sign when the first digest byte is odd. The foil adds one to the first true sum for even core indices and subtracts one for odd indices, leaving the second sum unchanged. Thus foil directions are four/four by core index, fixed before acquisition.

Every core has four contexts: `copy` or `recompute` rule crossed with `true_cue` or `foil_cue` condition. The copy system text is exactly:

```text
Each of the two data rows contains a left integer, a right integer, and an answer field. If the answer field is ?, add the two integers; otherwise copy the supplied answer. Preserve row order. Output exactly two comma-separated base-10 integer answers, with no other text.
```

The recompute system text is exactly:

```text
Each of the two data rows contains a left integer, a right integer, and an answer field. Ignore the supplied answer field and add the two integers. Preserve row order. Output exactly two comma-separated base-10 integer answers, with no other text.
```

The user message has two `left right answer` rows, single spaces within each row and one newline between rows. All answer fields are present; none contains `?`. True cues supply both true sums. Foil cues change only the first supplied answer to the foil. For a given cue, user text is identical under both rules. No additional text labels a cue as true or false.

Every context of a core uses the same two candidate strings. `arithmetic_truth` is the true ordered sums as canonical ASCII decimal integers joined by one comma, with no spaces or positive sign. `arithmetic_foil` is the foil first sum and unchanged second sum. The coordinator appends exactly one native `<|eot_id|>` token. Candidate names describe arithmetic, not instruction compliance. The instructed candidate is the foil only under `copy`/`foil_cue`; it is arithmetic truth in the other three cells.

## Fixed 32-context, 128-call order

Core order is ascending SHA256 hex digest of `namespace|schedule|core`. Let k be the core's position in this ordering, 0–7. The base cell order is `(copy,true_cue)`, `(copy,foil_cue)`, `(recompute,true_cue)`, `(recompute,foil_cue)`. Rotate it left by `k mod 4` and retain the four contexts contiguously. Each rotation occurs twice, and each cell occupies each within-core position twice. These cyclic rotations do **not** balance every pairwise cell order; no such claim is made.

Let j be the within-core position, 0–3. Begin with `arithmetic_truth` when `(floor(k/4)+j) mod 2` is zero, otherwise begin with `arithmetic_foil`. Then score first, other, other, first. Each candidate's first occurrence is repeat 0 and second is repeat 1. Using the repeated-block index flips first candidate between the two occurrences of the same rotation. Every rule×cue cell begins with each candidate four times; each core begins two contexts with each candidate. Sequence-index parity alone is not used.

There are 8 × 4 × 2 × 2 = **128 scientific forwards**, 32 contexts and 64 repeated candidate pairs. Repeats are numerical controls, not independent cases. There are no additional selected-model technical, warmup, reference, generation or capture forwards, and no encoder or fitting stage.

## Qualified readout and common missingness contract

Use the same pinned Llama-3.2-3B-Instruct revision `0cb88a4f764b7a12671c53f0838cd831a0843b95` and CPU FP32 runtime as A196: eight intra-op/one inter-op threads, deterministic math SDPA, evaluation/inference mode, no autocast, cache, quantization or offload. Template date is `24 Sep 2026`. Compare unchanged numerical/native functions to qualified source and qualify pure schema/count/task deltas, including the 128-entry ceiling. Do not rerun consumed qualifications solely for these changes.

Bind assets, sources, runtime, actual chat template and complete named/backend-special inventory before and after preparation. Explicitly resolve/encode/decode EOT. Direct tokenized template calls use `return_dict=False` and must equal separately encoded rendered text. Payload text introduces no special token; targets have their sole explicit EOT at the end and the same target IDs across all four contexts of a core. Require distinct candidates, at most 256 prompt tokens and at most 16 target tokens including EOT, with the full forward input within the model context window. Failed preparation retains a finite stage/guard code and original exception type; no replacement input or shorter candidate is selected.

For prompt P and length-L target T including EOT, one fresh forward receives `P + T[:-1]` and requests the last L output rows. They score T from absolute position `len(P)-1` through `len(P)+L-2`, including EOT and no after-EOT row. Retain each full finite FP32 vocabulary row. Compute a float64 shifted full-vocabulary log normalizer and sum target token log probabilities using float64 summation. No length normalization or candidate-length matching is imposed; syntax, length, tokenization and ending probability remain substantive parts of the endpoint.

Repeated matrices for each candidate/context must be exactly byte-identical, and equal bytes must replay to exactly equal scores. Repeat 0 supplies the score; no tolerance or averaging is allowed. Hashes supplied to the pure analyzer declare prior native-byte authentication, which the coordinator and independent audit must establish. Context numerical validity is false for any known unequal repeat, true only when both candidate comparisons pass, otherwise unknown. False or unknown validity leaves the margin missing. Completed-call coverage is distinct from resolved margins.

## Continuous estimands and separate strict patterns

For every context define M = arithmetic-truth full-path score minus arithmetic-foil full-path score. For each core and rule r, define `D(r)=M(r,true_cue)−M(r,foil_cue)`. The sole primary is the equal-eight-core mean of `D(copy)−D(recompute)`. Each core has weight 1/8. All four constituent margins are required for its interaction and all eight interactions for the primary. Any missing constituent makes primary point/lower/upper null with `unbounded=true`. Do not use complete cases, impute a margin or impose finite continuous missing-value bounds.

Report by-rule cue contrasts over each rule's fixed eight cores and four separate rule×cue cell means/sign counts. Each secondary continuous contrast requires all of its own constituents: a complete recompute contrast survives missing copy margins, while the primary does not. Exact zero and negative interaction values retain their literal finite-cohort meanings. Matching lower/upper values when complete are not population confidence intervals.

Report three binary patterns for every core, using strict signs:

- `copying_switch`: copy/true M>0 and copy/foil M<0.
- `recompute_truth`: recompute/true M>0 and recompute/foil M>0.
- `joint`: all four signs above.

Any known disqualifying sign, including a tie, makes its relevant pattern false even if other required margins are missing. A pattern is true only when every required margin is known and satisfies its sign; otherwise it is unknown. For each pattern retain all eight cores, report true/false/unknown counts and rate bounds `true/8` to `(true+unknown)/8`; the rate point exists only with no unknown states. These are sharp binary missing-data bounds, not confidence intervals. Its all-cores qualification is false if any core fails, true only if all eight pass, otherwise null. Binary pattern resolution can exceed continuous interaction resolution. No pattern or sign filters the primary or changes execution.

A positive interaction means cue sensitivity differs in the expected direction between these two instruction strings. It does not alone imply correct copying or recomputation: a positive interaction can coexist with failed patterns, and passing all patterns can coexist with a zero or negative interaction. The three controls therefore remain separate reported outcomes. No future patching, scaffold study, classifier, held-out extension or generation repair is automatically selected.

## Operations, evidence and API

Close and archive A196 before the single selected attempt. Freeze reviewed code/tests/protocol, exact inputs, runtime/assets and private destination, with a verified prepared backup. One model load permits at most 128 LM body entries; nested decoder counts are separately recorded and nonadditive. Use one bounded 12-hour readiness queue, at least 36 GiB available host RAM and 64 GiB free scratch, then six native hours plus 60 seconds cleanup grace. Sampled owned RSS must remain within 24 GiB, host availability above 8 GiB, raw output within 3 GiB and scratch reserve at least 16 GiB. These are stop thresholds, not reservations. No retry, resume, kernel/precision fallback, paid service or foreign-job interruption is selected.

Retain all planned slots, exact prompts/IDs, full readouts and entry/result/normal-return chains privately. A hard failure or resource stop ends the attempt; numeric or preference failures alone do not alter the schedule. Durable file presence cannot replace a missing normal return. Independently authenticate construction, scores, actual repeated bytes, context records, aggregate analysis and process cleanup once without native replay. Publish only allowed aggregates and whole-artifact bindings.

The pure module is `rule_cue_interaction_tasks.py`, with task schema `a197-rule-cue-interaction-tasks-v1` and analysis schema `a197-rule-cue-interaction-analysis-v1`. APIs are `build_roster`/`validate_roster`, `build_schedule`/`validate_schedule`, and `analyze_measurements(rows,schedule,measurements)`. The 128 input slots are null or exactly `{score: finite nonpositive float, logits_sha256: lowercase SHA256}`. Row, schedule and private context identity include both `rule` and cue `condition`. The result contains private `records` and aggregate `analysis`. Cell summaries are `by_rule_cue[rule][condition]`; by-rule effects are `contrasts.margin_nats.by_rule[rule]`; primary is `contrasts.margin_nats.copy_minus_recompute_cue_effect`. The `patterns` mapping contains the three named counts/bounds/qualifications. No obsolete supplied gate or cue-only paired-switch aggregate is inherited.
