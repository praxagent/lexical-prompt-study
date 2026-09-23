# A188: finite arithmetic task load under an explicit CPU BF16 regime

This prospective study asks whether requesting four rather than eight arithmetic answers changes exact finite completion on a fixed fresh cohort, and whether the retained t8 prefix leaves outcomes unflagged with at least two fields not yet closed. It is a task-response and endpoint-feasibility experiment under a newly declared CPU BF16 realization of the same pinned Llama-3.1-8B-Instruct checkpoint. It estimates neither FP32 equivalence nor a precision effect; incomplete earlier FP32 cohorts are not control arms. It does not fit a detector, measure incremental internal information, reopen an earlier cohort, or repair a consumed experiment. Native acquisition requires the reviewed, qualified and frozen source package and the selected resource gates. Source qualification alone makes no claim about model performance.

The entire cohort is **32 generations: eight core blocks × two requested counts × two operand-magnitude packages**. Every slot is retained. The single model load first receives two fixed technical forward checks, defined below. There is no task-competence gate, condition selection, additional scientific prompt, retry, resume, cap extension, warmup, internal capture, reference stage, encoder or fit. A hard operational or technical-check failure stops acquisition and leaves the remaining slots explicitly unattempted; an ordinary incorrect science answer does not alter the schedule. The maximum is **2,050 target body entries: two technical checks plus 2,048 science forwards**, counting the EOS-producing step and including no forward after EOS or the cap. Nested decoder entries are recorded separately and are nonadditive.

## Frozen inputs and schedule

Core indices are 0 through 7, item indices 0 through 7, and operand sides 0 and 1. Hash the ASCII bytes `bf16-arithmetic-load-v1|{core}|{item}|{side}` with SHA-256. Let `n` be the whole digest interpreted as an unsigned big-endian integer. The sign is negative exactly when the low bit of the first digest byte is one. The absolute operand is `1 + n % 9` in `one_digit` and `10 + n % 90` in `two_digit`. Thus signs are shared across the two magnitude packages; no zero operand is generated. No outcome, historical input or rejection sampling enters construction.

Requested counts are `4` and `8`. A four-item request uses the first four items of its corresponding eight-item core. Each row is plain ASCII `a b`; rows are joined by one newline with no heading or trailing newline. The exact system messages are:

```text
For each of the four data rows, add its two integers. Preserve their row order. Output exactly four comma-separated base-10 integer sums, with no other text.
```

```text
For each of the eight data rows, add its two integers. Preserve their row order. Output exactly eight comma-separated base-10 integer sums, with no other text.
```

The user message is the corresponding plain data rows. There are no additional messages or reminders. The fixed native template and EOS definitions are bound by the acquisition plan. Native rendered text and token IDs must agree under the inherited native-rendering checks; prompt length must be at most 256 tokens. A preparation failure does not authorize truncation, padding, replacement values or changed wording.

An observation identifier has the form `core_00_n4_one_digit`. All four variants share its core and group identifier. The execution key is the hexadecimal SHA-256 of ASCII `bf16-arithmetic-load-v1|schedule|{core}|{item_count}|{magnitude}`. Sort ascending by that key, with `(core_index, item_count, magnitude)` as deterministic tie breakers, and number the resulting slots 0 through 31. This schedule is independent of outputs and does not make the eight fixed cores a random population sample.

Input-only strata retain counts of positive/positive, negative/negative and opposite-sign pairs. A units-column carry is recorded for same-sign operands when their absolute units digits sum to at least ten. A borrow is recorded for opposite-sign operands when the larger absolute operand's units digit is smaller than the other units digit. Zero-sum counts are also retained. These descriptors are not observed difficulty, balancing targets or grounds for changing the roster.

## Declared numerical regime and two technical checks

Load the original pinned checkpoint **directly into CPU BF16 parameters**, with SDPA, four threads, deterministic settings and no autocast. Do not load FP32 parameters first and cast afterward. The fixed policy permits no quantization, device migration, offload, alternate target, dtype fallback or kernel search. Authenticate the checkpoint/tokenizer assets and exact runtime. Record actual parameter, buffer and native logit dtypes. BF16 parameters do not establish that every internal operation or accumulator runs in BF16. Hardware BF16 acceleration and a completion time are not assumed.

Before any science generation, perform two identical cache-free forwards on these exact messages:

```text
Follow the user's instruction.
```

```text
Reply with the digit 0.
```

The first is the system message and the second is the user message. Use the same frozen native template, authenticating its hash, exact rendered-text/token-ID agreement, literal messages, payload special-token exclusion and prompt/context bounds. The science preparation separately validates appended assistant closure and native EOS; the technical check samples no continuation. Each call receives exactly the same complete native prompt, all-one attention and sequential positions, with no target suffix, past cache, generated token, hidden-state collection or attentions. Retain the two planned technical slots and count actual model body entries; a failure can leave the second check and all science slots unattempted. These checks sample no completion and contribute no task outcome.

A passing technical check requires correctly shaped finite native BF16 output on CPU, exact repeated native readout bytes, faithful byte-preserving evidence, normal call return and hook/profiler cleanup. Retain the full native BF16 last-token logit vector as dtype-preserving bytes, plus its **exact FP32 widening** for first-index-argmax replay. Widening each BF16 value into its exactly representable FP32 value must preserve every value, ordering and tie; it is not an FP32 model evaluation. The repeated readout equality is a fixed engineering guard, not an accuracy/error bound or competence test. An unsupported kernel, dtype/device mismatch, nonfinite output, readout disagreement or unusable receipt stops the run without repair. When both checks pass, the same loaded model executes the complete fixed science schedule.

## Finite endpoint and stopping facts

Generation is deterministic greedy, cache-free, CPU BF16 SDPA, with four threads, no autocast and a **64 sampled-token cap including EOS**. The acquisition component retains sampled IDs, full finite native BF16 next-token rows and their exact FP32 widenings for pure first-index-argmax replay. Technical and science entries have separately retained accounting and a shared total ceiling of 2,050. The task module does not establish native decoding or model-call provenance; acquisition and its independent replay must bind those facts.

Success is integer `1` exactly when a completed generation terminates with an allowed native EOS by token 64 and its exact non-EOS decoded content contains the requested number of correct ordered sums. Split the text on literal ASCII commas. Every field must match `[+-]?[0-9]+` after stripping only ASCII space, tab, carriage return, newline, vertical tab and form feed. Compare mathematical integer values, allowing leading zeros, explicit plus and negative zero. Unicode digits, non-ASCII whitespace, wrappers, prose, empty fields, extra fields and wrong order fail. The implementation normalizes digit strings instead of imposing Python's decimal-to-integer length limit.

A completed 64-token non-EOS cap has label `0`, even if its text happens to contain correct answers. A completed EOS with an incorrect or malformed answer also has label `0`. Infrastructure failure, interruption, absent generation or missing required completion evidence has label `null`; partial text cannot rescue it. Earlier nonterminal EOS IDs and inconsistent stop receipts are rejected. EOS at token 64 is valid termination, rather than a cap.

Observed stopping facts and endpoint availability are separate. A retained EOS step followed by decode or parent-publication failure can establish `terminal_eos=true` while the endpoint remains unknown. For a never-attempted slot, terminal-EOS, cap and censoring fields are all null. For an observed incomplete stream, censoring means no retained terminal EOS and fewer than 64 retained sampled tokens; it is a declared finite-horizon flag, not a label or safety conclusion. A failed stream of length 64 without a completed cap receipt is not silently promoted to a completed capped outcome.

## t8 diagnostics from retained prefixes

Here t8 means the first **eight generated non-EOS tokens**. It requires no capture or additional model forward. Retain their exact decoded text without special-token stripping or cleanup, as authenticated by acquisition. An EOS after those eight content tokens leaves the landmark reached. EOS before eight establishes `prefix_reached=false`. At least eight retained content tokens establish `true`, even if later work fails. An incomplete stream shorter than eight leaves reachability unknown. No substitute prefix, later boundary or new decoding pass is introduced to rescue missing publication evidence.

The checker examines only fields closed by commas already present in the t8 decoded text. It counts all such fields. A closed malformed or wrong field is an irreversible error; the requested-count-th comma is also an error because an additional field is now required by the grammar. The open tail is never adjudicated, even if it appears incorrect or complete. Required fields not yet closed are `max(item_count - completed_fields, 0)`. The descriptive flag `unflagged_remaining_ge2` is true exactly when there is no detected closed-field error and at least two required fields remain unclosed. An unflagged prefix does not establish that the final outcome is unforeseeable from other visible information.

If exact prefix text is unavailable, `completed_fields`, `known_error`, `remaining_fields` and `unflagged_remaining_ge2` are all null, even when reachability is known. A known unreached row is a nonmember of the reached/unflagged/remaining-at-least-two subgroup; an unavailable flag on a reached or unknown row leaves subgroup membership unknown. Report these distinctions explicitly. No stopping fact, missing prefix or prefix error filters either endpoint contrast.

## Fixed estimands and missingness

Let `Y(c,n,m)` be the binary finite endpoint for core `c`, requested count `n`, and magnitude `m`.

| Role | Estimand | Fixed coverage |
|---|---|---|
| Primary | Mean of `Y(c,4,m) - Y(c,8,m)` | 16 pairs; 32 outcomes; coefficients ±1/16 |
| Secondary | Mean of `Y(c,n,one_digit) - Y(c,n,two_digit)` | 16 pairs; 32 outcomes; coefficients ±1/16 |

The primary key is `contrasts.success_rate.shorter_minus_longer`; the secondary is `secondary_contrasts.success_rate.one_digit_minus_two_digit`. Each has `point`, `lower`, `upper`, `planned_outcomes`, `resolved_outcomes`, `planned_pairs` and `resolved_pairs`. A point is null if any required endpoint is unknown. For each missing binary outcome with coefficient `w`, add `min(0,w)` to the lower bound and `max(0,w)` to the upper bound. Resolved terms contribute their exact signed value to both bounds. These sharp missingness bounds collapse when complete and are not confidence intervals. They range from −1 to +1 when everything is missing. No complete-case contrast or observed-subset rate replaces the fixed estimand.

The aggregate includes all 32 planned rows and a four-cell table, `by_condition["4" or "8"][magnitude]`, with eight planned rows in every cell. Overall and cell success rates use their fixed planned denominators, null points when incomplete, and binary missingness bounds. Retain success/failure/unknown counts, stopping availability and distinct core support. Core support categories may overlap; multiple variants of a core are not independent replicates.

Prespecified descriptive prefix summaries use the same fixed row denominators: reached/unreached/unknown counts, known-error true/false/unknown counts, completed- and remaining-field histograms with separate unknown counts, and reached/unflagged/remaining-at-least-two membership counts. For known subgroup members, retain success/failure/unknown labels and supporting core counts. Unknown membership remains explicit. This subgroup has no replacement point estimator or new prediction score, and no absent row is credited as a success. It cannot supersede the full-cohort contrasts. There is no interaction estimand, confidence interval, hypothesis test or outcome-driven selection of a cell.

## Public interface and qualification

`bf16_arithmetic_task_load.py` contains only pure standard-library code. `technical_messages()` returns fresh standard messages for the two identical technical forwards; `TECHNICAL_FORWARDS=2` and `MAX_TARGET_ENTRIES=2050` are declarative bounds. `build_roster()` (also `compile_roster`) returns fresh containers; `validate_roster()` and `validate_row()` reject changed fields, types, identities, schedule or slots. Row fields are `observation_id`, `core_index`, `core_id`, `group_id`, `item_count`, `magnitude`, `operands`, `expected_sums`, `strata`, `messages`, `schedule_sha256` and `sequence_index`.

`oracle_response(text, expected_sums)` tests grammar and arithmetic. `prefix_check(text, expected_sums)` returns the four descriptive prefix fields. `outcome_for_generation(row, generation, decoded_text, eos_token_ids)` separates the endpoint from stopping knowledge. Its generation input has `token_ids`, `observed_tokens`, `status` and `stop_reason` as used by the acquisition envelope; the task function validates the relevant fields, while acquisition binds the complete receipt.

`score_generation(row, generation, decoded_text, prefix_text, eos_token_ids)` returns `outcome_status`, `label`, `terminal_eos`, `capped`, `censored`, `prefix_reached` and the four prefix fields. `aggregate_records(rows, records)` (also `analyze`) requires all 32 records in roster order and validates exact identity and score types. Additional acquisition receipt fields may be retained privately but do not enter the aggregate. It emits counts, histograms, fixed-cohort bounds and contrasts, without per-item identifiers, hashes, text, token arrays or logits.

Synthetic qualification uses invented completions and receipt metadata, independent hash construction, grammar and EOS edge cases, conservative prefix counterexamples, exhaustive missing-outcome assignments, all-slot/identity guards and aggregate privacy checks. It makes no pretrained model, tokenizer, API, encoder or fitting call. The producer also needs analytical and tiny untrained CPU BF16 qualification of finite native bytes, exact widening/argmax/ties, repeated technical checks, dtype/device rejection and failure accounting. These untrained controls are apparatus evidence only. Native generation implementation and operational qualification are separate parts of the reviewed package; neither this pure module nor untrained controls establish pretrained execution success. Raw evidence remains private outside public source control.

## Resource bound and interpretation

The selected local policy allows one resource queue lasting at most 12 hours for this new study, with no target entry until at least 48 GiB available host RAM and 64 GiB scratch are available. Acquisition has a six-hour wall deadline plus 60 seconds termination grace. Owned RSS has a sampled 32 GiB stop threshold, not an instantaneous kernel allocation cap; host available RAM below 8 GiB also stops it. Raw output is bounded at 3 GiB, with 16 GiB reserved for staged copies and archives. The controller owns and signals only this run's process session. Cooperative study exclusivity does not reserve host RAM against unrelated work. Queue expiry or a resource/interruption stop yields a deferred or incomplete study without retries or precision substitution. These limits are fixed new-regime caps, not a predicted footprint or throughput promise. The same single load covers the two technical forwards and 32 scientific generations; no separately loaded native pilot is authorized by this protocol.

A positive complete primary supports the finite requested-count package comparison in this declared CPU BF16 realization of the pinned checkpoint. Shorter lists change input length, arithmetic load and required output burden together; they do not isolate any one mechanism. The magnitude comparison changes digit representation and arithmetic structure. A resolved zero at a shared floor or ceiling is not equivalence, and a negative contrast is a valid opposite-direction result.

Mixed outcomes and unflagged failures on several fresh cores may inform a separately selected future detector study. Overall class variation caused solely by different task packages is weak evidence for predicting still-unrevealed errors, so the fixed cell and prefix/core counts matter. Frequent early EOS or already-flagged failures may make t8 unsuitable for that future claim. No automatic change of landmark, task wording, cap, difficulty or selected cell follows. An all-zero or all-one result closes this bounded task-load question honestly. These 32 observations are not silently reused as a confirmatory detector evaluation. An eventual BF16 detector acquisition would need its own explicitly dtype-preserving internal-evidence design; the existing FP32-only capture contracts cannot be silently satisfied by casting. No conclusion here establishes agreement with the consumed FP32 studies. None of these outcomes establishes a safety-proxy result, hidden knowledge, internal mechanism, free-generation rescue, or comparative detector validity.
