# A189: finite arithmetic task load on Llama 3.2 3B with GPU BF16

This prospective study asks whether requesting two rather than four signed sums changes exact finite completion on a fixed fresh cohort, and what remains unresolved in a fixed t4 prefix. The selected target is `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, executed with direct single-GPU BF16 parameters and math SDPA. This is a new target and declared numerical regime. It estimates no model-size, precision or CPU-versus-GPU effect, and earlier consumed studies provide no control observations. It does not fit a detector, establish incremental internal information or reopen any earlier cohort.

There are **32 scientific generations: eight core blocks × two requested counts × two operand-magnitude packages**. Every slot remains. The single native model load first receives two fixed technical forwards. If those pass, the same load executes all science rows regardless of ordinary incorrect answers. A hard technical, infrastructure or resource failure stops the attempt and leaves remaining slots explicitly unattempted. There is no competence gate, retry, resume, target substitution, cap extension, generation warmup, internal capture, reference stage, encoder or fit. Maximum causal-LM body entries are **2,050: two technical plus 2,048 science**. Nested decoder entries are counted separately and are nonadditive.

## Fresh construction and fixed schedule

Use namespace `gpu3b-arithmetic-load-v1`, core indices 0 through 7, item indices 0 through 3 and operand sides 0 and 1. Hash ASCII `{namespace}|{core}|{item}|{side}` with SHA-256. Let n be the full digest as an unsigned big-endian integer. The sign is negative exactly when the low bit of the first digest byte is one. The absolute operand is `1+n%9` for `one_digit` and `10+n%90` for `two_digit`. Signs are shared across magnitude packages; operands are never zero. No output, historical input or rejection sampling enters construction.

Requested counts are 2 and 4. A two-item request uses the first two items of its corresponding four-item core. User rows are plain ASCII `a b`, separated by one newline, with no heading or trailing newline. The exact system messages are:

```text
For each of the two data rows, add its two integers. Preserve their row order. Output exactly two comma-separated base-10 integer sums, with no other text.
```

```text
For each of the four data rows, add its two integers. Preserve their row order. Output exactly four comma-separated base-10 integer sums, with no other text.
```

No additional instruction, message or reminder is included. The selected native template date is exactly **`23 Sep 2026`**, explicitly supplied rather than drawn from the current clock. Freeze the authenticated tokenizer/template and EOS definitions. Require rendered-text/token-ID agreement, payload special-token exclusion, correct native assistant closure and a prompt length at most 256 tokens. Native preparation failure does not authorize truncation, padding, changed values or rewritten wording.

Identifiers have the form `core_00_n2_one_digit`; the four variants share a core/group identity. Sort by the hexadecimal SHA-256 of ASCII `gpu3b-arithmetic-load-v1|schedule|{core}|{item_count}|{magnitude}`, with `(core_index,item_count,magnitude)` as tie breakers, then number slots 0 through 31. The order is independent of outputs. Eight fixed cores are not a random population sample.

Input-only strata retain positive/positive, negative/negative, opposite-sign and zero-sum counts. Same-sign operands have a units carry when their absolute units digits sum to at least ten. Opposite-sign operands have a units borrow when the larger absolute operand has the smaller units digit. These are descriptions of inputs, not observed difficulty, balancing targets or reasons to alter the roster.

## Declared runtime and integrated technical checks

Authenticate the selected checkpoint, tokenizer, complete source closure and runtime before native entry. Load once, directly into BF16 parameters on `cuda:0`, using streamed shard loading with no full CPU parameter replica. Fix SDPA's **math backend**, four CPU threads, deterministic settings, evaluation/inference mode, batch one, no autocast and TF32 disabled. Do not quantize, offload, migrate devices, cast a fully loaded FP32 model, select another kernel after failure or substitute a different model. Record actual parameter, buffer and native logit dtypes and device. BF16 parameters/native logits do not mean every operation or accumulator is BF16; math SDPA can use FP32 intermediates. No numerical equivalence to CPU BF16 or FP32 is claimed.

Before science generation, perform exactly two identical cache-free forwards on these messages:

```text
Follow the user's instruction.
```

```text
Reply with the digit 0.
```

The first is system content and the second user content. Use the same fixed native template/date, exact rendered-text/token-ID agreement, payload-special-token exclusion and context bounds. Each call receives the complete native prompt, all-one attention and sequential positions, with no target suffix, past cache, sampled continuation, hidden-state collection or attentions. These checks contribute no task outcome. They are two planned slots, not two guaranteed completed calls.

A pass requires synchronized successful CUDA completion, correctly shaped finite native BF16 last-token logits, exact equality of the two retained native readout byte arrays, faithful dtype-preserving publication, normal call return and clean hook/profiler removal. Retain the full native BF16 vector and its **bit-exact FP32 widening** for first-index argmax replay. Widening preserves BF16 values, ordering and ties; it is not an FP32 model forward. Repeated equality is an engineering guard, not an accuracy bound. Failure consumes the attempt without a separate native pilot or repair. Receipt counts distinguish entered calls, completed returns and unavailable/ambiguous evidence; CUDA errors discovered at synchronization invalidate usable completion even if Python previously returned a tensor.

## Finite endpoint and stopping facts

Generation uses deterministic first-index greedy argmax and a **64 sampled-token cap including EOS**. Every step is a new cache-free full-prefix forward. Count the EOS-producing step and do not call again after EOS or the cap. Retain sampled IDs and each full finite native BF16 next-token row plus its exact FP32 widening privately, permitting model-free first-tie argmax replay.

Success is integer 1 exactly when a completed generation terminates with a frozen allowed native EOS by token 64 and its exact non-EOS decoded content contains the required ordered sums. Split on literal ASCII commas. Each field, after stripping only ASCII space, tab, carriage return, newline, vertical tab and form feed, must match `[+-]?[0-9]+`. Compare mathematical integers; leading zeros, explicit plus and negative zero are allowed. Unicode digits/whitespace, wrappers, prose, extra or empty fields, wrong values and wrong order fail. The implementation normalizes digit strings without Python's decimal-to-integer length limit.

A completed non-EOS cap has label 0 even if its text contains correct answers. An incorrect/malformed completed EOS also has label 0. Infrastructure failure, interruption or missing completion evidence has label null; partial text does not rescue it. EOS at token 64 is valid termination. Earlier nonterminal EOS IDs and inconsistent stop receipts are rejected.

Endpoint availability and observed stopping facts are separate. A retained EOS step followed by decoding or parent-publication failure can establish `terminal_eos=true` while the endpoint stays unknown. Never-attempted rows have null terminal-EOS, cap and censoring fields. For an observed incomplete stream, censoring means no retained terminal EOS and fewer than 64 sampled tokens; it is a finite-horizon flag, not a label or safety claim. A failed stream of 64 tokens does not become a completed cap without its required completion evidence.

## Fixed t4 diagnostics

Here t4 is the first **four generated non-EOS tokens**. It is selected prospectively for the shorter two/four-answer task, without observing new outputs or comparing landmarks. It need not hide the answer or final outcome. Retain the exact prefix decode without special-token stripping or cleanup; no capture or extra forward is needed. EOS after four content tokens leaves t4 reached; EOS before four makes it unreached. An incomplete stream with at least four retained content tokens establishes reachability; fewer than four without EOS leaves reachability unknown. No alternative boundary or new decoding pass rescues missing publication evidence.

Only comma-closed fields already present in the prefix are adjudicated. A closed malformed or incorrect field is a known irreversible error. The requested-count-th comma is also an error because another field is now required by the grammar. Never adjudicate the open tail using a later delimiter, even if it appears complete or incorrect. `remaining_fields=max(item_count-completed_fields,0)`. `unflagged_remaining_ge2` is true exactly when no closed-field error is detected and at least two required fields remain unclosed. This is descriptive; it does not establish that the eventual label is unpredictable from visible text.

If the exact prefix text is unavailable, all four prefix-check fields stay null, even if reachability is known. Known early EOS is a nonmember of the reached/unflagged/remaining-at-least-two subgroup. Missing flags on a reached or unknown row leave membership unknown. Keep reached/unreached/unknown counts, error true/false/unknown counts, closed/remaining-field histograms and unknown counts on fixed row denominators. For known subgroup members retain success/failure/unknown labels and supporting core counts, with unknown membership separately visible. There is no subgroup replacement rate, score or filter on either contrast.

## Estimands and sharp missingness bounds

Let `Y(c,n,m)` be the binary finite endpoint for core c, requested count n and magnitude m.

| Role | Estimand | Fixed cohort |
|---|---|---|
| Primary | Mean `Y(c,2,m)-Y(c,4,m)` | 16 pairs, 32 outcomes, coefficients ±1/16 |
| Secondary | Mean `Y(c,n,one_digit)-Y(c,n,two_digit)` | 16 pairs, 32 outcomes, coefficients ±1/16 |

The primary path is `contrasts.success_rate.shorter_minus_longer`; the secondary is `secondary_contrasts.success_rate.one_digit_minus_two_digit`. Each returns `point`, `lower`, `upper`, `planned_outcomes`, `resolved_outcomes`, `planned_pairs` and `resolved_pairs`. Any required unknown makes the point null. A missing binary term with coefficient w contributes `min(0,w)` to the lower and `max(0,w)` to the upper bound. Resolved terms contribute their exact signed value to both. Fully unknown contrasts span −1 to +1; complete bounds collapse to the point. These are sharp missingness bounds, not confidence intervals.

Overall and `by_condition["2" or "4"][magnitude]` summaries preserve all 32 rows and eight rows per cell. Success-rate points are null when incomplete, with fixed-denominator binary bounds and success/failure/unknown counts. Stopping availability and distinct core support remain explicit; core-support categories can overlap, and variants are not independent replications. No complete-case analysis, interaction estimand, population significance test, winning-cell selection or observed-subset replacement is introduced.

## Public API and qualification

`gpu3b_arithmetic_task_load.py` is pure standard-library code. It defines `ITEM_COUNTS=(2,4)`, `LANDMARK=4`, `CAP=64`, `CHAT_TEMPLATE_DATE="23 Sep 2026"`, `TECHNICAL_FORWARDS=2` and `MAX_TARGET_ENTRIES=2050`. `technical_messages()` returns fresh dictionaries. `build_roster()` (also `compile_roster`) returns fresh containers; `validate_row()` and `validate_roster()` reject changed fields, types, identities, order or slots. Exact row fields are `observation_id`, `core_index`, `core_id`, `group_id`, `item_count`, `magnitude`, `operands`, `expected_sums`, `strata`, `messages`, `schedule_sha256` and `sequence_index`.

`oracle_response(text,expected_sums)` checks grammar/arithmetic; `prefix_check(text,expected_sums)` returns `completed_fields`, `known_error`, `remaining_fields` and `unflagged_remaining_ge2`. `outcome_for_generation(row,generation,decoded_text,eos_token_ids)` separates label availability from stopping knowledge. Its generation input includes `token_ids`, `observed_tokens`, `status` and `stop_reason`; acquisition authenticates the complete receipt. `score_generation(row,generation,decoded_text,prefix_text,eos_token_ids)` returns `outcome_status`, `label`, `terminal_eos`, `capped`, `censored`, `prefix_reached` and the four prefix fields. `aggregate_records(rows,records)` (also `analyze`) requires every row in roster order and validates exact identity/types. Extra private acquisition receipt fields are excluded from public aggregates. Schema identifiers are `a189-arithmetic-task-load-v1` and `a189-task-load-analysis-v1`.

Synthetic controls cover independent hash construction, exact two/four domains, first-two subset matching, t4 versus EOS boundaries, grammar, missing publication/stopping distinctions, conservative prefix counterexamples, exhaustive missing-outcome assignments, all-slot validation and aggregate privacy. They call no pretrained model or native tokenizer. CUDA-specific analytical and tiny untrained qualification must additionally cover native BF16 byte preservation, widening/argmax/ties, synchronization failures, repeated technical checks, device/backend rejection and truthful call accounting. Source tests and opaque asset pins alone do not prove native execution. Raw prompts, responses, IDs, vectors and per-item receipts remain private outside public source control.

## Resources, stopping and interpretation

One selected resource queue lasts at most **12 hours**. Native load requires at least **12 GiB available host RAM, 9 GiB free GPU memory and 64 GiB free scratch**. The native deadline is **two hours plus 60 seconds cleanup grace**. Sampled stops are owned host RSS above 8 GiB, owned GPU use above 8 GiB, host available below 4 GiB or free GPU memory below 768 MiB. Keep a 3 GiB raw-run ceiling and 16 GiB staging/archive reserve, with authenticated model-asset storage budgeted separately. These are stop thresholds, not instantaneous allocation caps, capacity reservations or throughput promises. CUDA allocator and driver measurements require truthful separate accounting; unrelated workloads remain untouched. Signal only the owned process session. Queue expiry, OOM, resource stop or technical failure consumes the attempt without requeue, CPU fallback or precision/target substitution.

A positive complete primary supports the finite requested-count package contrast in this target/runtime/cohort. Count changes input length, arithmetic load and required output together, so no isolated mechanism is identified. Magnitude changes both representation and arithmetic structure. Zero at a shared floor/ceiling is not equivalence, and a negative contrast is a valid opposite-direction result. Incomplete acquisition retains its bounds and cannot establish model or detector invalidity.

Class variation and unflagged failures across several cores may inform a separately selected fresh detector study. Variation confined to task packages, already-visible errors or early-completed responses supports a narrower recognition/coverage conclusion. No automatic change of task, cap, landmark, model or kernel follows a floor, ceiling or uninformative prefix. This cohort is not silently reused as confirmatory detector evaluation. A later BF16 internal-feature study needs an explicitly dtype-preserving capture design; casting into the closed FP32 capture contract is not authorized. No result here establishes safety detection, hidden knowledge, a CPU/GPU effect or a cross-model causal effect.
