# A174: fresh closing-constraint crossover

This is a new prospective 32-generation experiment. A173 verified 31 strict
successes among 32 legacy-system outputs, with no caps or fence markers and
valid JSON/EOS for all 32. Its A171-standard system produced zero successes
among 32 outputs, all capped with fence markers. The heading and value-form
contrasts each reflected a single legacy other-mapped-value response. These
findings support a large complete-system-wording difference on those fixed
inputs; they do not isolate any sentence, Markdown prohibition, token length,
semantic cause or internal mechanism. No A174 outcome is asserted here.

A174 asks whether the original final constraint sentence changes strict output
behavior across the two otherwise complete instruction cores. It crosses the
legacy instruction's first four sentences and the A171-standard instruction's
first two sentences with each original final sentence. Every core already
requests the selected value and its JSON response format. Both crossovers have
clear value references, and the two matching combinations exactly reconstruct
the original system messages. The cores and closing sentences are wording
packages, not isolated semantic interventions. There is no stimulus repair,
wording search, parent-output audit or reopening of an earlier study.

## Exact public worlds and instruction components

Use the following four new public worlds in the displayed order. Render only
their compact two-entry JSON mappings as user messages, preserving AB/BA order.

| World | Value under A | Value under B | Presentation order |
|---|---|---|---|
| w00 | s40 | s47 | AB |
| w01 | s44 | s41 | BA |
| w02 | s43 | s46 | BA |
| w03 | s45 | s42 | AB |

All eight values are distinct, differ from A/B, and are outside the earlier
public constructor vocabulary and A173 values. They are fixed by this protocol,
without opening historical sealed worlds or private parent stimuli or outputs.
Do not transform, prefix, replace or select values using tokenization or results.

Cross `instruction_core` and `closing_constraint`, each in level order `legacy`,
then `a171_standard`, and both selectors in order `A`, then `B`. Use these four
single-line ASCII component templates, in this exact order.

Legacy core, exactly the original instruction's first four sentences:

```text
The authoritative query selector is {selector}. The user supplies two explicit answer values labeled A and B. Copy the value with the authoritative label. Output exactly one JSON object with the single field "answer" containing that value as a string.
```

A171-standard core, exactly the original instruction's first two sentences:

```text
Select the entry labeled {selector} from the user's mapping. Return that entry's mapped value as exactly one JSON object with the single key "answer" and a string value.
```

Legacy closing constraint, exactly its original final sentence:

```text
Do not transform the value or include other fields, explanations, or Markdown fences.
```

A171-standard closing constraint, exactly its original final sentence:

```text
Do not return any other text.
```

Substitute the active uppercase selector into the core's one `{selector}` slot.
Join that core and the selected closing sentence with exactly one ASCII space.
Add no leading/trailing whitespace or punctuation. The legacy/legacy diagonal
must equal the original public `tasks.system_instruction(selector)` byte for
byte; the a171_standard/a171_standard diagonal must equal the original A171
`STANDARD.format(selector=selector)` byte for byte. Freeze all four combinations
for both selectors before any generation. No common bridge sentence is added.

The user message is exactly `json.dumps(mapping, separators=(',', ':'))` with
keys inserted in the world's displayed AB or BA order. It contains no heading,
value prefix, prose, reminder, example, trailing newline or additional request.
Within a world, all eight cells have byte-identical user messages. The correct
answer is the string under the requested selector; the other value is preserved
for strict error classification. Record native token lengths, but never use
lengths as a reason to change a component, world or schedule. Length, tokenization
and wording differences remain part of these fixed input treatments.

## Fixed 32-call schedule

Define the eight-element Cartesian product with selector varying fastest:

```python
base = list(product(
    ('legacy', 'a171_standard'),
    ('legacy', 'a171_standard'),
    ('A', 'B'),
))
rotated = base[4:] + base[:4]
world_schedules = (base, list(reversed(base)), rotated, list(reversed(rotated)))
```

The product axes are instruction core, closing constraint and selector, in that
order. Concatenate the four world schedules in w00--w03 order. Exactly 32 unique
cells result, with eight observations per core/constraint combination. Each of
the 28 pairs of condition triples appears in each relative order in two of the
four worlds. AB/BA presentation is balanced overall and within forward/reverse
schedules. Both selectors appear in every setting. Execute the entire frozen
schedule within the resource/time bound, without a baseline gate, retries,
outcome-dependent stopping, new values or extra samples.

## Unchanged strict scoring and fence marker

Reuse the pinned A173 preparation, original-reference greedy generation and
strict scorer including its single `fence_marker` extension. Preserve exactly
the existing categories `correct`, `selected_label`, `other_label`,
`other_mapped_value`, `other_or_format`, `cap`, `infrastructure`, and `missing`.
Strict correctness requires exactly one JSON object with the single key `answer`
and a string equal to the selected mapped value, plus valid terminal EOS.
Surrounding JSON whitespace is allowed; duplicate/additional keys, wrong types,
other shapes, trailing non-whitespace, and fence-wrapped JSON fail strict format.

Terminal EOS must belong to the frozen EOS list and have no earlier member of
that list in the generated sequence. Remove only the final EOS for decoding.
EOS at token 64 is valid; 64 generated tokens without terminal EOS is a cap,
including when decoded text resembles a correct answer. Cap precedence and
invalid-termination/format precedence are unchanged. Label and wrong-value
categories require valid JSON shape and termination.

Retain `strict_correct`, `selected_label`, `format_valid`, `eos_valid`, `capped`,
`attempted`, and `fence_marker`; unavailable behavioral indicators are null.
The fence indicator is true exactly when decoded response text contains a
literal ASCII triple-backtick or triple-tilde substring anywhere. Inline,
quoted, unpaired and longer runs count. No detector for fence pairing, embedded
objects, repetition, intent, hidden answer correctness or mechanisms is added.
Fences never rescue strict accuracy and never trigger an adaptive change.

## Two primary main effects and one secondary interaction

Let `Y[c,k,w,s]` be binary strict correctness for core `c`, closing constraint
`k`, world `w`, and selector `s`; `-` means legacy and `+` means a171_standard.
The two primary contrasts, each with 16 matched pairs and all 32 outcomes, are:

1. `instruction_core`: a171_standard minus legacy, averaging equally over
   closing constraints, selectors and worlds.
2. `closing_constraint`: a171_standard minus legacy, averaging equally over
   instruction cores, selectors and worlds.

Each primary matched difference has weight 1/16. A positive contrast favors
the a171_standard level of that factor on these fixed cells.

Prespecify one secondary strict-accuracy interaction, to distinguish an average
closing-sentence difference from a difference depending on the core. Within
each of the eight world/selector quartets, calculate:

`Y[+,+] + Y[-,-] - Y[+,-] - Y[-,+]`.

Average these eight quartet values equally, giving each signed outcome
coefficient +1/8 or -1/8. Equivalently, this is the a171_standard-minus-legacy
closing-constraint effect under the a171_standard core minus that effect under
the legacy core. Its possible range is [-2, 2]. It is a secondary descriptive
factorial contrast; it does not identify an internal mechanism or an isolated
Markdown instruction effect.

All contrasts use the entire planned cohort. If every required outcome is
resolved, report point/lower/upper equal to the exact finite-design contrast.
If any is unresolved, the point is null and sharp bounds assign every unknown
binary outcome independently zero or one with its signed coefficient. Do not
clip the interaction to [-1, 1], impute unknown outcomes as failures, renormalize
remaining weights, drop pairs/quartets, or report a complete-case estimate.
Primary metadata include `planned_outcomes=32`, `resolved_outcomes`,
`planned_pairs=16`, `resolved_pairs`. Interaction metadata include
`planned_outcomes=32`, `resolved_outcomes`, `planned_quartets=8`, and
`resolved_quartets`. A pair or quartet is resolved only if all its outcomes are
known. A completed cap is a known zero; infrastructure/missing remains unknown.

Report overall counts and the four core-by-constraint groups, each with eight
planned observations, using all original categories and indicator counts.
Strict-accuracy points are null for incomplete groups, with lower bound known
successes divided by planned size and upper bound adding unknown outcomes to
the numerator. Report planned/resolved denominators explicitly. Do not add
confidence intervals, significance tests, further interactions, selected-label
contrasts, or data-dependent conditional estimands. The four-cell table remains
available to expose the joint pattern.

Use `overall`, `by_condition[instruction_core][closing_constraint]`, and primary
`contrasts.strict_accuracy` keys `instruction_core` and `closing_constraint`.
The separately labeled secondary interaction uses
`secondary_contrasts.strict_accuracy.core_by_constraint_interaction`.

## Interpretation and stopping rules

This study compares two complete instruction cores and two complete final
sentences on four fixed worlds under one runtime. An effect can support a
behavioral difference for those exact text packages. It cannot establish
population transfer, isolated Markdown prohibition, semantic cause, token-length
control or a latent generation mechanism. Fences and caps are observed outputs,
not causal explanations. The repeated original diagonal conditions on new
worlds provide an explicitly scheduled transfer check without gating the mixed
conditions. No all-floor, all-ceiling, null, mixed or unexpected outcome permits
changing the protocol or reopening A169 or earlier sealed held-outs.

Stop after 32 attempts, the fixed deadline, interruption or terminal failure.
Recoverable evaluation errors remain infrastructure and are never retried;
continue only the remaining originally scheduled cells. Preserve all unknown
slots after interruption. Do not change the cap, expand the sample, edit a
prompt, add a detector, or make a new target call during export or review.

## Runtime, evidence and model-free qualification

Use the original local weights with the corrected pinned CPU FP32 reference,
CPU SDPA, four CPU threads, unchanged seed, batch size one, greedy decoding,
one beam, cache enabled, and at most 64 new tokens. Preserve the explicit frozen
EOS/pad/BOS settings and `logits_to_keep=1`. One fresh process loads the target
once. There is no warmup target call, sampling, constrained decoding, likelihood
measurement, external API, download, rental, GPU execution or new spending.

Bind original configuration SHA-256
`1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745`
and corrected reference-script SHA-256
`f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`.
The configuration's historical NF4 description does not change the effective
original-weight CPU FP32 loader. Preserve the pinned interpreter path without
resolving it:
`/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python`.
The pinned A173 helper source SHA-256 is
`9d197e0c91e63bfcc51b70016a19e1c7ac477ebecfc3e12405f594202413e075`.
Bind public helper implementation only; no private parent response is an input.

Freeze reviewed source, protocol, tests, plan, public helper dependencies and
all 32 native constructions before generation. Check exact system/user roles,
chat rendering and generation boundary, assistant-probe round-trip and EOS,
vocabulary bounds, and prompt length plus all 64 generation tokens against the
bound model context limit. A failed input audit does not authorize repairs.
Root handles the independent verifier, code-only Git backup/push/freeze,
execution and archive; qualification itself uses public synthetic fixtures.

Respect shared host jobs and require at least 48 GiB RAM available immediately
before loading. Use the existing reservation/exclusive lock. Enforce a
3,600-second internal wall deadline including preflight/loading, an external
3,600-second limit, and a 60-second kill grace. SIGTERM, SIGINT, SIGHUP and
SIGALRM terminate without entering recoverable error handling. Preserve actual
signal number/name and receiving PID when known; direct KeyboardInterrupt has
null signal provenance. Generic BaseException is not a recoverable forward.

Use a new absolute private run directory under
`/data2/PRAX/lexical-prompt-study-data/runs/a174/`, private permissions, immutable
startup and one-shot claims, and a process-local consumption guard. Write each
attempt before generation and result durably afterward. Preserve pre-header
startup failure receipts with zero target calls. Never resume or retry a
consumed run, even under a new output directory. Retain all 32 planned slots and
separate attempted/completed/infrastructure/interrupted/unattempted coverage.

Export under lock without target-model loading. Reconstruct every input,
schedule position, native receipt, score, fence flag, count and all three
signed contrasts/bounds; verify source/input/model hashes. Raw text, token
arrays, per-item identities and records stay private outside Git. Public output
contains reviewed aggregates and whole-artifact provenance only; CLI stdout
contains safe status and coverage.

Synthetic qualification covers the exact components and original diagonal
reconstruction; fresh worlds and 32-cell schedule; all 28 pair-order balances;
selector/user-message invariance; context/vocabulary/native EOS guards; strict
categories, termination, cap precedence and literal fence flags; both independent
16-pair primary formulas; the eight-quartet secondary formula, extrema +/-2 and
exhaustive signed missingness bounds; four groups and planned denominators;
no gate/retry/resume; startup/runtime signals/deadline; durable result receipts;
locks, tamper rejection, independent replay and privacy. No target tokenizer,
pretrained model call, raw prior response or original held-out input is used.
