# A176: normalized selection/format crossover with original controls

This is a new prospective 48-generation experiment. A174 verified perfect strict
accuracy for its legacy core with either closing sentence and zero strict
accuracy for its standard core with either closing sentence. Under the standard
core, the legacy closing produced eight terminated format failures, whereas the
standard closing produced eight caps. A175's separately frozen post hoc audit
found fence markers and no complete exact-schema answer-object substrings in
both standard-core groups, with no positives under its fixed repetition rule.
These findings describe the fixed outputs; they do not establish failed
selection, absent internal answers or a cause of the failures.

A176 asks how two selection-framing packages and two format-request packages
behave after replacing their cross-sentence value references with a common
explicit noun phrase. The normalized format requests repeat the selector. This
changes wording, tokenization and length, and makes the value request explicit
within the format sentence itself. The factorial therefore does not isolate
selection ability, pure formatting semantics or an individual word. Two
unchanged original controls measure the diagonal rewrites on the same fresh
worlds. No A176 outcome is asserted here.

## Exact public worlds and templates

Use these four worlds in the displayed order:

| World | Value under A | Value under B | Presentation order |
|---|---|---|---|
| w00 | s60 | s67 | AB |
| w01 | s64 | s61 | BA |
| w02 | s63 | s66 | BA |
| w03 | s65 | s62 | AB |

The eight distinct public values differ from A/B and earlier public values.
They are fixed without opening any historical sealed world, A169 context,
private prior prompt, response, token array or row-level record. There is no
world search, alternative vocabulary or input revision using tokenization or
outcomes.

Use the following seven single-line ASCII templates, in the displayed order.

Legacy selection block, exactly the first three sentences of the original
legacy instruction:

```text
The authoritative query selector is {selector}. The user supplies two explicit answer values labeled A and B. Copy the value with the authoritative label.
```

Standard selection block, exactly the first sentence of A171 STANDARD:

```text
Select the entry labeled {selector} from the user's mapping.
```

Normalized legacy format request:

```text
Output exactly one JSON object with the single field "answer" containing the mapped value of the entry labeled {selector} as a string.
```

Normalized standard format request:

```text
Return the mapped value of the entry labeled {selector} as exactly one JSON object with the single key "answer" and a string value.
```

Original legacy format request, used only in its original control:

```text
Output exactly one JSON object with the single field "answer" containing that value as a string.
```

Original standard format request, used only in its original control:

```text
Return that entry's mapped value as exactly one JSON object with the single key "answer" and a string value.
```

Common closing sentence, exactly the original legacy closing:

```text
Do not transform the value or include other fields, explanations, or Markdown fences.
```

The factor levels `legacy` and `a171_standard` respectively select the legacy
and standard selection/format templates. Substitute the active uppercase label
into every `{selector}` slot. Join the selection block, chosen format request
and common closing with exactly one ASCII space at each boundary, with no
leading/trailing whitespace. The four normalized cells share the exact noun
phrase `the mapped value of the entry labeled {selector}`; this removes the
mixed original `that value` / `that entry` antecedent problem. Every normalized
system message names the active selector twice. Each original control names it
once. Do not claim exact original reconstruction for a normalized diagonal.

The six conditions, in fixed order, are:

| condition | construction | selection_block | format_request |
|---|---|---|---|
| normalized_legacy_legacy | normalized | legacy | legacy |
| normalized_legacy_standard | normalized | legacy | a171_standard |
| normalized_standard_legacy | normalized | a171_standard | legacy |
| normalized_standard_standard | normalized | a171_standard | a171_standard |
| original_legacy | original | legacy | legacy |
| original_standard | original | a171_standard | a171_standard |

For `construction=normalized`, use the corresponding normalized format request.
For `construction=original`, use the corresponding original format request;
there are only the two matching original controls, with no mixed original
conditions. The `original_legacy` system message must equal the complete public
`tasks.system_instruction(selector)` byte for byte. The `original_standard`
system message must equal A174's standard core plus legacy closing byte for
byte. Equivalently, both original conditions equal the respective A174
`system_instruction(core, 'legacy', selector)` messages. These are the exact
A174 legacy-closing controls, not two exact A171 original full instructions.

The common legacy closing is fixed because A174 produced valid EOS under both
cores with it, enabling observation of uncapped format failures. It does not
guarantee valid termination, successful selection or accuracy on fresh worlds.
No common bridge, reminder, demonstration or new detector is added.

Render each world's user mapping as compact JSON, using double quotes and
`separators=(',', ':')`, with A/B keys inserted in the displayed presentation
order. The JSON object is the entire user message: no heading, prefix, prose,
trailing newline or other request. All 12 cells within a world have identical
user messages. The system specifies the selector; the selected and other
mapped values define the original strict oracle. Record native lengths without
padding, matching lengths, revising wording or selecting inputs from them.

## Fixed 48-call schedule

Let `conditions` be the six condition names in table order. Define the base
schedule with selector varying fastest:

```python
base = list(product(conditions, ('A', 'B')))
rotated = base[6:] + base[:6]
world_schedules = (base, list(reversed(base)), rotated, list(reversed(rotated)))
```

Concatenate schedules in w00--w03 order. There are exactly 48 unique cells and
eight observations per condition. The 66 pairs among the 12 condition/selector
combinations each occur in both relative orders twice across the four worlds.
Both selectors occur in every condition; AB/BA presentation is balanced overall
and within forward/reverse scheduling. Freeze all cells and ordering before
generation. Execute the entire schedule within fixed time/resource bounds,
without a control gate, conditional sample expansion or adaptive revision.

## Unchanged strict scoring

Reuse the pinned A174 native preparation, original-reference greedy generation
and strict scorer, including the existing literal fence indicator. Preserve
categories `correct`, `selected_label`, `other_label`, `other_mapped_value`,
`other_or_format`, `cap`, `infrastructure` and `missing`.

Strict correctness requires a complete JSON object containing exactly the key
`answer`, with a string equal to the selected mapped value and valid terminal
EOS. Allow surrounding JSON whitespace; reject duplicate/additional keys,
wrong types, other shapes, trailing non-whitespace and fenced JSON. A valid EOS
is a final member of the frozen EOS list with no earlier member in the generated
sequence. Remove only the final EOS for decoding. EOS on token 64 is valid;
64 tokens without terminal EOS is a cap even if the decoded text resembles a
correct answer. Preserve cap and invalid-termination/format precedence. Label
and wrong-value categories require valid format and EOS.

Retain strict correctness, selected-label, format/EOS/cap, attempted, and
`fence_marker` indicators; unavailable behavioral indicators remain null.
`fence_marker` means a literal ASCII triple-backtick or triple-tilde substring
anywhere in the saved response text, with no pairing or intent interpretation.
No embedded-object detector, repetition detector, score rescue, likelihood,
response-intent judgment or additional scoring feature is part of A176.

## Exactly two primary normalized-factor contrasts

Use only the 32 `construction=normalized` cells. Original-control rows must
never enter either primary contrast. Within these cells, match world, selector
and the other normalized factor. Each contrast contains 16 equally weighted
pairs and all 32 normalized outcomes:

1. `selection_block`: a171_standard minus legacy selection block.
2. `format_request`: a171_standard minus legacy normalized format request.

Each paired difference has weight 1/16. Average equally over both levels of the
other factor, both selectors and the four worlds. These effects describe exact
text packages in the normalized instructions with their repeated selectors.

## Exactly three secondary contrasts

Let `Y[s,f,w,q]` denote normalized strict correctness, with `-` for legacy and
`+` for a171_standard. The secondary `selection_by_format_interaction` is the
mean of `Y[+,+] + Y[-,-] - Y[+,-] - Y[-,+]` over the eight world/selector
quartets. Each signed outcome has coefficient +1/8 or -1/8; the range is
[-2, 2]. It uses exactly the same 32 normalized outcomes, with
`planned_quartets=8`. This is a descriptive package interaction.

The two secondary normalization comparisons use matching world/selector pairs:

- `normalization_legacy`: normalized_legacy_legacy minus original_legacy.
- `normalization_a171_standard`: normalized_standard_standard minus original_standard.

Each has eight pairs, 16 required outcomes and coefficient +/-1/8 per outcome;
the possible range is [-1, 1]. They measure the complete diagonal rewrite,
including explicit reference wording, repeated selector, length and
tokenization. They do not establish normalization neutrality in the mixed
conditions or isolate which aspect of the rewrite matters.

For every contrast, report `point`, `lower`, `upper`, `planned_outcomes` and
`resolved_outcomes`, plus planned/resolved pair counts or quartet counts as
appropriate. If every required outcome is known, point and bounds equal the
exact finite-design contrast. Otherwise the point is null, and sharp bounds
assign each unresolved binary outcome zero or one according to its signed
coefficient, without dropping rows, zero imputation or renormalization. Do not
clip interaction bounds to [-1, 1]. A pair/quartet is resolved only when all its
required outcomes are known. A completed cap is a known strict failure.

Missingness is local to each prespecified contrast's required cells. In
particular, a missing original control must not invalidate complete normalized
primary contrasts or their interaction. A missing original_legacy outcome
invalidates the point for normalization_legacy, but does not invalidate
normalization_a171_standard. A missing normalized mixed-cell outcome does not
invalidate either normalization comparison. Every unknown remains explicit in
the full 48-row cohort and its overall/condition summaries.

## Fixed reporting and interpretation

Report `overall` for all 48 planned cells and `by_condition[condition]` for all
six groups, each with eight planned cells. Retain all strict categories and
format/EOS/cap/fence indicator counts. Incomplete-group strict accuracy is null,
with lower bound known successes divided by planned size and upper bound adding
unknown outcomes to the numerator. Include planned/resolved denominators.

Use `contrasts.strict_accuracy` with exactly `selection_block` and
`format_request`; use `secondary_contrasts.strict_accuracy` with exactly
`selection_by_format_interaction`, `normalization_legacy`, and
`normalization_a171_standard`. No additional conditional estimands, confidence
intervals, significance tests, detector-derived contrasts or population claims
are prespecified. The six-cell table exposes joint and control patterns.

If normalization changes the original diagonal pattern, report that change
explicitly; do not reinterpret the normalized factorial as an unchanged
component decomposition of the original instructions. If normalization deltas
are zero, they establish only equality of these diagonal strict outcomes; a
floor or ceiling can mask behavioral differences. Even a complete normalized
factorial effect identifies a text-package difference on these fixed worlds,
not isolated word semantics, pure selection ability, length control or an
internal mechanism. Fences and caps remain observed output features.

All outcomes, including all-floor, all-ceiling, null or reversed patterns, end
the fixed study. Stop after 48 attempts, the deadline, interruption or terminal
failure. Keep recoverable forward failures as infrastructure and continue only
the remaining frozen cells; never retry an attempt. Preserve unattempted slots
as missing. Do not edit prompts, add worlds, extend generation or detector scope,
resume a consumed run, reopen A169 or read earlier sealed held-outs.

## Runtime and immutable evidence

Use original local weights with the corrected pinned CPU FP32 reference,
CPU SDPA, four CPU threads, unchanged seed, batch size one, greedy decoding,
one beam, cache enabled and at most 64 new tokens. Preserve explicit frozen
EOS/padding/BOS settings and `logits_to_keep=1`. One fresh process loads the
target once. No target warmup, sampling, constrained decoding, likelihood
measurement, GPU, API, download, rental or new spending is authorized.

Bind original configuration SHA-256
`1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745`,
corrected reference-script SHA-256
`f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`,
and A174 public helper source SHA-256
`3de139bdd52be83e850cb9f942ef1f728d7efb96aabd65c58a9903dc5cc7d13b`.
The historical configuration's NF4 description does not change the effective
original-weight CPU FP32 loader. Preserve the pinned interpreter path without
resolving it:
`/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python`.

Freeze reviewed protocol, source, tests, plan, public dependency hashes and all
48 native constructions before any target call. Check exact system/user roles,
chat rendering and generation boundary, assistant-probe round-trip and EOS,
token vocabulary bounds, and prompt length plus all 64 generation tokens against
the bound model context limit. Failed input qualification does not authorize
stimulus changes. No private parent plan, record or response is an A176 input.

Respect shared jobs and require at least 48 GiB available RAM immediately
before loading. Use the established reservation and process lock. Enforce a
3,600-second internal wall deadline including preflight/loading, an external
3,600-second limit and 60-second kill grace. SIGTERM, SIGINT, SIGHUP and SIGALRM
are terminal, bypassing recoverable evaluation handling; preserve actual signal
number/name and PID where known. A direct KeyboardInterrupt has null signal
number/name. Generic BaseException is not a recoverable evaluation error.

Use a fresh absolute private directory under
`/data2/PRAX/lexical-prompt-study-data/runs/a176/`, private permissions, exclusive
locks, immutable startup/one-shot claims and process-local consumption guard.
Write each attempt before generation and each result durably afterward. Preserve
zero-call pre-header failure receipts and every planned slot. Never retry or
resume a consumed run under another directory.

Export under lock without loading the target model. Independently reconstruct
all inputs/schedule positions, native receipts, token termination, strict scores,
fence flags, counts and all five signed contrasts/bounds. Bind source/input/model
hashes and archive separately. Root owns review, code-only Git backup/push/freeze,
execution, independent verification and archive. Private text, token arrays,
row-level records and per-item identities remain outside Git; publish only
reviewed aggregates and whole-artifact provenance. CLI output is safe status
and coverage only.

## Public synthetic qualification

Before freezing, qualify all seven templates, two exact original controls,
selector multiplicity, fresh worlds, 48 unique cells, six groups, all 66 pairwise
schedule-order balances and world-level identical user messages. Exercise native
rendering/EOS/context/vocabulary guards and inherited strict categories,
termination/cap precedence and fence behavior. Independently calculate the two
normalized-only main effects, secondary interaction extrema +/-2, both
normalization effects and exhaustive signed missingness bounds. Explicitly test
that controls never affect the primary weights and missing controls cannot null
complete normalized results, while every affected contrast retains its planned
denominator. Test no gate/retry/resume, startup/runtime signals and deadlines,
durable publication, locks, tampering, replay and privacy using public synthetic
fixtures only. No target tokenizer, pretrained model call or private parent raw
evidence is used in qualification.
