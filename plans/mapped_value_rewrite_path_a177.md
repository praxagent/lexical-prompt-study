# A177: mapped-value reference rewrite path

This is a new prospective 24-generation experiment. Verified A176 completed all
48 cells with valid EOS and no caps, infrastructure failures or missing outcomes.
Its original standard control had 0/8 strict successes; its normalized standard
diagonal had 7/8. The normalized legacy-selection groups each had 8/8 successes,
while the two normalized standard-selection groups had 6/8 and 7/8. The legacy
normalization contrast was zero at the strict-accuracy ceiling. All eleven
failures were fence-marked and format-invalid. These observations do not establish
whether those failures contain answer substrings or why the rewrite helped.

A177 asks which step along one fixed reference-phrase path changes strict
accuracy: original possessive reference, an of-construction retaining the same
demonstrative reference, or an of-construction naming the selector locally. The
original and explicit-label endpoints reproduce A176's original_standard and
normalized_standard_standard messages. The intermediate phrase has a clear
antecedent after the unchanged selection sentence. This is a path of exact text
packages, not an orthogonal factorial or an isolation of syntax, reference
resolution, repetition, length, tokenization or an internal mechanism.

## Exact public worlds and five templates

Use these four fresh public worlds in the displayed order:

| World | Value under A | Value under B | Presentation order |
|---|---|---|---|
| w00 | s72 | s79 | AB |
| w01 | s76 | s73 | BA |
| w02 | s75 | s78 | BA |
| w03 | s77 | s74 | AB |

The eight distinct values differ from A/B and earlier public study values.
Select no alternatives using tokenization, errors or outcomes. No historical
sealed worlds, original A169 contexts, private parent prompts, outputs, token
arrays or row-level records are inputs to this study.

Use exactly these five single-line ASCII templates in the displayed order.

Common standard selection sentence:

```text
Select the entry labeled {selector} from the user's mapping.
```

`original_possessive` format request (O):

```text
Return that entry's mapped value as exactly one JSON object with the single key "answer" and a string value.
```

`of_demonstrative` format request (B):

```text
Return the mapped value of that entry as exactly one JSON object with the single key "answer" and a string value.
```

`of_explicit_label` format request (E):

```text
Return the mapped value of the entry labeled {selector} as exactly one JSON object with the single key "answer" and a string value.
```

Common unchanged legacy closing sentence:

```text
Do not transform the value or include other fields, explanations, or Markdown fences.
```

Substitute the active uppercase selector into every slot. Join selection, format
request and closing with exactly one ASCII space at each boundary, with no
leading/trailing whitespace. O and B name the selector once; E names it twice.
The O message must equal A176
`system_instruction('original', 'a171_standard', 'a171_standard', selector)`;
the E message must equal A176
`system_instruction('normalized', 'a171_standard', 'a171_standard', selector)`
byte for byte for both selectors. B is the sole new instruction variant.

The common legacy closing previously allowed terminated standard-core failures;
its use does not guarantee termination or accuracy on these new inputs. Both
demonstrative references are linked to the entry explicitly selected in the
preceding sentence. No bridge, reminder, demonstration or other sentence is
added. E changes the reference phrase, repeats the selector and changes length
and tokenization. Neither path contrast isolates one of those properties.

Render the user mapping as compact JSON with double quotes and
`separators=(',', ':')`, inserting A/B keys in the world's presentation order.
The object is the entire user message: no heading, prefix, prose, trailing
newline or other request. All six cells within a world have identical user
messages. The selected and other mapped values define the unchanged strict
oracle. Record native lengths without padding, length matching or input revision.

## Fixed 24-call schedule

Condition order is exactly `original_possessive`, `of_demonstrative`,
`of_explicit_label`. Selector varies fastest:

```python
base = list(product(conditions, ('A', 'B')))
rotated = base[3:] + base[:3]
world_schedules = (base, list(reversed(base)), rotated, list(reversed(rotated)))
```

Concatenate schedules in w00--w03 order. There are 24 unique cells, eight per
condition, both selectors in every condition/world, and all 15 pairs among the
six condition/selector combinations occur in either relative order twice.
AB/BA presentation is balanced overall and within forward/reverse scheduling.
Freeze and execute the entire schedule without control gates or sample expansion.

Trial metadata contains only `trial_id`, `sequence_index`, `world_index`,
`world_id`, `presentation_order`, `condition`, `selector`, `selected_answer`,
`unselected_answer` and `messages`. A176's construction/selection/format factors
are not A177 axes or trial fields.

## Unchanged strict scoring and exactly three contrasts

Reuse the pinned A176 native preparation, original-reference greedy generation
and strict scorer, including its existing literal fence indicator. Preserve
categories `correct`, `selected_label`, `other_label`, `other_mapped_value`,
`other_or_format`, `cap`, `infrastructure` and `missing`.

Strict correctness requires a complete JSON object with exactly the key
`answer`, whose string value equals the selected mapped value, and valid EOS.
Allow surrounding JSON whitespace; reject duplicate/additional keys, other
shapes/types, trailing non-whitespace and fenced JSON. Valid EOS is a final
member of the frozen EOS list with no earlier member. Remove only that final
EOS for decoding. EOS at token 64 is valid if no earlier EOS occurs; 64 tokens
without terminal EOS is a cap, even if the text resembles a correct answer.
A final EOS together with an earlier EOS is invalid termination, not a cap,
including at token 64. Preserve cap and
invalid-termination/format precedence. Label/wrong-value categories require
valid format and EOS. Unknown behavioral indicators remain null.

Retain strict correctness, selected-label, format/EOS/cap, attempted and
`fence_marker` indicators. The latter means an ASCII triple-backtick or
triple-tilde substring anywhere in saved text, without pairing or intent
interpretation. No embedded-object detector, repetition detector, likelihood,
score rescue or new response-intent judgment is added.

Primary `contrasts.strict_accuracy` contains exactly:

- `of_rewrite`: B minus O.
- `explicit_label_rewrite`: E minus B.

Secondary `secondary_contrasts.strict_accuracy` contains exactly
`endpoint_total`: E minus O. Each contrast matches world and selector across
its two required conditions: eight equally weighted pairs and sixteen required
outcomes, with coefficients +1/8 and -1/8. Average equally over all four worlds
and both selectors. The endpoint comparison checks transfer of the A176
rewrite effect. No additional conditional, factorial or detector-based
contrasts, significance tests or confidence intervals are prespecified.

For each contrast report `point`, `lower`, `upper`, `planned_outcomes`,
`resolved_outcomes`, `planned_pairs` and `resolved_pairs`. Required outcomes
are Boolean strict success/failure or unknown. Completed caps are known
failures. When all sixteen required outcomes are known, the point and bounds
equal the exact signed paired mean. Otherwise point is null and sharp bounds
assign each unknown zero or one according to its signed coefficient. Keep
fixed planned denominators; never drop rows, impute failures or renormalize.
A pair is resolved only if both of its required outcomes are known.

Missingness is local to the two conditions in each contrast. Unknown O leaves
E-minus-B resolved when B and E are complete. Unknown E leaves B-minus-O
resolved when B and O are complete. Unknown B leaves E-minus-O resolved when
E and O are complete. Missing outcomes still remain explicit in the overall
24-cell cohort. Complete contrasts range from -1 to +1; all-unknown bounds
are [-1, 1]. The three point estimates obey endpoint_total = of_rewrite +
explicit_label_rewrite when all are resolved; separate missingness bounds
need not add because the shared B outcomes cancel only in the endpoint.

## Reporting and stop rules

Report `overall` for all 24 planned cells and `by_condition[condition]` for
three groups of eight. Retain every category and format/EOS/cap/fence indicator
count. Incomplete-group strict accuracy is null, with lower bound known
successes/planned size and upper bound adding unknown outcomes; include
planned/resolved denominators. Preserve all three contrasts and their local
coverage. Bounds describe unresolved outcomes, not sampling uncertainty.

Interpret changes as exact phrase-package differences on these fixed worlds
and runtime. If B resembles O and E improves, the second rewrite step carries
the improvement in this design; if B already improves, the first step matters.
Neither pattern identifies the underlying mechanism, isolated grammatical
operation, individual token or general population effect. Report reversed,
ceiling/floor and non-transferring endpoint patterns equally. A fence remains
an observed substring; invalid format does not establish incorrect selection
or absent answer content. The previous legacy ceiling does not justify a
normalization-neutrality claim.

All outcomes end the fixed study. Stop after 24 attempts, deadline,
interruption or terminal failure. Recoverable forward failures remain
infrastructure; continue only remaining frozen cells without retries.
Unattempted slots remain missing. Do not edit prompts, add worlds, extend
generation, expand detectors, resume a consumed run, reopen A169 or inspect
earlier sealed held-outs.

## Runtime and immutable evidence

Use original local weights and the corrected pinned CPU FP32 reference, CPU
SDPA, four CPU threads, unchanged seed, batch size one, greedy decoding, one
beam, cache enabled and at most 64 new tokens. Preserve explicit frozen EOS,
padding/BOS settings and `logits_to_keep=1`. One fresh process loads the target
once. No target warmup, sampling, constrained decoding, likelihood collection,
GPU, API, download, rental or new spending is authorized.

Bind original configuration SHA-256
`1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745`,
corrected reference-script SHA-256
`f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`,
and A176 public helper source SHA-256
`87bed6b20a0dda0b438c58c411b104c80f9b1efe63f65f9a04711733e9495e5b`.
The historical configuration's NF4 description does not change the effective
original-weight CPU FP32 loader. Preserve the pinned interpreter path without
resolving it:
`/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python`.

Freeze reviewed source, protocol, tests, plan, dependency hashes and all 24
native constructions before any target call. Check exact roles, chat rendering,
generation boundary, assistant-probe round-trip/EOS, native vocabulary bounds,
and prompt length plus 64 tokens against the bound context limit. Failed
qualification does not authorize stimulus changes. No parent private plan,
record or response is an A177 input.

Respect shared jobs and require at least 48 GiB available RAM immediately
before loading. Use the established reservation and exclusive process lock.
Enforce a 3,600-second internal wall deadline including preflight/loading,
external 3,600-second limit and 60-second kill grace. SIGTERM, SIGINT, SIGHUP
and SIGALRM are terminal and bypass recoverable evaluation handling. Preserve
actual signal number/name and PID where known; direct KeyboardInterrupt has
null signal number/name. Generic BaseException is not recoverable evaluation.

Use a fresh absolute private directory under
`/data2/PRAX/lexical-prompt-study-data/runs/a177/`, private permissions,
exclusive locks, immutable startup/one-shot claims and process consumption
guard. Write attempts before generation and results durably afterward. Preserve
zero-call pre-header failures and every planned slot. Never retry or resume
a consumed run under a different directory.

Export under lock without target loading. Independently reconstruct inputs,
schedule, native receipts, strict scores, EOS/fence indicators, counts and all
three paired contrasts/bounds. Bind source/input/model hashes and archive
separately. Root owns review, code-only Git backup/push/freeze, execution,
independent verification and archive. Private text, token arrays, rows and
per-item identities remain outside Git. Publish only reviewed aggregates and
whole-artifact provenance; CLI output is safe status and coverage only.

## Public synthetic qualification

Qualify the five literal templates, endpoint reconstruction, selector counts,
fresh worlds, 24 unique cells, three groups, all 15 pairwise order balances,
and identical user messages within worlds. Independently calculate all three
signed paired means, extrema and exhaustive missingness bounds. Test that
adversarial outcomes or missingness in the excluded third condition cannot
affect a contrast, that required unknowns null its point, and that fixed pair
denominators remain intact. Qualify inherited rendering/EOS/context/vocabulary
guards, strict categories, cap precedence and fence behavior. Exercise no
gate/retry/resume, signals/deadline, durable publication, locks, tamper
rejection, replay and privacy using synthetic fixtures only. No target
tokenizer, pretrained model call or private parent raw evidence is used.
