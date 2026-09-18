# A178: explicit-reference scaffold bridge

This is a prospective, fixed 56-generation return to the scaffold-specificity
question. A177 verified 0/8, 5/8 and 8/8 strict successes along its original,
demonstrative-of and explicit-label instruction path. Its explicit endpoint
also achieved 7/8 in A176. These fixed-input results support testing that exact
instruction with the original scaffold contrasts; they do not establish a
generally reliable instruction or isolate a linguistic mechanism. Concurrent
clean controls remain part of A178, with no outcome-dependent gate.

A178 compares full and sham scaffold packages at both placements using fresh
worlds. An inert-versus-clean comparison describes adding the fixed inert
material. It is not an isolated length, semantic-content or placement effect.
The materials have unequal native lengths; record lengths without padding,
matching, selection or revision. No new generation outcome is asserted here.

## Exact public worlds and instruction

Use these four worlds in the displayed order:

| World | Value under A | Value under B | Presentation order |
|---|---|---|---|
| w00 | s84 | s91 | AB |
| w01 | s88 | s85 | BA |
| w02 | s87 | s90 | BA |
| w03 | s89 | s86 | AB |

The eight distinct public values differ from A/B and prior public study values.
They are fixed without reading earlier sealed worlds, A169 contexts, private
parent world assignments, responses, token arrays or row-level records. Do not
search for errors or choose alternative values from tokenization or outcomes.

Use exactly these three single-line ASCII templates, in order:

```text
Select the entry labeled {selector} from the user's mapping.
```

```text
Return the mapped value of the entry labeled {selector} as exactly one JSON object with the single key "answer" and a string value.
```

```text
Do not transform the value or include other fields, explanations, or Markdown fences.
```

Substitute the active uppercase selector in both slots and join the three
sentences with one ASCII space at each boundary, without leading/trailing
whitespace. For both selectors the system message must equal A177
`system_instruction('of_explicit_label', selector)` byte for byte. No additional
instruction, reminder, demonstration or target-specific decoding constraint is
added. This instruction is fixed in all seven conditions.

## Restricted materials and exact user construction

The three existing scaffold materials are restricted frozen UTF-8 strings.
Do not reproduce their contents in public code, tests, protocol or console
output. Their source artifact SHA-256 is
`ee899a47ceca1b53e90787c5279174412e8d01e5872db9bba44a486921eba345`.
Root freezes the exact 25,055-byte source artifact privately at
`frozen/materials/materials-source.private.json` within the run package. It is
part of the frozen-file inventory, preserving self-contained evidence without
another runtime CLI argument or changing the seven-input inventory. The
independent native audit hashes this source, independently assembles each
four-block material with its original joiner, and compares the result against
the plan strings and material receipts. This restricted artifact never enters
Git or public qualification fixtures.

| Material key | UTF-8 bytes | SHA-256 |
|---|---:|---|
| full | 901 | 150cfb4e3b6fffab221543a8434a3b093859f28f44ee11510c169cafeb6822f6 |
| sham | 1019 | 3f3819e8b468a35cc62e60e19ba2a4fbd6d30432479d2e6f645ce7673e355cc9 |
| inert | 1275 | 402a805911f0ad6e6708ca7447bc47c2149183433cbde15df62310cbacb021ed |

Root supplies exactly these three materials to
`compile_plan(config_sha256, protocol_sha256, tests_sha256, materials)`.
Validate exact keys, string types, UTF-8 byte lengths and hashes. Bind the
source-artifact hash and canonical materials-object hash in the plan. The plan
stores the material strings only in its private artifact. Plan validation must
reconstruct the complete schedule/messages from those bound strings. Do not
trim, normalize, rename, replace or otherwise edit material bytes. This reuse
does not authorize reconstructing or rerunning any original A169 context or
sealed held-out world. Every assembled A178 prompt uses the fresh worlds above.

Render each mapping as compact JSON, with double quotes and
`separators=(',', ':')`, inserting keys in that world's presentation order.
There is no heading or prefix. The exact seven conditions are:

| condition | scaffold_kind | placement | user message |
|---|---|---|---|
| clean | none | none | mapping |
| full_before | full | before | material + two newline characters + mapping |
| full_after | full | after | mapping + two newline characters + material |
| sham_before | sham | before | material + two newline characters + mapping |
| sham_after | sham | after | mapping + two newline characters + material |
| inert_before | inert | before | material + two newline characters + mapping |
| inert_after | inert | after | mapping + two newline characters + material |

The delimiter is exactly `\n\n`; no additional leading/trailing characters
are added. Preserve any characters already contained in each material. Both
selectors share the same user message within a world/condition. The selected
and other mapped values define the unchanged strict oracle.

## Fixed schedule and fields

Let conditions be in the table order and selector vary fastest:

```python
base = list(product(conditions, ('A', 'B')))
rotated = base[7:] + base[:7]
world_schedules = (base, list(reversed(base)), rotated, list(reversed(rotated)))
```

Concatenate world schedules in w00--w03 order: exactly 56 unique cells, eight
per condition. Each of the 91 pairs among the fourteen condition/selector
combinations occurs in each relative order twice. Both selectors occur in
every world/condition. AB/BA presentation is balanced overall and within
forward/reverse scheduling. Execute the entire frozen matrix, including clean
and every context cell, regardless of earlier outcomes.

Trial fields are `trial_id`, `sequence_index`, `world_index`, `world_id`,
`presentation_order`, `condition`, `scaffold_kind`, `placement`, `selector`,
`selected_answer`, `unselected_answer` and `messages`. There is no instruction
factor or additional scoring feature.

## Unchanged strict scoring

Reuse pinned A177 native preparation, original-reference generation and strict
scoring. Preserve categories `correct`, `selected_label`, `other_label`,
`other_mapped_value`, `other_or_format`, `cap`, `infrastructure` and `missing`.
Strict success requires valid terminal EOS and a complete JSON object with
exactly the key `answer` and the selected mapped value as a string. Allow
surrounding JSON whitespace; reject duplicate/additional keys, wrong shapes or
types, trailing non-whitespace and fenced JSON.

A valid EOS is a final frozen EOS token with no earlier EOS; remove only the
final EOS for decoding. EOS at token 64 is valid if no earlier EOS occurs.
Sixty-four tokens without terminal EOS is a cap, even if the text resembles a
correct answer. A final EOS plus an earlier EOS is invalid termination, not a
cap, including at token 64. Preserve cap and invalid-termination/format
precedence. Label/wrong-value categories require valid format and EOS.

Retain strict correctness, selected-label, format/EOS/cap, attempted and existing
`fence_marker` indicators. Unknown behavioral indicators remain null. The fence
indicator means any literal ASCII triple-backtick or triple-tilde substring;
it does not require paired fences or establish output intent. No new detector,
embedded-object scoring, repetition rule, score rescue, likelihood or human
judgment is added.

## Primary full-minus-sham contrast

`contrasts.strict_accuracy` contains exactly `full_minus_sham`. Match full and
sham within world, selector and placement, averaging equally over four worlds,
two selectors and both placements. There are 16 pairs and 32 unique required
outcomes. Each full outcome has coefficient +1/16 and each sham outcome -1/16.
Clean and inert outcomes never enter this contrast.

## Five secondary contrasts

`secondary_contrasts.strict_accuracy` contains exactly:

- `inert_minus_clean`: mean of `(inert_before + inert_after)/2 - clean`
  over the eight world/selector triples. The 16 inert outcomes each have
  coefficient +1/16; the eight shared clean outcomes each have coefficient
  -1/8. There are 24 unique required outcomes and eight triples.
- `full_minus_sham_before` and `full_minus_sham_after`: each placement's
  full-minus-sham mean over eight matched world/selector pairs, sixteen unique
  required outcomes, with coefficients +1/8 and -1/8.
- `inert_minus_clean_before` and `inert_minus_clean_after`: each placement's
  inert-minus-clean mean over eight matched world/selector pairs, sixteen
  unique required outcomes, with coefficients +1/8 and -1/8.

The shared clean output is one observation per world/selector. Combine its
coefficient before bounding unknown outcomes; never duplicate it into
independent observations or inflate the unique-outcome denominator. There are
no confidence intervals, significance tests, interactions or further contrasts.

For each contrast report `point`, `lower`, `upper`, `planned_outcomes` and
`resolved_outcomes`. Paired contrasts add `planned_pairs` and `resolved_pairs`;
the overall inert comparison instead adds `planned_triples` and
`resolved_triples`. A pair/triple is resolved only when all required distinct
outcomes are known. Each completed cap is a known strict failure.

When all required outcomes are known, point and bounds equal the exact signed
mean. Otherwise point is null, with sharp bounds assigning each unknown binary
outcome zero or one according to its combined signed coefficient. Keep the
fixed planned denominator; do not drop outcomes, impute failures or renormalize.
All contrasts range from -1 to +1 and all-unknown bounds are [-1, 1].

Missingness is local to each required cohort. Missing clean/inert cells do not
null any full-minus-sham point. Missing full/sham cells do not null inert
comparisons. A missing clean outcome affects the overall inert contrast and
both placement-specific inert contrasts. Missing only inert_before leaves
inert_minus_clean_after resolved when its own required outcomes are complete.
Every missing slot remains explicit in the overall cohort.

## Reporting, interpretation and stop rules

Report `overall` for all 56 planned cells and `by_condition[condition]` for all
seven groups of eight, preserving every strict category and format/EOS/cap/fence
indicator count. Incomplete-group strict accuracy is null, with lower bound
known successes/planned size and upper bound adding unknowns. Report coverage
and planned/resolved denominators. Bounds describe missingness, not statistical
precision. Keep the placement-specific table even when contrasts are zero.

Full-minus-sham describes the difference between these exact packages in this
instruction/runtime/input construction. It does not isolate semantic content,
length, geometry, an internal causal mechanism or a population effect.
Inert-minus-clean includes added content, length and placement. Clean success
supports task competence on these inputs; it does not guarantee interpretable
context outcomes. Shared floor or ceiling values and zero contrasts do not
establish equivalence or absence of a scaffold effect. Format failures do not
establish wrong selection or absent answer content.

All outcomes, including context collapse, end this bounded bridge. Stop after
56 attempts, deadline, interruption or terminal failure. Recoverable forward
failures remain infrastructure; continue only the remaining frozen cells without
retrying. Preserve unattempted slots as missing. No outcome authorizes another
wording-repair loop, expanded sample, length matching, detector expansion,
generation extension, resumed run, original held-out reuse or A169 reopening.
Target patching and likelihood readouts require separate qualification and are
not part of this study.

## Runtime and immutable evidence

Use original local weights with pinned corrected CPU FP32 reference, CPU SDPA,
four CPU threads, unchanged seed, batch size one, greedy decoding, one beam,
cache enabled and at most 64 new tokens. Preserve explicit frozen EOS,
padding/BOS settings and `logits_to_keep=1`. One fresh process loads the target
once. No target warmup, sampling, constrained decoding, GPU, API, download,
rental or new spending is authorized.

Bind original configuration SHA-256
`1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745`,
corrected reference-script SHA-256
`f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`,
and A177 helper source SHA-256
`7a10f0e4c361f35b0c56b92c6921bd89ef7cec2f7b8deefdb99912dacaeb394f`.
The historical config's NF4 description does not change the effective original
CPU FP32 loader. Preserve the interpreter path without resolving it:
`/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python`.

Freeze reviewed protocol/source/tests, dependency and material receipts, plan
and all 56 native constructions before any target call. Check roles, exact
rendering/generation boundary, assistant-probe round-trip/EOS, native vocabulary
bounds and prompt length plus all 64 tokens against the bound context limit.
Failed qualification does not authorize changing stimuli. Native lengths are
recorded only; equal prompt lengths are not promised.

Respect shared jobs and require at least 48 GiB available RAM immediately before
loading. Use the established reservation and exclusive process lock. Enforce
7,200-second internal and external wall deadlines, including preflight/loading,
plus 60-second kill grace. SIGTERM, SIGINT, SIGHUP and SIGALRM are terminal and
bypass recoverable handling; preserve actual signal number/name and PID.
Direct KeyboardInterrupt has null signal number/name. Generic BaseException
is not a recoverable evaluation failure.

Use a fresh absolute private directory under
`/data2/PRAX/lexical-prompt-study-data/runs/a178/`, private permissions, locks,
immutable startup/one-shot claims and process-local consumption guard. Write
each attempt before generation and each result durably afterward. Preserve
zero-call pre-header failures and every planned slot. No consumed run can be
retried or resumed through another directory.

Export under lock without loading the target. Independently reconstruct all
messages/schedule/native receipts, strict scores, categories, EOS/fence flags,
counts and six contrasts/bounds. Bind source/input/model/material hashes and
archive separately. Root owns code-only Git backup, push/freeze, execution,
independent verification and archive. Restricted materials, prompts, responses,
token arrays, rows and per-item identities remain outside Git. Publish only
reviewed aggregates and whole-artifact provenance. CLI output is safe status
and coverage only.

## Public synthetic qualification

Tests use public stand-in strings and monkeypatched material receipts, never
read the restricted material source or parent raw evidence. Qualify exact
instruction inheritance, material keys/types/bytes/hashes, condition metadata,
delimiters, 56 unique cells, seven groups and all 91 balanced relative orders.
Exercise plan/material tamper rejection and native geometry/EOS/vocabulary
guards. Independently calculate all six contrasts, shared-clean coefficients,
triple/pair coverage and exhaustive sharp missingness bounds. Check that
excluded conditions cannot change a contrast, and required unknowns null only
affected points. Retain inherited strict categories/cap precedence, all planned
slots, no gate/retry/resume, startup/runtime signals and deadlines, durable
publication, locks, tamper detection, replay and privacy checks. Qualification
uses no target tokenizer, pretrained weights or target model call.
