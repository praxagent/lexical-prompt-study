# A171: mapped-value clarification on fresh synthetic inputs

Status: prospective protocol. No A171 outcome is asserted here. This is one new,
bounded generation experiment, not a repair or continuation of A164--A170. In
particular, it does not read, retry, resume, recompute or replace original A169
contexts or any original sealed held-out cohort. A170's descriptive numerical
results do not supply A171 task outcomes or an accuracy qualification.

The question is whether one explicit selector-versus-value reminder changes
strict answer accuracy and selector-label returns when a new neutral prose
passage is present. The standard instruction already accurately requests the
mapped value. This is a comparison of two fixed wording treatments, not a test
of an intentionally ambiguous instruction or an identified internal mechanism.

## Exact public task construction

Use precisely the following eight worlds in numeric order. These values belong
to the new `a171_` namespace, differ from the labels `A` and `B`, and do not reuse
the historical `s00`--`s15` answer vocabulary. Do not open historical sealed worlds
to construct or select them. There is no sampled development cohort, held-out
cohort, difficulty search, value substitution or adaptive expansion.

| World | Value under A | Value under B | Mapping presentation order |
|---|---|---|---|
| w00 | a171_v47 | a171_v83 | AB |
| w01 | a171_v26 | a171_v91 | BA |
| w02 | a171_v68 | a171_v14 | BA |
| w03 | a171_v35 | a171_v72 | AB |
| w04 | a171_v89 | a171_v20 | AB |
| w05 | a171_v56 | a171_v03 | BA |
| w06 | a171_v11 | a171_v64 | BA |
| w07 | a171_v97 | a171_v42 | AB |

Each world is evaluated under selectors `A` and `B`, wordings `standard` and
`clarified`, and contexts `none`, `before` and `after`: exactly 12 generations per
world and 96 planned generations. Its oracle is the value under the active
selector. All 16 values are distinct. Presentation is fixed across every
condition of a world, with four AB and four BA worlds; both presentation orders
occur twice among forward-scheduled and twice among reverse-scheduled worlds.

The standard system message is exactly the following single line, replacing
`{selector}` with the active uppercase label and retaining the ASCII punctuation:

```text
Select the entry labeled {selector} from the user's mapping. Return that entry's mapped value as exactly one JSON object with the single key "answer" and a string value. Do not return any other text.
```

For `clarified`, append exactly one ASCII space and this fixed reminder to the
standard system message:

```text
The selector names a mapping entry; the answer is its mapped value, not the selector label.
```

The user task core is the literal `Mapping:\n` followed immediately by a compact
JSON object containing the two label/value entries in the world's presentation
order. Use ASCII double quotes, a comma between entries, colons between keys and
values, and no spaces within that JSON object. There is no trailing newline or
additional request sentence. Selector and wording counterparts have
byte-identical user messages.

Use exactly this new public neutral prose paragraph, on one line:

```text
Beyond the window, a narrow garden path curved around a patch of low shrubs. Rain from the previous evening remained on the broad leaves, while the gravel had begun to dry. A wooden bench stood beneath the nearest tree. Its surface was smooth where visitors usually sat, and a few fallen leaves rested at one end. Farther along the path, small flowers grew beside a shallow stone basin. Water moved gently when a breeze crossed the garden. The branches above made shifting patterns of light on the ground. Near the wall, a climbing plant reached toward a sheltered corner. The air felt cool in the shade and warmer beside the open lawn. From time to time a bird landed on the fence, paused, and flew toward the neighboring trees. The scene changed slowly as the sun rose. Shadows shortened, the leaves became less damp, and the quiet path remained open between the plants.
```

Construct the user message as `core` for `none`, `prose + '\n\n' + core` for
`before`, and `core + '\n\n' + prose` for `after`. Do not trim, pad, normalize,
replace punctuation, add delimiters, search for an effective passage, match an
earlier passage's length or tune any text after native tokenization. There is
one passage, selected before outcomes. Native prompt lengths are preparation
receipts, not a criterion for changing the stimuli. The wording intervention
and prose placement need not preserve native prompt length.

## Exact balanced execution schedule

Visit worlds w00 through w07 consecutively. Define this base list of
`(context, wording, selector)` triples:

```python
base = [
    ('none',   'standard',  'A'),
    ('before', 'clarified', 'B'),
    ('after',  'standard',  'A'),
    ('none',   'clarified', 'B'),
    ('before', 'standard',  'A'),
    ('after',  'clarified', 'B'),
    ('none',   'standard',  'B'),
    ('before', 'clarified', 'A'),
    ('after',  'standard',  'B'),
    ('none',   'clarified', 'A'),
    ('before', 'standard',  'B'),
    ('after',  'clarified', 'A'),
]
```

For numeric world index `w`, set `p = w // 2`, `k = 3 * p`, and form the left
rotation `base[k:] + base[:k]`. Use that rotated order for even `w` and its exact
reversal for odd `w`. Concatenate the eight 12-call world schedules. This fixes
all 96 attempts without runtime randomization. Each world pair uses mirrored
ordering, balancing the relative order of every pair of condition triples over
the eight worlds, including selector, wording and context comparisons. Clean
controls are interleaved with prose conditions; there is no initial baseline
phase or outcome-dependent baseline gate. Freeze the full schedule before any
target-model call and verify the schedule independently from these rules.

## Generation, termination and scoring

Generate once per planned cell from a fresh native chat input, batch size one,
using the pinned CPU reference's greedy settings and a maximum of 64 new tokens.
Use no demonstrations, constrained decoder, answer-path likelihoods, candidate
scoring, judge model or extra qualification generation. The exact supplied
mapping provides a deterministic oracle.

Before loading the target model, freeze every system/user message, native chat
rendering, prompt-token array, oracle, schedule position and generation settings
privately. Audit the pinned native chat template, generation-prompt boundary,
special-token rules, EOS configuration and no-truncation context fit for every
cell. Reject an inconsistent or overlong construction rather than changing the
stimulus. Save the exact generated token sequence and termination receipt for
each completed attempt, and decode and re-score them independently at export.

Strict format requires one complete valid JSON object with exactly the single
key `answer` and a string value. Permit JSON whitespace outside the object, but
reject duplicate keys, extra keys, trailing non-whitespace text, fences,
non-string answers and any other JSON shape. Valid termination requires a final
EOS token from the frozen generation EOS list, with no earlier member of that
list in the response; exclude only that final EOS from the decoded answer text.
A response without that terminal EOS cannot be strictly correct. Reaching the
64-token limit without terminal EOS is a cap, including when the preceding text
looks correct; EOS on the final permitted token counts as EOS termination.

Assign exactly one primary outcome category per planned cell, in this order:

1. `missing`: no evaluation attempt was made, or an attempt was interrupted
   before a complete, verified generation result could be recorded.
2. `infrastructure`: a recorded recoverable evaluation failure without a
   complete generation result. Retain a separate failure receipt and no retry.
3. `cap`: 64 new tokens were produced without terminal EOS.
4. `other_or_format`: termination is invalid or strict JSON format is invalid.
5. `correct`: the valid answer string equals the selected mapped value.
6. `selected_label`: the valid answer string equals the active selector label.
7. `other_label`: the valid answer string equals the other selector label.
8. `other_mapped_value`: the valid answer string equals the nonselected value.
9. `other_or_format`: any remaining valid answer string.

The repeated `other_or_format` branches yield one category, not separate output
categories. Retain orthogonal `format_valid`, `eos_valid`, `capped` and
`attempted` indicators; use null where an outcome cannot be determined. Also
retain a private reason distinguishing invalid format, invalid termination and
a valid but unrecognized answer. Do not rescue strict accuracy using embedded
JSON, partial outputs, copied substrings or a later post hoc category. A
selector-label category is an observable exact response pattern, not proof of
an internal copying or binding mechanism.

## Fixed paired summaries and missingness

Let `Y[w,s,c,t]` be the strict-correctness indicator for world `w`, selector `s`,
context `c` and wording `t`, and let `L[w,s,c,t]` indicate `selected_label`.
For each context separately, define the wording contrast by averaging
`Y[w,s,c,clarified] - Y[w,s,c,standard]` equally over both selectors within each
world, then equally over all eight worlds. Thus each context contrast contains
16 paired cells and every world has weight one eighth. Report analogous paired
contrasts for `L`, without conditioning on previously observed label errors.

The primary strict-accuracy contrast is the equal average of the `before` and
`after` wording contrasts, equivalently 32 paired prose cases. Report the two
placement-specific contrasts, the `none` wording contrast, and the prose-minus-
none interaction (the primary prose wording contrast minus the no-prose wording
contrast). Report each placement-minus-none interaction as a secondary
description. The analogous selected-label contrasts are prespecified diagnostic
summaries. Report the six context-by-wording accuracy and category counts, clean
control performance, format/EOS/cap coverage and attempted/completed/failed/
missing totals. Aggregate publication contains no per-world response details.

Do not drop a world or pair because one member failed, capped, was interrupted
or remained unattempted. A completed behavioral failure is strict zero;
infrastructure and missing outcomes are unresolved, not zero. For every binary
contrast, report a point only when all required planned outcomes are resolved;
otherwise set it to null and report sharp worst/best bounds by allowing each
unresolved binary outcome to range independently over zero and one, retaining
the specified weights and signs. The same rule applies to interactions and
selected-label contrasts. Report planned and resolved denominators explicitly.
Do not substitute complete-case contrasts or average available placements.

These are descriptive results for eight fixed synthetic worlds, one prose
passage, one model and one runtime. There is no bootstrap, significance test,
confidence interval, population generalization, practical-loss exclusion or
equivalence claim. Clean-control errors constrain claims of prose-specific
damage or rescue; they neither stop the schedule nor justify deleting worlds.
Ceiling, floor and zero contrasts remain possible and do not authorize a new
passage, reminder, sample or follow-up within this protocol. Wording changes
text, length and token geometry together, and adding prose changes multiple
input properties; no isolated semantic or internal-mechanism cause is claimed.

## Runtime, boundaries and durable evidence

Use original local weights and the pinned original CPU FP32 reference loader,
CPU SDPA, four CPU threads and the unchanged pinned seed. Bind the original
configuration and corrected reference script by their existing SHA-256 values:

- Configuration: `1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745`.
- Reference script: `f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`.

The retained configuration describes a historical NF4 instrument; the effective
loader must still be the corrected original-weight CPU FP32 reference. Preserve
the pinned interpreter path without resolving it:
`/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python`.
Validate source, configuration, tokenizer/chat-template and model-file hashes
before execution. Reuse public runtime helpers only, not historical private
prompt/response artifacts or parent-study freeze dependencies. No GPU, remote
model API, download, rental or new spending is part of A171.

Before substantial work and again before model load, check shared CPU load, RAM,
scratch space and GPU/process occupancy without disturbing another job. Require
at least 48 GiB available RAM immediately before model loading, use the existing
resource reservation and exclusive process lock, and release only A171-owned
resources. Waiting for capacity does not consume a model attempt.

The run has a 7,200-second internal wall deadline including preflight/loading and
an external 7,200-second limit with a 60-second kill grace. There is exactly one
fresh process, one target-model load and at most the frozen 96 generation calls.
No warmup model call, retry, resume, replacement output directory, baseline gate,
adaptive stopping for task performance or automatic wording revision is allowed.
Infrastructure failures receive no retry; signals and deadline preserve all
remaining slots as missing. Record signal number and receiving PID when known;
do not infer the sender or cause. SIGTERM, SIGINT, SIGHUP and SIGALRM must bypass
recoverable evaluation handling; a generic `BaseException` is not recoverable.

Compile a fresh A171 plan binding this protocol, current source/tests, original
config and pinned runtime dependencies. Freeze the 96 native inputs and exact
schedule before any target-model call. Preserve immutable one-shot claims,
process-local consumption guards, durable per-attempt and per-result receipts,
run completion/interruption receipt and all planned outcome slots. A consumed
or interrupted run is not restarted under a new path. Keep all raw and row-level
artifacts under an absolute private directory in
`/data2/PRAX/lexical-prompt-study-data/runs/a171/`, outside Git, with private
permissions. CLI output is safe status/coverage only: no prompts, responses,
token arrays, per-item identifiers or their hashes.

Export under the appropriate lock, without loading the model. Independently
rebuild task strings, the complete schedule, native token receipts, generated
sequence termination/scoring, the mutually exclusive categories, missingness and
every aggregate contrast/bound. Verify all source/config/model bindings and
archive the evidence with an independently checked manifest. Freeze, test and
back up reviewed code through the established code-only Git worktree before
execution. Release only separately reviewed aggregate counts/contrasts and
whole-artifact provenance; new raw evidence remains private.

## Model-free qualification and stop rules

Before target-model execution, test exact task/prose bytes; all eight worlds,
distinct values and 96 unique cells; all selector/presentation balances; exact
rotation/reversal order and mirrored comparisons; selector/wording user-message
invariance; native chat/EOS and context audits; strict parsing and category
precedence; EOS on the last permitted token; cap versus missing/infrastructure;
paired weights, interaction signs and unresolved-outcome bounds. Use synthetic
generation/receipt fixtures to test tamper rejection, one-shot/lock enforcement,
deadline/signal interruption, no retry and independent replay. These tests make
no target-model calls or pretrained-weight qualification generations.

Stop preparation for a failed construction, binding, resource or implementation
guard. Correct code/protocol defects before freezing rather than selecting
stimuli from model outcomes. After the one-shot run begins, preserve failures
and follow the unchanged schedule while resources and the deadline permit.
Stop at 96 calls, the deadline or an interrupt, then verify and archive whatever
coverage exists. No A171 result permits reopening A169 or original held-outs.
