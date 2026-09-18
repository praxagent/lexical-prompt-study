# A173: fresh clean-mapping factorial

This is a new prospective 64-generation experiment. A171 finished with four
strict successes among 96 outputs and zero among its 32 clean controls. Its
separately frozen post hoc A172 audit found a literal fence marker in all 92
capped outputs, no complete exact answer-object substring in those outputs,
and no capped output meeting the fixed repetition rule. These findings describe
the saved outputs; they do not establish why generation capped or identify an
internal mechanism. No A173 outcome is asserted here.

Public constructors show that A171 changed system wording, the user-message
heading and the answer-value form relative to the earlier task. A173 asks which
of those fixed input changes affects strict correctness on fresh clean mappings.
It crosses the three components without prose, a clarification reminder,
worked examples, a baseline gate or outcome-dependent revisions. It does not
resume, repair or alter any earlier study.

## Exact public worlds and input factors

Use these four underlying worlds, in the displayed order:

| World | Short value under A | Short value under B | Presentation order |
|---|---|---|---|
| w00 | s20 | s27 | AB |
| w01 | s24 | s21 | BA |
| w02 | s23 | s26 | BA |
| w03 | s25 | s22 | AB |

All eight short symbols are distinct and outside the historical public
constructor's `s00`--`s15` vocabulary. Their study-prefixed counterparts are also
new. No historical sealed world, original A169 context, prior private prompt,
response, score or native freeze is read to choose or prepare these worlds.
There is no random world search, alternative vocabulary or sample expansion.

For each world, cross all three binary factors and both selectors `A`, `B`:

- `system_wording`: `legacy`, then `a171_standard`.
- `mapping_heading`: `off`, then `on`.
- `value_form`: `short_symbol`, then `study_prefixed`.
- `selector`: `A`, then `B`.

Under `short_symbol`, use the exact table values. Under `study_prefixed`, prepend
the literal ASCII string `a173_` to each table value, for example `a173_s20`.
Keep the same underlying A/B assignment and presentation order in both forms.
The correct answer is the active selector's resulting value. Every value differs
from both selector labels and from its counterpart in the same mapping.

For `legacy`, use the exact public `tasks.system_instruction(selector)` wording,
shown here with its single substitution slot:

```text
The authoritative query selector is {selector}. The user supplies two explicit answer values labeled A and B. Copy the value with the authoritative label. Output exactly one JSON object with the single field "answer" containing that value as a string. Do not transform the value or include other fields, explanations, or Markdown fences.
```

For `a171_standard`, use the exact A171 `STANDARD` wording, without its reminder:

```text
Select the entry labeled {selector} from the user's mapping. Return that entry's mapped value as exactly one JSON object with the single key "answer" and a string value. Do not return any other text.
```

Replace `{selector}` with the active uppercase label, preserving the remaining
ASCII bytes. The system message is the sole source of the requested selector.

Render the two-entry user mapping as compact JSON in the world's AB or BA
presentation order, using double quotes and no spaces. For heading `off`, that
JSON object is the complete user message. For heading `on`, prepend exactly
`Mapping:\n`. Add no trailing newline, prose, delimiter, request sentence or
other text. Within each world/heading/value-form combination, both selector and
system-wording counterparts have byte-identical user messages.

Do not trim, normalize, pad, match native lengths, alter punctuation or select
values from tokenization or model outcomes. Record native lengths without
treating them as a criterion for revising the inputs. This is an input-factor
comparison, not isolation of length, semantic meaning or an individual phrase.

## Fixed 64-call schedule

Define the 16-element `base` as the ordinary Cartesian-product order of
`system_wording`, `mapping_heading`, `value_form`, `selector`, using the level
orders above; the selector varies fastest. Schedule the worlds as follows:

```python
base = list(product(
    ('legacy', 'a171_standard'),
    ('off', 'on'),
    ('short_symbol', 'study_prefixed'),
    ('A', 'B'),
))
rotated = base[8:] + base[:8]
world_schedules = (base, list(reversed(base)), rotated, list(reversed(rotated)))
```

Concatenate those four world schedules in w00--w03 order. This gives exactly 64
unique cells and eight observations per complete factorial setting. Each pair
of condition quadruples occurs in each relative order in two of the four
worlds. AB/BA presentation is balanced overall and within forward/reverse
scheduling. Each selector appears in every condition. Freeze the full schedule
before execution. Run every cell while the fixed resource and time bounds
permit; there is no clean-control gate or outcome-dependent stop.

## Original strict scoring and one fence indicator

Reuse the pinned A171 native preparation, original-reference greedy generation
and strict scorer without changing their behavior. The only new scoring field
is `fence_marker`, true exactly when the saved decoded response text contains
the literal ASCII substring of three consecutive backticks or three consecutive
tildes anywhere. Longer runs, inline occurrences and quoted occurrences count.
Do not require paired fences or interpret a marker as intended code. This is
the unchanged marker definition from A172, now prespecified for A173.

Strict correctness requires a complete JSON object containing exactly the key
`answer` and a string equal to the selected mapped value, with valid terminal
EOS. Permit surrounding JSON whitespace; reject duplicate or additional keys,
invalid types, other JSON shapes, trailing non-whitespace text and fences.
Terminal EOS must belong to the frozen generation EOS list, with no earlier
member of that list in the generated sequence. Decode after removing only the
final EOS, when present. EOS on the 64th permitted token is valid termination;
64 tokens without terminal EOS is a cap even if the text looks correct.

Preserve the A171 mutually exclusive categories: `correct`, `selected_label`,
`other_label`, `other_mapped_value`, `other_or_format`, `cap`, `infrastructure`
and `missing`. Cap precedence and invalid-termination/format precedence remain
unchanged. `selected_label` and other answer-string categories require valid
format and termination. Infrastructure and missing outcomes have null
behavioral indicators; completed behavioral failures have strict correctness
zero. Retain `format_valid`, `eos_valid`, `capped`, `attempted` and the new
`fence_marker`, with null where the outcome is unavailable.

Do not compute answer-path likelihoods, embedded answer-object counts, repeated
grams, substring rescue scores or additional detectors in A173. A fence marker
is an observable output feature; it does not establish a failure mechanism.

## Exactly three primary contrasts

Let `Y` denote strict correctness for a planned cell. For each factor, match
the two levels while holding world, selector and both other input factors
fixed. There are exactly 32 pairs per factor. Average the paired differences
equally over the other factors and both selectors within each world, then
equally over the four worlds. Equivalently, each matched difference has weight
one thirty-second.

The three primary contrasts are:

1. `system_wording`: `a171_standard` minus `legacy`.
2. `mapping_heading`: `on` minus `off`.
3. `value_form`: `study_prefixed` minus `short_symbol`.

Each includes all 64 planned outcomes and 32 planned pairs. For a fully resolved
contrast, report the exact point difference. If any required outcome is
unresolved, the point is null; retain every planned cell and report sharp lower
and upper bounds by independently assigning each unresolved binary outcome zero
or one with its signed coefficient. Do not zero-impute infrastructure/missing
outcomes, discard pairs, use a complete-case estimate or renormalize available
outcomes. Report planned and resolved outcome and pair counts explicitly.

Report overall counts and the eight `system_wording` by `mapping_heading` by
`value_form` cells, each with eight planned observations. Retain all original
strict categories and indicator counts, including the fence indicator, plus
strict accuracy with its resolved denominator and missing-outcome bounds. An
incomplete group has a null strict-accuracy point, lower bound equal to known
successes divided by its planned denominator, and upper bound adding all its
unresolved outcomes to the numerator. A completed cap is a known strict failure.

Use aggregate keys `overall`, `by_condition[system_wording][mapping_heading]
[value_form]`, and `contrasts.strict_accuracy` with exactly the three factor
names above. Indicators add `fence_marker` beside format, EOS and cap coverage.
Do not add interaction estimates, conditional contrast families, confidence
intervals, significance tests or additional primary endpoints. The eight-cell
table is retained to expose heterogeneous patterns without an interaction claim.

## Interpretation and stopping rules

The factorial contrasts describe these four fixed synthetic worlds and this
runtime. They do not establish population generalization or an internal causal
mechanism. System wording bundles multiple differences, including explicit
copying and formatting instructions. Adding the heading or value prefix changes
text, native tokenization and input geometry together. A system-wording effect
does not isolate the Markdown prohibition or any other one phrase.

All-floor, all-ceiling, mixed and null outcomes are possible. None authorizes
extra worlds, another wording, a changed cap, a new detector or follow-up within
this protocol. Preserve every observation and missing slot; do not reopen A169
or historical held-outs. Stop after the fixed 64 attempts, the deadline or an
interrupt. A recoverable evaluation failure remains infrastructure with no
retry, while the remaining fixed schedule proceeds. A terminal failure or
interrupt preserves the remaining cells as missing.

## Runtime and immutable evidence

Use the original local weights with the corrected pinned CPU FP32 reference,
CPU SDPA, four CPU threads, the unchanged pinned seed, fresh batch size one,
greedy decoding, one beam, cache enabled and at most 64 new tokens. Keep the
explicit frozen EOS/padding/BOS settings and generation `logits_to_keep=1`.
There is one fresh process and one target-model load, no target warmup,
sampling, constrained generation, likelihood measurement, external API,
download, rental, GPU computation or new spending.

Bind the original configuration SHA-256
`1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745`
and corrected reference-script SHA-256
`f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`.
The historical configuration describes an NF4 instrument; the effective loader
must remain the corrected original-weight CPU FP32 reference. Preserve the
interpreter path without resolving it:
`/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python`.

Freeze the new protocol, source, tests, public helper dependencies, model/config
hashes, plan and all 64 native constructions before a target-model call. Bind
the A171 source reused as implementation, not its private plan, run or outcomes.
Audit native system/user roles, chat rendering, generation boundary, exact
assistant-probe round-trip and EOS, all token vocabulary bounds, and prompt
length plus the full 64-token generation allowance against the bound model
context limit. A failed construction is not repaired by changing the stimulus.

Check shared resource headroom without disturbing another job. Require at least
48 GiB available RAM immediately before loading, using the existing reservation
and process lock. Use a 7,200-second internal wall deadline including preflight
and loading, an external 7,200-second limit and a 60-second kill grace. Signals
SIGTERM, SIGINT, SIGHUP and SIGALRM are terminal and bypass recoverable evaluation
handling. Preserve actual signal number/name and receiving PID when known; a
plain `KeyboardInterrupt` without a signal receipt has a null signal number.
A generic `BaseException` is not recoverable.

Use a fresh absolute private directory under
`/data2/PRAX/lexical-prompt-study-data/runs/a173/`, private permissions, exclusive
locks, an immutable startup/one-shot claim and process-local consumption guard.
Write each attempt before generation and each result durably afterward. Preserve
pre-header startup failures with a zero-model-call receipt and actual terminal
provenance. A consumed or interrupted run is never resumed or retried under a
new directory. Retain all 64 planned slots and distinguish attempted, completed,
infrastructure-failed, interrupted and unattempted counts.

Export under a lock without loading the target model. Independently rebuild
every task, schedule position, native receipt, generated-token termination,
strict score, fence indicator, count and signed contrast/bound. Verify frozen
source/input/model hashes and archive evidence separately. Root freezes and
backs up reviewed code through the established code-only Git worktree before
execution. Raw strings, tokens, row-level records and per-item identities remain
private outside Git. Release only reviewed aggregates and whole-artifact
provenance; CLI output is safe status and coverage only.

## Model-free qualification

Before freezing, qualify exact world values and all input bytes; all 64 unique
cells and factorial balances; the 16-condition product, rotations and reversals;
every pairwise schedule-order balance; selector/system user-message invariance;
native preparation and context/vocabulary bounds; unchanged strict categories,
EOS/cap precedence and exact fence markers; all eight aggregate cells; the three
signed 32-pair contrasts and independently calculated missing-outcome bounds.
Use wholly synthetic fixtures to test no gate, no retry, one-shot/locks, startup
and runtime interruptions/deadlines, durable receipts, tamper rejection,
independent replay and stdout privacy. No raw prior response or pretrained target
call is used in qualification.
