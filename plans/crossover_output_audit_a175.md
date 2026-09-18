# A175: fixed post hoc structure audit of A174 outputs

This is a new, explicitly post hoc, zero-model-call audit. A174's independently
verified aggregate contains 32 completed outputs. Both legacy-core groups were
strictly correct (16/16 together). Standard core with legacy closing produced
eight uncapped, EOS-terminated, fenced format failures; standard core with its
standard closing produced eight capped, fenced failures. These aggregate results
motivate the audit, but no raw A174 response is inspected during setup. No A175
finding is asserted here.

A175 asks which unchanged A172 surface structures occur in all four groups,
particularly whether the eight terminated, fenced failures contain complete
answer-object substrings and which exact answer-string categories those objects
match. Reuse the three A172 detectors unchanged through a direct function alias
to its pinned public source, SHA-256
`4a5628bfd6bc835fb476e717d01197b4294c855ad7dfaf6798814f7245740c02`.
Freeze the A175 protocol, implementation, tests and all parent bindings before
any new raw-output read. This prospective freeze of an explicitly post hoc audit
does not create a new confirmatory endpoint or rescue score for A174. There is
no new generation, detector search, threshold change or response-intent judgment.

## Fixed cohort and evidence boundary

Audit all 32 planned, independently verified, completed outputs from A174
`run-001`. Do not choose outputs by content, outcome, world, instruction core or closing constraint.
The parent must have a successful bound independent verification, 32 attempted
and completed generations, and no infrastructure failures, interrupted attempts
or missing cells. Require the complete unique 32-result hash manifest from that
verification. A missing or inconsistent parent binding stops preparation or
execution; it does not authorize an observed subset, substituted output, parent
repair, model call or retry.

Bind the parent's frozen package, plan, native preparation, run header,
independent verification, aggregate, records and all 32 individual result SHA-256
values through the verified parent metadata. During setup, obtain the result
hashes from that metadata without opening or hashing raw result contents. Do not
read raw output text, token arrays or row-level records while choosing or
qualifying the detectors. Public synthetic fixtures alone qualify them.

The frozen execution has two phases. First, in the fixed parent schedule, write
an attempt receipt immediately before reading and hashing each raw result's
bytes. Verify all 32 result SHA-256 bindings before parsing or interpreting any
response text or token array. Only after the complete hash phase passes may the
audit parse and inspect the results in the same fixed order, writing a numeric
feature result after each successful inspection. An interruption can therefore
leave 32 attempted raw reads but fewer than 32 processed outputs; these counts
describe different stages and must remain separate.

Use the saved `score.response_text` and `generated_token_ids` whose decoding
and scoring were already independently verified for A174. No tokenizer or model
load is needed. Use the frozen A174 plan for its oracle values, active selector,
instruction core and closing constraint; use the original verified `capped` indicator for strata and
the run header's frozen EOS list for the sole token removal described below.
Reconcile the saved result with the bound parent record and original score; do
not infer a replacement cap, correctness or error category from A175 features.
The parent score already includes `fence_marker`; reconcile that saved field
without adding a new detector or changing the original score. The audit
feature retains the unchanged A172 name `fence`.

No original sealed held-out input or original A169 context is an A175 input.
Parent evidence is read-only. A174's original strict scores, categories,
termination and missingness stay unchanged.

## Exactly three detectors

### ASCII fence presence

Set `fence` to true exactly when the saved decoded response text contains the
literal ASCII substring of three consecutive backticks or three consecutive
tildes. Search anywhere in the text, with no stripping, case conversion,
normalization, Markdown parsing, opening/closing-pair requirement or line-boundary
rule. Longer runs count as presence. Inline or quoted occurrences also count.
This detector reports a marker, not the function or intent of a code block.

### Complete answer-object substrings

Visit every character offset whose literal character is `{`, in ascending
order. At each offset, independently call a strict `JSONDecoder.raw_decode` on
the original response text starting at that offset. Reject duplicate keys using
an object-pairs hook, and reject `NaN`, `Infinity` and `-Infinity` using a
parse-constant hook. A `ValueError`, `TypeError` or `RecursionError` rejects that
candidate offset only; continue to the next literal opening brace. Any other
unexpected failure remains an audit error.

A candidate counts exactly when decoding succeeds and the decoded value is an
object with exactly one key, `answer`, whose value is a string. JSON whitespace
inside that fragment is allowed. The decoder need not consume the surrounding
response or its trailing suffix. Do not require that the fragment be a complete
response, top-level object, unfenced object or intended answer. Reject extra
keys, duplicate keys, other JSON shapes and non-string answer values. Do not
repair truncation, escapes, unmatched braces or malformed JSON.

Count one object for each qualifying start offset, without deduplicating by
answer string, span or enclosing structure. Every literal opening brace remains
eligible independently: a valid inner answer object may count even if an outer
object has extra keys, duplicate keys, malformed content or a different schema.
Conversely, braces within an escaped JSON string count only if the actual
substring beginning there independently decodes under the same rule. These
are literal JSON-shaped substrings and may be examples or quoted material.

Assign every qualifying object to exactly one category by exact, case-sensitive
string equality, in this order:

1. `selected_mapped_value`: equals the active selector's mapped value.
2. `other_mapped_value`: equals the other supplied mapped value.
3. `selected_label`: equals the active selector label.
4. `other_label`: equals the other selector label.
5. `other_string`: any remaining string.

The parent task uses distinct values and labels, so these categories are
disjoint per object. Preserve the total `answer_object_count`, five
`answer_object_counts`, five binary `answer_object_presence` indicators and the
binary `any_answer_object`. Multiple categories may be present in one response;
their per-output presence counts must not be added as though mutually exclusive.
The total object count must equal the sum of the five category counts, and each
presence flag is true exactly when its corresponding count is positive.

### Repeated four-token gram

Start from the complete saved generated-token array. If and only if its final
token belongs to the frozen A174 EOS list, remove that one final token. Remove
no other tokens, including an earlier EOS token, whitespace token or marker.
Do not independently retokenize or normalize the response text.

Enumerate every contiguous four-token window, advancing one token at a time.
Set `repetition` to true exactly when any identical four-token tuple occurs at
least four times. Include overlapping windows; the matching windows need not
be adjacent or consecutive. Fewer than four retained tokens gives false. For
example, seven identical retained tokens contain four overlapping occurrences
of their four-token gram and pass; six identical retained tokens do not. This
is the only gram size and frequency threshold. It does not identify a decoder
loop, deliberate repetition or any internal mechanism.

## Numeric output and fixed summaries

The per-output detector result has exactly these fields:

- `fence`, `any_answer_object` and `repetition`: booleans.
- `answer_object_count`: nonnegative integer.
- `answer_object_counts`: the five-category nonnegative integer mapping above.
- `answer_object_presence`: the same five-category boolean mapping.
- `overlap`: a three-character string in `fence`, `any_answer_object`,
  `repetition` order, using `1` for present and `0` for absent.

Retain numeric per-output records privately, with bound parent identity and the
metadata needed to replay group membership. Do not retain matched strings,
answer fragments, character offsets, token grams or additional detector traces
in the new feature record. The original raw evidence remains in its existing
private parent files.

For each complete reported group, give `outputs`, `planned_outputs` and
`resolved_outputs` (equal for a complete group), counts of
outputs with each of the three features, total complete answer-object count,
the five object-category totals, the five per-output category-presence counts,
and all eight mutually exclusive overlap counts `000` through `111`, including
zeros. The overlap counts sum to the group denominator; their marginals must
reproduce the three feature-presence counts. Category object totals sum to the
total object count. Use counts with explicit denominators rather than adding
inferential statistics.

Report these fixed groupings:

- All 32 outputs together.
- The four `instruction_core` by `closing_constraint` groups, each containing eight outputs.
- Original `cap` and `noncap` strata, containing eight and 24 outputs respectively.
- The original cap/noncap strata within each of the four core/closing-constraint groups.

An empty subgroup has denominator zero and zero counts in every count field;
do not divide by zero or manufacture an accuracy rate. Group membership comes
from the bound parent metadata, not a detector or a newly inferred termination.
Use aggregate keys `overall`, `by_cap_status`, and
`by_condition[instruction_core][closing_constraint]`; each condition contains
`all` and `by_cap_status`. Both factor levels are `legacy`, `a171_standard`.
Preserve overlapping feature/category presences explicitly. Do not publish
per-world, per-item or response-level details.

An incomplete or failed audit has explicit coverage and failure provenance,
with the planned 32-output cohort retained and null feature records for
unprocessed outputs. All requested group summaries are null unless the entire
32-output audit is complete; coverage remains explicit. Do not publish
complete-case feature comparisons, impute absent features for an unprocessed
output or silently omit a required subgroup. No partially observed feature
result changes an A174 score.

## Interpretation and stopping boundaries

This audit can describe observable structures in already observed outputs. An
embedded selected-value object does not mean that the model supplied a valid
answer, that the strict scorer was wrong, or that the capped continuation would
eventually terminate correctly. A label-shaped fragment does not establish a
selector-copying mechanism. A fence marker or repeated gram does not establish
why generation failed. The 24 original noncap outputs include both strict successes and format failures;
they are descriptive strata, not an independently sampled control cohort.

There is no new accuracy measure, score rescue, response-intent judgment,
latent-mechanism attribution, significance test, confidence interval,
population claim, detector search or confirmatory reinterpretation of A174.
Do not add detectors, change thresholds, scan extra outputs, inspect additional
substring classes or tune the rules after seeing raw outputs or audit results.
No human or model judge is required.

Freeze this protocol, source, tests and all plan/parent bindings before the
first raw A174 result read for the audit. Use one fresh audit process, a private
absolute output directory under
`/data2/PRAX/lexical-prompt-study-data/runs/a175/`, an exclusive lock, an immutable
one-shot claim and a process-local consumption guard. Save a durable attempt
receipt before each raw parent result read and a numeric result after its
successful audit. Preserve actual interruption/deadline provenance and every
planned slot. A failed or interrupted consumed audit is not retried or resumed,
including under another output directory.

Use a 600-second internal wall deadline and an external 600-second limit with
a 60-second kill grace. This is local CPU/file work only: no model, tokenizer,
API, GPU, download or rental is used. Stop after the fixed 32 outputs, a binding
failure, unexpected error, deadline or interruption. Retain incomplete evidence
without repairing the parent or revising the frozen rules.

Export and independently replay the saved audit from the same bound parent
evidence without a model call. Recheck all parent result hashes before response
interpretation, then recompute detectors only for already completed audit
entries. Never compute missing feature results or finish an interrupted audit
through export. A replay verifies the fixed detectors and numeric summaries;
it is not permission to change the detectors or generation outputs. Verify
source/input hashes and archive the new evidence separately.
Publish reviewed aggregate counts and whole-artifact provenance only. CLI
output is safe execution/coverage status, never raw strings, token arrays,
matched fragments, per-item identifiers or their hashes.

## Public synthetic qualification before freezing

Qualify the exact ASCII fence substrings; strict object parsing; all five value
categories; duplicate and nonfinite rejection; nested, malformed, quoted and
multiple-object cases; candidate-offset independence; repetition frequency
edges, nonconsecutive recurrence, overlapping windows and final-EOS-only removal.
Independently verify all eight feature-overlap cells, their marginals, category
object totals versus per-output presence, all fixed group denominators and empty
cap strata. Test parent binding drift, exact 32-result coverage, one-shot and
lock behavior, terminal interruption, evidence tampering and stdout/privacy
controls using public synthetic evidence only. These tests inspect no raw A174
output and make no model or tokenizer calls.
