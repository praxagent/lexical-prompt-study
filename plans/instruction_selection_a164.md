# A164: direct answer selection under lexical scaffolds

Status: prospective implementation; no A164 model calls or results. This is a new
instrument prompted solely by A163's failed baseline competence controls. It is
not another A163 wording repair. Preserve A163 sources, outputs, failed gates and
floor-limited contrasts; those contrasts do not establish absence of an effect.
The earlier A164 proposal remains a separate record.

## Fixed task and cohort

The system message alone specifies authoritative selector A or B. The user gives
two explicit distinct answer values, such as `{"A":"s03","B":"s11"}`. Return
exactly one JSON object with the single string field `answer` containing the
selected value. There is no table lookup, composition, arithmetic or transformation.
The task family is `direct_answer_selection`; it has no task-depth field.

The symbols are fixed as s00 through s15. A world is an ordered A/B value pair,
independent of active selector, presentation order and scaffold. There are 240
possible distinct-value pairs. Before any call, create and freeze the complete
80-world manifest using seed **2026091800**: five sequential shuffled derangements
with no repeated ordered pair, the first for 16 development worlds and the other
four for 64 held-out worlds. Within each cohort, shuffle world order and assign
alternating AB/BA presentation. Each symbol occurs once under each label in
development and four times under each label in held-out. Presentation counts are
8/8 and 32/32. Validate all identities, balances and cross-cohort disjointness.

This is restricted randomization over a small finite task population. Worlds
are the analysis units; selectors, placements and scaffolds are paired descendants.
The design does not make its worlds independent draws from general instruction
tasks. A world's presentation is fixed across its conditions. Selector A/B
counterparts must have byte-identical user messages.

## Development gate and fixed held-out matrix

Freeze one system wording, JSON renderer, scorer, manifest, protocol and runtime
before 32 development no-scaffold cells. Require at least 90% strict correctness
and at least 90% of worlds with both selector answers correct: 29/32 cells and
15/16 pairs. All baseline records must be completed. Report AB/BA strata without
selecting a presentation. A failed gate stops A164: no automatic wording repair,
worked examples, renderer search, model change or precision change.

Only after the runtime replays concrete development receipts and verifies the
gate may the held-out plan load the model and launch. It must share exactly the
same cohort, source, protocol, material and model-configuration bindings. The
held-out matrix is fixed at 18 cells per world: full/sham/replacement/inert ×
before/after × A/B, plus no-scaffold × A/B, totaling 1,152 generations. Reuse the
existing verified materials through the caller's private adapter; record exact
material hashes and token lengths. No scaffold results may select the instrument.

Require the same 90% held-out baseline gates: 116/128 strict cells and 58/64 pairs.
Missing baseline cells prevent passage. Never exclude a world because its task
answer is difficult, malformed or unfavorable. Failed held-out controls constrain
interpretation; they do not authorize a replacement cohort or altered prompt.

## Endpoints and uncertainty

Strict correctness requires one complete valid JSON object, exactly the allowed
answer field, and the keyed answer. Duplicate keys, extra fields, trailing text,
fences and invalid types are format failures. A cap termination is a strict
failure even if the text is otherwise correct. Retain separate exact, other-
selector, other-answer and format categories; format damage is not semantic
rebinding. Infrastructure failures remain unresolved, with all planned cells in
coverage reports and best/worst binary bounds. They may receive at most two
explicit unchanged-configuration retries; completed task failures are never retried.

Primary estimand: the mean of each world's full-minus-sham strict-correctness
difference, averaging both selectors and placements within world. The practical
loss of interest is 10 percentage points. Use 10,000 whole-world bootstrap draws,
seed **20260915**, for percentile intervals and missing-data bound intervals.
These are approximate descriptive uncertainty summaries under this balanced
finite-population design, not an i.i.d. guarantee or general-task confidence claim.
Keep placement-specific and replacement/inert contrasts secondary. No optional
sample expansion or significance-driven accrual is authorized here.

Report estimates and bounds regardless of validity. A practical-loss exclusion or
interpretable-null flag requires passed baseline controls, complete scheduled
coverage and no simultaneous full/sham floor. Define that floor prospectively as
both strict accuracies at or below 10%; this arbitrary conservative interpretation
guard is not a statistical test. Do not relabel a floor-limited contrast a valid
null. The exact supplied value is the oracle; no model judge supplies ground truth.

## Runtime, provenance and resource bounds

Collect **generation only; no likelihood measurements** in this first instrument.
New modules and schemas are `a164-cohort-v1`, `a164-plan-v1`, `a164-cell-v1` and
`a164-run-v1`. Preserve all consumed A163 modules unchanged. Reuse only their
frozen low-level local NF4 loader, native-token verification, greedy generation,
strict JSON helpers and private immutable writer. Do not label A164 as A163 depth
one. Use the existing exact checkpoint, tokenizer/chat template, NF4/BF16 settings,
four-thread limit, 10 GiB allocator ceiling and 64-token generation cap. The shared
engine configuration retains its historical schema; A164 run identity is separate.

Root's pre-call freeze must record the exact shared configuration SHA, full model
and tokenizer hashes, native template SHA, this protocol SHA, all three new module
SHAs and the reused instruction_binding_runtime.py/instruction_binding_tasks.py
SHAs. Hash-bound plans include the complete cohort, material receipts, system/user
messages and oracle-derived trial identities. Bind the existing successful A163
16-call harmless runtime qualification receipt and unchanged configuration as the
runtime qualification evidence; do not add A164 qualification generations. Audit
native system-role rendering and full token lengths for every new A164 prompt
without model calls; never truncate. Keep exact rendered tokens, response tokens, text, safe
error locations and timings private. Receipt export decodes and re-scores native
tokens before yielding a numeric list; it does not load a model.

Prefer the local shared server. Recheck GPU headroom before load, retain ownership
of task PIDs, and time the initial development batch before committing to the full
matrix. No new weights, paid API or rental spend is needed. Freeze and back up
code before calls. All plans, task outcomes and receipts stay outside Git; only
reviewed aggregates may enter the research readout.
