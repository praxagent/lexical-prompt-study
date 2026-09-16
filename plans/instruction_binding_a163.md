# A163: scaffold effects on objectively scored instruction selection

Status: prospective implementation and development preparation. No model results
have been collected under this plan. The final runtime, materials, seeds and
renderers require a recorded freeze before the held-out experiment. This is a new
study; earlier instruments, outcomes and confirmation panels remain unchanged.
Pre-execution clarification, September 15, 2026: counterbalanced presentation,
conditional answer likelihoods, and a fixed-size primary analysis are specified
below. External suggestions to adapt the test size to observed interval width or
substitute a tighter equivalence margin are not adopted.

## Question and design

Does a fixed scaffold change adherence to an authoritative selector on harmless,
mechanically solvable tasks? The directional hypothesis is lower strict task
correctness under the full scaffold than its structural sham. The practical effect
of interest is a loss of 10 percentage points. Alternative explanations include
format disruption, generic context interference and a failure to transfer to these
tasks. Improvements are reported as improvements and reject the directional claim.

Generate independent lookup worlds with eight-symbol permutation tables. Half
contain one table for a single lookup; half contain two independent tables and apply
the first table and then the second. Each world has two query keys, A and B, whose correct answers differ at the
specified depth. Validate the oracle independently. Randomize
mapping, keys and selector labels; record their balance. Keep all descendants of a
world together. A system message selects A or B; within each scaffold/placement
condition, changing this message must leave user payload bytes identical.

Assign query presentation order AB/BA before calls, with exact balance within
each depth. At depth two, also counterbalance forward/reverse display of the
named tables, crossing it evenly with query order. Reversing display never
changes the required composition `table_2(table_1(key))`; depth-one worlds
display only `table_1`. Hold this assigned presentation fixed across all 18
cells and renderer diagnostics for a world. Logical world identity excludes
presentation, so reordered copies cannot become additional independent worlds.
Random permutation contents and query-label assignment distribute answer symbols;
record the realized symbol balance without selecting worlds using model outcomes.

Each world has 18 cells:

| Factor | Levels |
|---|---|
| Scaffold | Full, structural sham, structural replacement, inert padding |
| Placement | Before or after the user task payload |
| Selector | A or B |
| Additional baseline | No scaffold under both selectors |

Reuse existing private material through a separately verified adapter. Do not
reconstruct or embed those materials in public source. Record material hashes and
actual target-token lengths. Inert padding needs a frozen matching rule; record
length mismatches rather than describing byte matching as token matching.

The outcome-free native-template token audit found all four standalone material
strings to contain 252 target tokens. Under the development JSON renderer,
full/sham inputs match exactly within each world, selector and placement
(494–575 tokens). Before-payload replacement/inert inputs have one additional
boundary token (495–576); their after-payload versions match the 494–575 range.
Preserve those existing material bytes and record this secondary-control mismatch;
do not claim all rendered arms are exactly length matched. No-scaffold JSON
inputs span 242–323 tokens and the line-table baseline spans 194–250. Bind the
audit to the final tokenizer/template and regenerate its metadata if either changes.

## Development and held-out separation

Use 16 development worlds, balanced by task depth; the initial eight-world timing
pilot is a subset. Freeze a JSON-world renderer and a line-table renderer. Qualify
both renderer syntaxes and selector competence on development material, then assign
one locked renderer to the held-out cohort. This tests new worlds under a renderer
already qualified for basic competence; do not claim wholly unobserved-renderer
generalization. No task or renderer repair may follow held-out inspection.

Use 64 untouched worlds, 32 per task depth, with disjoint mapping/query identities.
Generate all seeds and sample identities before model execution, recording a cohort
manifest. Different selectors, placements, conditions and repeated decodes are not
independent samples. If development changes consume an intended test world, retire
it and generate a new untouched cohort before freezing; never silently relabel it.

The initial main matrix has 288 development plus 1,152 held-out completions. An
additional no-scaffold A/B check under the second renderer is diagnostic: 16
extra cells for the eight-world pilot and 16 for the remaining eight development
worlds. Both development subsets must preserve depth and presentation balance.
These diagnostics are excluded from the 18-cell primary matrix and counted
separately. The held-out cohort has one locked primary renderer and no such
additional diagnostic cells.
64 worlds is a precision pilot, not a claim of adequate power: world-difference SD
0.35–0.50 implies roughly 9–12 percentage-point 95% interval half-width.
The primary sample is fixed at 64 worlds and analyzed once after its scheduled
run, including unresolved failures. Do not enlarge this primary cohort based on
its effect, significance, or observed interval width. A later independent study
may use a separately frozen sample size; do not pool a width-triggered extension
and report an ordinary fixed-sample interval as though its stopping rule were fixed.

## Endpoints and interpretation

Primary outcome is strict correctness: exactly one valid JSON object, exactly the
allowed answer field, and the keyed answer. Invalid/trailing/extra-field responses
fail strict correctness. The parser first assigns `format` to invalid schema or
extra content; for a valid schema it assigns `exact`, `other_selector`, or
`other_answer`, in that order. Token-cap termination remains a strict failure
even if its visible text parses as an exact answer. Report these capped outcomes
separately from the four parser categories, retaining both the parser diagnostic
and the strict success bit. A parseable keyed
answer inside an invalid format may be described diagnostically but remains a strict
failure. Do not infer semantic rebinding from format-only damage.

For each world, average full-minus-sham strict-correctness differences across both
selectors and placements. Primary estimand is the mean world difference, expressed
in percentage points, with equal weighting of the two task families. Bootstrap
whole worlds, stratified by task family. Predeclare and retain placement-specific
and family-specific contrasts as secondary; a pooled result must not hide opposite
placement effects. Replacement and padding comparisons distinguish structural and
generic interference hypotheses, with exploratory status stated explicitly.

Use 10,000 percentile-bootstrap replicates with seed 20260915, independently
resampling whole worlds within each depth while equally weighting the two
families. Do not resample cells as independent trials. The primary contrast uses
all scheduled worlds: if any contributing generation is unresolved, no fully
observed primary point estimate is reported. Instead, obtain sharp worst/best
binary-outcome bounds by setting each missing full-arm success to 0/1 and each
missing sham-arm success to 1/0. Average those bounds within worlds and families;
bootstrap the lower and upper bound estimators to give an explicitly labeled
outer uncertainty envelope. Report missing cells and complete-world counts.
This is an approximate bootstrap uncertainty description, not a simultaneous
guarantee across exploratory contrasts or a missing-at-random assumption.
Missing auxiliary-arm or diagnostic cells do not erase an otherwise observed
full-versus-sham contrast.

A 95% interval excluding losses of 10 percentage points or greater narrows the
prespecified disruption hypothesis. It does not establish two-sided equivalence,
zero effect, or a post hoc 5–7.5-point equivalence margin. Retain the numeric
estimate and interval rather than forcing every outcome into a success category.

## Conditional answer-likelihood secondary outcome

Independently of free generation, append the fixed assistant prefix
`{"answer":"` to the native rendered conversation. Score the selected and
unselected canonical candidate suffixes, each consisting of its symbol followed
by the closing quote and brace. Exclude EOS. For each candidate, sum the
teacher-forced log probabilities of its suffix tokens only, in natural-log units;
do not score the supplied prompt/prefix or normalize by suffix length. Record
both token counts. Symbols need not be single tokenizer tokens.

Require exact token-prefix stability: tokenizing the context plus each suffix
must begin with exactly the same frozen context token IDs. Reject the secondary
score if this boundary check or finite-probability check fails; never silently
retokenize or change the candidate definition to obtain a favorable score.
The margin is selected minus unselected summed log likelihood. Analyze its
full-minus-sham change with the same within-world pairing and family weighting
as the primary outcome. Its unit is nats per paired task world, conditional on
the supplied canonical JSON prefix; it does not measure unconditional free
generation or remove all output-token effects. Keep this outcome secondary,
including when it is more favorable than strict correctness.

Generation failure and likelihood failure are separate coverage events. Missing
likelihoods have no justified finite worst/best bound: report full-cohort margin
estimates only with complete contributing coverage, and label any complete-world
descriptive subset explicitly with its reduced denominator. Do not replace a
missing score by zero or infer a margin from the generated answer. Any later
causal use requires its own controls and prospective protocol.

No-scaffold correctness and selector-switch success must each be at least 90% in
development. Inspect each depth and renderer. Wrong answers, malformed output and
token-cap terminations count as task failures. Infrastructure failures may be retried
at most twice with unchanged configuration; preserve attempts and unresolved cells.
Report best/worst-case effect bounds for unresolved cells and complete-world counts.
No response is discarded because it is inconvenient for the hypothesis.

All scorer fixtures must pass, including correct answer, wrong selector, wrong key,
extra field, trailing text and invalid JSON. A selector-switch positive control must
change the answer to the newly correct value. A failed control limits the inference;
repair on development material and retain failures, rather than interpreting a
broken experiment as a null.

## Runtime and private data

Prefer local shared CPU/RAM/GPU. Check live capacity before loading a model; do not
interrupt other agents. Candidate target is an 8B instruct checkpoint in a separately
qualified implementation that fits the approximately 16GiB GPU. The existing full
BF16 runtime is not assumed to fit. Before execution pin checkpoint, tokenizer, chat
template, quantization/backend versions, precision, attention implementation, seeds,
EOS handling, maximum prompt length and a 64-token generation cap. Use deterministic
decoding, batch one initially. Verify native system-role construction, repeated-run
agreement, and any batching/caching parity before changing execution settings.

Do not silently substitute a different model or precision after observing outcomes.
Quantized results concern that implementation; they do not replicate earlier precision
conditions automatically. Optional residual capture at three prespecified layers at
the final prompt token requires its own hook-integrity check. Behavioral execution
need not collect broad internal features or select layers using held-out outcomes.

The timing pilot determines actual throughput and memory use. Provisional estimates
are 4–24 local GPU-hours and below 1GB for selected states, not guarantees. No API
judging is required. A rental remains a fallback under the owner's applicable scoped
compute authorization; the consultation budget does not authorize unrelated spending.
Store cohort manifests, material adapters, prompts, completions, hashes and receipts
under ../lexical-prompt-study-data/runs/a163, never Git or agent transcript. Public
code/tests contain harmless synthetic fixtures only. Emit aggregate results to agents.

## Result-dependent next experiment

- Substantial wrong-selector effect with successful controls: develop token-aligned,
  bidirectional activation-transfer tests with self-patch and mismatched-donor controls.
- Format-only effect: investigate output-contract disruption rather than claiming
  selector rebinding.
- Padding-equivalent effect: test context interference with frozen matched controls.
- Opposite-sign effect: test scaffold-assisted task selection; preserve the improvement.
- Precise null excluding the practical loss: prioritize a strong semantic text baseline
  against existing internal predictors, preserving the measured-horizon endpoint.
- Imprecise result: prospectively size a separately reported new study; do not
  adaptively extend or replace this fixed 64-world primary result.
- Failed validity controls: repair only on development and use a fresh untouched cohort.

Longer-term directions are a strong text comparator and sequential correction judged
by objectively correct completed tasks per fixed token budget. Model-judge agreement
is not a substitute for exact task correctness. No human reviewer pool is required.

## Local runtime qualification before the pilot

The first GPU job uses only harmless no-scaffold inputs from two separately seeded
worlds disjoint from development and held-out cohorts. Cross both renderers and
selectors and repeat each input twice: 16 calls. Require completed generation and
finite conditional likelihoods, exactly repeated generated token sequences, and
repeated candidate log likelihoods within 1e-5 nats. Accuracy is reported descriptively
at this stage; it is not substituted for the development competence gate. This job
does not test a scaffold effect and its worlds never enter that effect estimate.

Freeze NF4 weights with double quantization and BF16 compute, batch one, SDPA,
64 generated-token cap, and a 2,048-token prompt limit. The native template's date
argument is fixed to 26 Jul 2024. Pin all local model/config/tokenizer files by SHA-256
and package versions in the runtime configuration. The dedicated Python 3.11 runtime
uses its own inference add-ons and reads existing installed PyTorch dependencies
without modifying the other project's environment. Record this dependency source.

After qualification, run the 160-cell development pilot. All ordinary task failures
remain observations; infrastructure failures have no automatic retries. Diagnose
failures using exception types and traceback locations without printing payloads.
Only explicit bounded infrastructure retries are allowed. Proceed to the remaining
development cohort and later held-out freeze according to the stated controls.
