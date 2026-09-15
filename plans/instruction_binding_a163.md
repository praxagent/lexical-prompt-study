# A163: scaffold effects on objectively scored instruction selection

Status: prospective implementation and development preparation. No model results
have been collected under this plan. The final runtime, materials, seeds and
renderers require a recorded freeze before the held-out experiment. This is a new
study; earlier instruments, outcomes and confirmation panels remain unchanged.

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
extra development-only renderer check is diagnostic and must be separately counted.
64 worlds is a precision pilot, not a claim of adequate power: world-difference SD
0.35–0.50 implies roughly 9–12 percentage-point 95% interval half-width. If precision
is insufficient, specify one fresh fixed-size extension without changing endpoints.

## Endpoints and interpretation

Primary outcome is strict correctness: exactly one valid JSON object, exactly the
allowed answer field, and the keyed answer. Invalid/trailing/extra-field responses
fail strict correctness. Retain separate diagnostics for wrong-selector answers,
other answers and format failures, with explicit precedence. A parseable keyed
answer inside an invalid format may be described diagnostically but remains a strict
failure. Do not infer semantic rebinding from format-only damage.

For each world, average full-minus-sham strict-correctness differences across both
selectors and placements. Primary estimand is the mean world difference, expressed
in percentage points, with equal weighting of the two task families. Bootstrap
whole worlds, stratified by task family. Predeclare and retain placement-specific
and family-specific contrasts as secondary; a pooled result must not hide opposite
placement effects. Replacement and padding comparisons distinguish structural and
generic interference hypotheses, with exploratory status stated explicitly.

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
- Imprecise result: prospectively size one new cohort; avoid significance-driven accrual.
- Failed validity controls: repair only on development and use a fresh untouched cohort.

Longer-term directions are a strong text comparator and sequential correction judged
by objectively correct completed tasks per fixed token budget. Model-judge agreement
is not a substitute for exact task correctness. No human reviewer pool is required.
