# A164 proposal: direct selection of an explicitly supplied answer

**Status: proposal only; no execution or results.** This is a new experiment
motivated solely by A163's failed no-scaffold competence controls. It is not
another A163-v2 wording repair. Preserve both A163 runs and their failed gates;
their floor-limited scaffold contrast does not establish absence of an effect.
Root must review, freeze and back up this instrument before any calls.

## Task and independent unit

Use one deliberately simple family: **direct answer selection**. The system
message names authoritative selector A or B and requires exactly
`{"answer":"<selected value>"}`. The user supplies two explicit distinct values,
for example `{"A":"s03","B":"s11"}`. The task copies the selected value; it
requires no lookup table, composition, arithmetic, or inferred transformation.
One family is enough to test this question; adding a nominal second family would
not establish broader task generalization.

A world is an ordered A/B value pair drawn from 16 fixed opaque symbols
`s00`–`s15`. There are 240 possible distinct-value pairs, enough for 16 development
and 64 disjoint held-out worlds. Freeze a seeded manifest before execution,
sampling unique pairs without replacement and balancing each symbol's occurrence
under A and B within each cohort: once per label in development, four times per
label in held-out. Verify feasibility and all balance counts before freezing.
Use eight AB-first and eight BA-first development worlds, and 32 of each held-out.
Keep a world's presentation order fixed across all its conditions. A/B selector
switches change only the system message; user bytes must remain identical.

The world is the sampling and analysis unit. Selectors, placements and scaffolds
are paired descendants, not independent observations. This is a small balanced
finite task population, not a representative sample of general instruction tasks.

## Gate, freeze and experiment

Freeze one prompt, one JSON renderer, scorer, seeds and runtime before running
the 32 no-scaffold development cells. Require at least 90% strict cell correctness
and at least 90% of worlds with both selector answers correct: at least 29/32
cells and 15/16 pairs. Report presentation-specific counts. Missing infrastructure
outcomes prevent passage; malformed outputs and cap terminations are failures.
If the gate fails, stop A164. No model-guided example addition, wording repair,
renderer search or substitution of another model under this protocol.

If it passes, lock the instrument and run 64 fresh worlds under the original
18-cell design: full/sham/replacement/inert × before/after × A/B, plus no-scaffold
× A/B, totaling 1,152 held-out completions. No development scaffold sweep is
needed. Apply the same 90% no-scaffold gates in held-out: at least 116/128 cells
and 58/64 pairs. Failure limits interpretation and never licenses dropping worlds.

Primary estimand: equal-world mean full-minus-sham strict correctness, averaging
both selectors and placements within world. Retain the prospective 10-percentage-
point loss as the practical effect of interest. Report whole-world bootstrap
intervals, placement-specific secondary contrasts, and replacement/inert secondary
comparisons. Sixty-four worlds remains a precision pilot, not a power guarantee.
Keep format, wrong-selector and other-answer errors separate; preserve missing-cell
best/worst bounds. A valid null concerns this direct-selection instrument only.
The oracle is the supplied value, with no LLM-derived ground-truth labels.

## Implementation and resources

Add an explicit-answer world type, direct oracle, and new `a164-plan-v1`/
`a164-cell-v1` adapters. Do not label this task as A163 depth one or reuse A163
plan identities. Reuse frozen low-level native-template tokenization, NF4 loading,
deterministic decoding, 64-token cap, private receipts and strict JSON parsing;
keep all consumed A163 sources unchanged. Optional likelihoods require separately
declared A164 bindings and cannot replace behavioral competence.

Prefer the available local server and existing pinned model/runtime, with the
same four-thread limit and observed GPU headroom check. Time the first development
baseline cells before scheduling the remaining work; estimate completion time
from measured throughput. The initial gate costs 32 short generations; passage
permits 1,152 more. No API, rental or model download is needed. Keep all task plans,
responses and tokens private; release only reviewed aggregate results.
