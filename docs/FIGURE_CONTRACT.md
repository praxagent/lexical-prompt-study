# Result-free figure contract

Status: planning inventory, no empirical values

Date: 2026-07-25

The study is designed for visual inspection without allowing figures to become
an untracked analysis surface. Each empirical figure below must be generated
from immutable receipts, show its controls, and carry a self-contained
per-figure receipt plus a post-wide provenance entry.

Conceptual figures may be hand-authored because they contain no empirical
geometry or result values. Any number added to a conceptual plate must still
trace to a source or receipt.

## Shared receipt fields

Every model-touching trial receipt must retain enough information to build the
contracted figures without another GPU run:

- stable study, plan, run, trial, base-behavior, category, split, arm, turn,
  seed, and attempt IDs;
- exact model, tokenizer, chat-template, lens, and SAE revisions and hashes;
- prompt/transformation hashes and token IDs;
- generated token IDs, decoded text access classification, finish reason, and
  token counts;
- primary evaluator score, secondary judge components, judge revision,
  parsing status, and human-validation linkage;
- refusal, divider, post-divider, truncation, and format indicators;
- J-lens top-k token IDs and scores by transport, layer, and frozen position;
- frozen probe-set scores and refusal-compliance margins;
- SAE feature activations, selected feature IDs, decoder norms, requested and
  realized intervention vectors/norms, sign, alpha, layer, and token positions;
- matched-control identities and matching diagnostics;
- latency, peak memory, dtype, sharding, kernel path, software versions, and
  determinism settings;
- error, retry, resume, source-commit, plan-manifest, and shard hashes.

## Conceptual figures

### C01. What changes and what does not

Question: What access does the attack have?

Visual: text-only scaffold changes the forward-pass context; weights, tools,
runtime, and infrastructure remain unchanged.

Teaching job: establish the threat model before introducing internals.

### C02. The dual-channel phase machine

Question: Why can naive refusal scoring miss the attack?

Visual: base request versus prescribed refusal, divider, and post-divider body.

Teaching job: motivate post-divider scoring without asserting that the
mechanism has been observed on the raw checkpoint.

### C03. Claim ladder and stop gates

Question: Which evidence supports behavioral, descriptive, causal, and defense
claims?

Visual: behavioral gate to J-lens map to SAE intervention to optional transfer
test, with a stop branch at every failed gate.

### C04. Evidence-to-figure provenance chain

Question: How can a reader verify a plotted value?

Visual: remote raw shard to immutable run receipt to per-figure receipt to
generator/verification to published figure and prose statistic.

## Empirical figures

### E01. Full scaffold versus structural sham

Question: Does the scaffold change turn-2 harmful compliance?

Chart: paired dot or interval plot by arm, with the paired full-minus-sham
effect and cluster-bootstrap interval.

Controls: base and inert-length arms on the same score and axis.

Independent unit: base behavior ID.

Required fields: arm, behavior ID, category, turn, primary score, parsing and
failure status, bootstrap seed and cluster membership.

Forbidden overread: a positive effect does not identify an internal mechanism.

### E02. Response-phase outcomes

Question: Is the effect refusal suppression, format completion, post-divider
compliance, or a combination?

Chart: condition-aligned proportions or a small multiple showing refusal,
divider, and post-divider compliance with intervals.

Controls: every primary arm; turn 1 and turn 2 clearly separated.

Required fields: binary phase indicators, response length, truncation,
behavior ID, arm, and turn.

Forbidden overread: divider production alone is not harmful compliance.

### E03. Layerwise J-lens refusal-compliance margin

Question: At the shared assistant boundary, where do the paired readout curves
differ?

Chart: layerwise paired margin difference with uncertainty.

Controls: J-lens, identity/logit, and random transports on the identical probe
and layer summary.

Required fields: transport ID, layer, frozen position, probe token IDs and
scores, aggregate margin, behavior ID, arm, and random-control seed.

Forbidden overread: a readout difference is not a causal mechanism or hidden
thought.

### E04. Layer-by-position trajectory map

Question: How does the descriptive contrast evolve over early generation?

Chart: layer-by-position heatmap of paired margin differences, paired with a
same-statistic random-transport null panel.

Controls: same selection surface under identity and random transports.

Required fields: all E03 fields for each frozen generated position, token IDs,
position availability, and missing-position reason.

Forbidden overread: the visually strongest cell is a max-statistic.

### E05. Discovery-only SAE candidate map

Question: Which SAE features were considered, and why was the primary feature
selected?

Chart: activation-delta versus activation-frequency or decoder-norm diagnostic,
with selected and rejected candidates visibly retained.

Controls: benign/refusal corpora and matched-control selection diagnostics.

Required fields: feature ID, split, activation summary, prevalence, decoder
norm, reconstruction diagnostics, public-label provenance, and selection rule.

Forbidden overread: labels are hypotheses, and this discovery figure is not
held-out evidence.

### E05a. Discovery calibration stop surface

Question: Did any prospectively frozen dose qualify for held-out causal
confirmation?

Chart: signed half-span and bootstrap interval by rho against the frozen
efficacy threshold, paired with directional components and the two independent
runtime safety ceilings.

Controls: zero intervention and both signs at every frozen rho; every candidate
is retained, including ineligible candidates.

Independent unit: discovery base behavior ID.

Required fields: rho, realized alpha, restoring sign, restoring-minus-zero,
opposite-minus-zero, signed half-span and interval, representation mismatch,
effective delta-to-residual ratio, parsing, truncation, runtime-error, and
eligibility fields.

Forbidden overread: this is discovery calibration, not held-out causal evidence;
passing runtime gates does not establish efficacy, and a failed efficacy gate
does not prove that no SAE intervention could work.

### E06. Bidirectional SAE dose response

Question: Does the same frozen direction move held-out behavior in opposite
directions?

Chart: outcome versus signed alpha with zero, matched-feature, and isotropic
controls.

Controls: every realized intervention shown on the same axis, including null
features and zero.

Independent unit: base behavior ID.

Required fields: feature/control ID, requested and realized sign/alpha/norm,
behavior score, behavior ID, arm, token positions, clipping, and runtime
validation status.

Forbidden overread: asymmetry or output degeneration cannot be hidden by the
signed headline contrast.

### E07. Safety-utility frontier

Question: Does refusal restoration preserve useful behavior?

Chart: harmful-compliance reduction versus benign utility or overrefusal, with
uncertainty and unsteered baseline.

Controls: zero intervention and matched directions.

Required fields: attack and utility dataset IDs, scores, non-inferiority margin,
latency, intervention condition, and behavior cluster.

Forbidden overread: a point that improves safety by collapsing utility is not a
defense.

### E08. Held-out scaffold transfer

Question: Does the intervention generalize beyond the feature-selection attack?

Chart: effect sizes by seen versus held-out scaffold family, with utility
alongside them.

Controls: unsteered, matched-feature, and cheapest external baseline chosen in
the later defense protocol.

Required fields: scaffold family, seen/held-out designation, adaptive budget,
all E06 behavioral fields, and utility fields.

Forbidden overread: this figure is absent unless the separate defense-bridge
gate is authorized and frozen.

### E09. Follow-up selected SAE candidate across complete arms

Question: Does the calibration-selected Llama 3.1 8B SAE candidate distinguish
the complete scaffold from base, inert-length, and structural-sham controls in
both frozen scaffold placements?

Chart: grouped mean-activation bars for discovery and calibration, faceted by
`ep_before_request` and `ep_after_request`, with positive-activation counts
printed for every arm.

Controls: shared base, ordering-specific inert-length, and ordering-specific
structural-sham arms appear beside the full arm. Placements are never pooled.

Independent unit: base behavior ID within discovery or calibration.

Required fields: partition, placement, candidate ID and feature IDs, arm,
mean activation, positive count, total count, paired full-minus-sham mean,
RMS, and standardized effect.

Permitted inference: the selected coordinate is a reproducible
placement-stratified internal correlate on this model, SAE, and sample.

Forbidden overread: positive activation is not a fitted detector threshold,
and the figure does not establish causality, benign specificity, or a circuit
breaker.

### E10. Frozen SAE candidate selection

Question: Which prospectively declared SAE candidate maximizes the minimum
calibration standardized full-minus-structural-sham effect across the two
scaffold placements?

Chart: grouped calibration standardized-effect bars for the single-feature
and linear-subspace candidates, with direct before/after labels and the
maximin minimum printed for each candidate.

Controls: both orderings are shown separately for every candidate; the frozen
selector uses the lower of the two values.

Independent unit: base behavior ID in calibration.

Required fields: candidate ID, kind, feature IDs, weights, placement-specific
mean, RMS, standardized effect, eligibility, minimum raw mean, minimum
standardized effect, and selected-candidate identity.

Permitted inference: under the frozen maximin rule, the single feature is the
selected candidate for any later threshold-fit or intervention study.

Forbidden overread: a higher calibration contrast is not evidence that the
candidate causes behavior or supports deployment.

### E11. Placement-stratified Jacobian-lens trajectories

Question: What full-minus-structural-sham refusal-minus-compliance margin does
each declared transport expose across the Llama 3.1 8B source layers, and does
the shape replicate from discovery to calibration?

Chart: four line panels, discovery and calibration crossed with
`ep_before_request` and `ep_after_request`, over all source layers. Fitted
Jacobian-lens, identity, and deterministic Frobenius-matched Gaussian
transports share each axis and use distinct colors and line styles.

Controls: identity and random-Gaussian trajectories appear in every panel.
Placements and partitions are never pooled.

Independent unit: base behavior ID within partition.

Required fields: partition, placement, layer, transport, full and
structural-sham means, paired difference, behavior-bootstrap interval, shared
base and inert-length summaries, transport norm metadata, and probe identity.

Permitted inference: the fitted trajectory is a reproducible descriptive
readout with ordering-dependent sign and depth structure on this exact setup.

Forbidden overread: fitted-versus-random visual structure does not establish a
causal circuit, transport equivalence, moderation, or detection performance.

### E13. Complete feature-6779 factorial prevalence

Question: Does strict-positive feature-6779 activation require harmful content,
inert length, matched structure, or the full scaffold?

Chart: two placement panels containing all request-class by material prevalence
cells, with positive counts printed on every bar. The shared no-scaffold
reference is repeated only for visual alignment.

Independent unit: prompt family ID within request class.

Required fields: request class, placement, material, strict-positive count,
denominator, prevalence, and canonical-size identity.

Permitted inference: placement-stratified strict-positive prevalence on the
pinned model, SAE, materials, and fixed request panels.

Forbidden overread: strict positivity is not a detector threshold, and the
figure does not establish harmful specificity, causality, or deployment value.

### E14. Primary feature-6779 effects and interactions

Question: Does the full scaffold add activation beyond matched structural sham,
and is that increment uniquely larger for harmful requests?

Chart: full-minus-sham effects for neutral, benign, and harmful requests plus
the two frozen harmful-minus-comparator interactions, separately for each
placement, with the single familywise simultaneous stability interval.

Independent unit: prompt family ID, resampled independently within request
class while preserving every paired arm and both placements.

Required fields: estimate, simultaneous lower and upper bounds, practical
margin, placement, request class or interaction identity, bootstrap seed,
replicate count, and complete-vector critical value.

Permitted inference: exact-material fixed-panel effects and the prespecified
interaction decision for each placement.

Forbidden overread: no population, request-class-independence, placement,
threshold, causal, or deployment claim.

### E15. Secondary frozen-subspace and Jacobian-lens readouts

Question: Do the frozen eight-feature subspace and assistant-boundary
Jacobian-lens margin show a similar full-over-sham pattern?

Chart: four placement-by-readout panels showing mean full-minus-sham contrast
and observed prompt-family range for each request class.

Independent unit: prompt family ID within request class.

Required fields: metric identity, placement, request class, paired contrast
mean, minimum, maximum, and count.

Permitted inference: descriptive behavior of the two frozen secondary readouts
on the complete canonical receipt matrix.

Forbidden overread: neither secondary readout is reused for the primary
decision, threshold fitting, causal localization, or detector performance.

### E21. Weaponization calibration gate

Question: Does the frozen two-key candidate meet the 0.90 recall floor and
0.02 false-positive ceiling in every critical negative stratum, separately
for scaffold-before-request and scaffold-after-request?

Chart: placement columns with positive recall above and all six negative-stratum
false-positive rates below, including exact counts and frozen gate lines.

Independent unit: calibration request or harmless-wrapper family.

Permitted inference: calibration eligibility on the pinned model, candidate,
placements, and fixed panels.

Forbidden overread: no held-out, successful-weaponization, adaptive-attack,
causal, or deployment claim.

### E22. Weaponization candidate comparison

Question: Which frozen readout supplies ranking power, and do scaffold-only,
lexical, or structural baselines meet the same calibration gate?

Chart: placement-separated AUROC and average-precision bars for the two-key
candidate, J-lens head, feature 6779, frozen subspace, exact match, private
fuzzy five-byte-gram coverage, and structural head.

Independent unit: critical calibration observation.

Permitted inference: descriptive calibration ranking and gate eligibility.

Forbidden overread: no out-of-sample, behavioral-success, causal, or
production-superiority claim.

### E23. Weaponization SAE contrast decomposition

Question: Do feature 6779 and the frozen subspace distinguish harmful from
benign use of the attack scaffold, attack from harmless structure around the
same harmful request, or their difference-in-differences?

Chart: three fixed mean contrasts for each SAE readout and placement.

Independent unit: calibration request or wrapper family.

Permitted inference: fixed-panel mean SAE contrasts.

Forbidden overread: no SAE-only detector, uncertainty, causal, or generic
harmless-scaffold population claim.

### E24. Weaponization Jacobian-lens trajectories

Question: Across model depth, how do attack-plus-harmful, attack-plus-benign,
harmless-plus-harmful, and sham-plus-harmful prompts differ?

Chart: both placements, all 31 source-layer mean refusal-minus-compliance
coordinates, and four claim-defining strata.

Independent unit: calibration request or wrapper family.

Permitted inference: descriptive placement-specific internal trajectory
separation on calibration.

Forbidden overread: no selected layer, causal circuit, held-out replication,
or behavior-success classification.

## Per-figure receipt schema

Each empirical figure gets `<figure-stem>.receipt.json` containing:

- figure ID, title, question, description, and highly descriptive alt text;
- permitted inference and explicit non-claims;
- source receipt paths and SHA-256 values;
- exact row filters, groupings, transforms, estimands, and uncertainty method;
- expected and realized row/cluster counts;
- derived plotted-data table;
- generator path, generator SHA-256, plotting-library version, and command;
- SVG/PDF/PNG paths and SHA-256 values;
- accessibility checks and text-equivalent content;
- verification timestamp and byte-identity result.

The post-wide `provenance.json` indexes every figure receipt and every
evidentiary number in the prose. Verification must fail if a source hash,
selection guard, derived value, generator hash, output byte, or pinned path
drifts.
