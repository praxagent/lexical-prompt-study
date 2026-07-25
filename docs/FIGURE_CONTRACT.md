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
