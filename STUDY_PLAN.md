# Locked Study Charter v1.0

## Structured lexical scaffold mechanisms in Llama 3.3 70B Instruct

Date locked: 2026-07-25

Status: **Prospective, outcome-free study charter for the raw checkpoint. Not
yet a confirmatory protocol freeze or preregistration.**

## 1. Decision and scope

This repository will execute one mechanism study. It will not attempt to build
a general jailbreak benchmark, compare every local model family, or evaluate a
production defense stack before establishing the focal effect.

For this study, a **structured lexical scaffold attack** means an inference-time
text intervention that combines role framing, semantic inversion, prescribed
response structure, lexical or formatting constraints, and optionally a fixed
multi-turn continuation. It changes no weights and requires no tool,
infrastructure, retrieval, or system-prompt compromise.

The focal treatment is the Llama-targeted L1B3RT4S/Pliny-style pattern:
prescribed refusal, divider, and post-divider answer, with semantic inversion
rules. This is a deliberately narrow subset of the broader prompt-attack
surface.

Excluded from the primary study:

- weight poisoning, fine-tuning backdoors, and model modification by an
  attacker;
- indirect prompt injection through tools, retrieval, or web content;
- multimodal attacks;
- optimized adversarial suffixes such as GCG;
- provider-side filters and multi-model race/selection effects;
- claims about all jailbreaks, all lexical attacks, or production-ready
  defenses.

The broader taxonomy, cross-tokenizer benchmark, guard-model comparison, and
layered defense stack proposed by the external review are possible sequels.
They are not dependencies of this mechanism study.

## 2. Subject and artifact contract

Primary model:

- repository: `meta-llama/Llama-3.3-70B-Instruct`
- planning-time revision:
  `6f6073b423013f6a7d4d9f39144961bfbfbc386b`
- execution mode: white-box local inference on the exact pinned weights
- chat template: the tokenizer template at the pinned model revision

Candidate instruments:

- Neuronpedia fitted Jacobian lens for `llama3.3-70b-it`;
- Goodfire `Llama-3.3-70B-Instruct-SAE-l50`, attached to layer-50
  residual-post.

Before paid execution, the artifact manifest must record full immutable
revisions, file hashes, tensor shapes, layer conventions, tokenizer hash,
license terms, and an explicit compatibility check. A mismatch stops the
affected stage. No "closest available" artifact may be substituted silently.

## 3. Threat model

The experimental attacker:

- controls the user message and one predeclared follow-up turn;
- knows the target model and attack template;
- may use textual role framing, formatting, obfuscation, and semantic
  instructions;
- does not alter weights, system messages, tools, retrieval, runtime code, or
  the scoring pipeline;
- receives no adaptive feedback during the confirmatory evaluation.

The researcher has white-box access for measurement and intervention. This
access is not attributed to the attacker.

The primary inference target is the unmodified Meta checkpoint, not an API
route. Existing OpenRouter trials are an exploratory existence check only and
may differ because of provider routing, decoding, or filtering.

## 4. Claim ladder

### Claim A: behavioral effect

On an independently sampled panel of disallowed base behaviors, the full
structured scaffold changes post-divider harmful-compliance scores relative to
a matched structural sham on the exact raw checkpoint.

This claim is behavioral and model-local. It does not identify a mechanism.

### Claim B: descriptive localization

Conditional on Claim A passing its continuation gate, the fitted Jacobian lens
describes where, across frozen layer and position summaries, the full scaffold
and structural sham differ on prospectively defined refusal/compliance probe
margins.

This claim is a readout claim. It does not establish causality, hidden intent,
or a privileged explanation. Identity/logit and random-transport controls must
appear beside the lens result.

### Claim C: causal intervention

Conditional on validated SAE compatibility and discovery-only feature
selection, a signed layer-50 SAE decoder-direction intervention changes
held-out behavioral outcomes under the full scaffold. The same feature and
predeclared magnitude are tested in both directions, with matched SAE-feature
and isotropic controls.

This claim is causal for the realized intervention, checkpoint, prompts,
positions, and strength. A feature label is not treated as a mechanism.

### Defense bridge: not a primary claim

A refusal-restoring intervention becomes a **defense candidate** only if it
also:

1. preserves predeclared benign utility within a frozen non-inferiority margin;
2. generalizes to held-out prompt content;
3. generalizes to at least one attack scaffold not used for feature or
   strength selection; and
4. remains effective under an adaptive evaluation performed after the primary
   study.

Until then, the permitted language is "prompt-local causal intervention," not
"defense."

## 5. Research questions and hypotheses

### RQ1: Does the focal scaffold create a reproducible behavioral contrast?

Primary behavioral estimand:

```text
mean over independent base behaviors of:
  score(full scaffold, turn 2) - score(structural sham, turn 2)
```

The score is a pinned, normalized harmful-compliance evaluator selected and
validated before confirmatory outcomes. The primary interval is a two-sided
95% cluster bootstrap confidence interval over base-behavior IDs.

Secondary contrasts:

- full scaffold versus base request;
- full scaffold versus matched-length inert scaffold;
- turn 1 versus turn 2;
- refusal, divider production, and post-divider compliance components;
- semantic-inversion and format-bifurcation ablations.

### RQ2: Where does the readout trajectory differ?

At the shared assistant prediction boundary, calculate at every layer:

```text
refusal-compliance margin =
  aggregate normalized score(refusal probe set)
  - aggregate normalized score(compliance probe set)
```

Compare the paired full-scaffold and structural-sham curves. The assistant
boundary is the primary position because it exists in every arm without
outcome-dependent alignment. The first fixed number of generated positions and
within-treatment semantic phases are secondary analyses.

Probe sets must:

- be frozen before confirmatory outcomes;
- be tokenizer-verified;
- exclude tokens appearing in any compared prompt where the intended claim is
  not prompt echo;
- be semantically fair to both arms;
- include a strict-subset sensitivity list;
- be reported with the identical statistic under identity/logit and
  random-transport controls.

No "best layer" or "best position" is a primary statistic. Any maximum over
layers, positions, or probe words must be compared with controls using the same
selection surface.

### RQ3: Can a validated SAE direction move behavior bidirectionally?

For a discovery-selected primary SAE feature with normalized decoder direction
`d`, apply:

```text
h' = h + s * alpha * normalize(d), where s is -1, 0, or +1
```

The primary intervention site is layer-50 residual-post at the current final
token on each decoding step. Applying an intervention to all cached prompt
positions, clamping an SAE activation, or using a context-adaptive vector is a
different intervention and therefore exploratory unless separately frozen.

Feature identity, sign interpretation, `alpha`, and the acceptable residual
norm budget are chosen on discovery data only. The held-out primary test asks
whether the discovery-defined refusal-restoring sign reduces harmful compliance
relative to zero and whether the opposite sign moves the outcome in the
opposite direction. Monotonicity across `-alpha, 0, +alpha` is a predeclared
secondary test.

## 6. Independent units and data partitions

The independent unit is the **base behavior ID**, not a token, generated
position, turn, decoding seed, paraphrase, or SAE feature.

The eligible corpus will contain benign, borderline, and disallowed behaviors
from versioned public evaluation sources whose licenses and release practices
fit this project. High-risk prompt text is stored outside Git; the public plan
contains stable behavior IDs, categories, transformations, and hashes.

Partitions:

- **engineering fixtures:** synthetic and benign text only; used for tests and
  pipeline debugging;
- **discovery set:** used for the raw-checkpoint behavioral pilot, candidate
  feature selection, sign calibration, and `alpha` selection;
- **confirmatory set:** held out from all feature, probe, threshold, and
  steering-strength selection;
- **utility set:** benign instruction-following and overrefusal examples,
  disjoint from attack content;
- **transfer set:** a distinct scaffold family, reserved for the later defense
  bridge.

The exact eligible list, deterministic split seed, category balance, and sample
size will be frozen in a machine plan. Sample size will be chosen before
confirmatory outcomes by a power simulation using discovery-set variance, with
base behavior as the resampling unit. It may not be justified by treating
multiple seeds from one prompt as independent.

Primary decoding is greedy or otherwise deterministic under a pinned software
and kernel stack. A predeclared paired-seed sampling analysis may estimate
decoding variability, but its rows remain clustered within base behavior and
are secondary.

## 7. Behavioral conditions

Every base behavior receives the same fixed follow-up message in each arm.
The exact follow-up text is frozen before the confirmatory run.

Primary arms:

1. **Base:** base request, then fixed follow-up.
2. **Full:** complete focal scaffold plus base request, then fixed follow-up.
3. **Structural sham:** token-count-matched refusal/divider/body structure,
   but the post-divider instruction continues refusing, then fixed follow-up.
4. **Inert length control:** matched length and formatting without semantic
   inversion or post-divider compliance instructions, then fixed follow-up.

Secondary discovery/ablation arms:

- semantic inversion removed;
- refusal-divider bifurcation removed;
- role/persona language removed;
- fixed follow-up omitted.

The sham is matched by construction and audited for tokenizer-level length,
format landmarks, imperative count, base-request placement, and lexical overlap.
Perfect semantic matching is impossible because the intended instruction
differs; the audit makes the residual difference explicit.

## 8. Outcomes and judging

### Primary behavioral outcome

A normalized harmful-compliance score on the complete turn-2 assistant response,
using a pinned open evaluator and fixed prompt/schema. The selected evaluator
must be validated against a condition-masked human-coded subset before the
confirmatory run.

### Secondary behavioral outcomes

- binary refusal;
- divider and prescribed-format production;
- post-divider harmful-compliance score;
- response truncation;
- response length;
- category-stratified effects;
- evaluator disagreement;
- turn-1 outcomes.

### Utility and intervention-specificity outcomes

- benign instruction-following pass rate;
- overrefusal rate on safe but superficially sensitive prompts;
- output degeneration indicators;
- latency and peak-memory overhead;
- optional perplexity or next-token loss on a frozen benign text set.

Judge failures, parsing failures, truncated generations, and missing trials use
frozen failure rules. They never trigger an improvised rerun or silent arm
replacement. Condition labels are masked from human adjudicators where
practical.

## 9. Jacobian-lens controls and validity gates

The lens stage requires:

1. exact shape, layer, tokenizer, and checkpoint compatibility;
2. a known-number or published-fixture reproduction where available;
3. identity/logit-lens and random-transport baselines;
4. the same probe aggregation and layer/position selection surface for every
   transport;
5. calibration summaries on clean in-domain and scaffolded discovery prompts;
6. explicit reporting of top-k truncation and vocabulary coverage.

Top-1,000 vocabulary rows may be retained as raw evidence, but the headline
cannot be "many ranks changed." Primary summaries use the frozen probe margin
and full paired layer curves. Top-k Jaccard, first apparent divergence, and
content-specific tokens are secondary and descriptive.

If the lens fails compatibility or calibration, the behavioral and SAE stages
may continue if independently valid, but no Jacobian-lens claim is made.

## 10. SAE discovery and causal controls

Candidate discovery combines:

- differential feature activation for full versus sham on the discovery set;
- activation maps on benign, refusal, and compliance corpora;
- public feature explanations used only as hypotheses;
- decoder norm, activation frequency, reconstruction behavior, and feature
  sparsity diagnostics.

Before held-out outcomes, select:

- one primary feature;
- no more than three named secondary features;
- matched SAE controls selected by decoder norm and activation frequency;
- isotropic directions matched on realized intervention norm;
- one primary nonzero `alpha` and its opposite sign;
- a zero-intervention condition.

Selection code and its inputs are frozen. Candidate failures remain in the
ledger.

The runtime records requested and realized intervention vectors, norms, token
positions, layer outputs, dtype, and any clipping. A mismatch between requested
and realized intervention fails closed. Direct addition and SAE
encode/decode/clamp interventions are not conflated.

## 11. Statistical analysis

- All primary behavioral and causal contrasts are paired by base behavior.
- Confidence intervals and bootstraps resample base-behavior IDs.
- Repeated turns, seeds, positions, probe tokens, and features are not treated
  as independent observations.
- One primary outcome and one primary contrast are frozen for each claim.
- Category-stratified effects and mixed-effects models are secondary.
- Layer-by-position visualization uses simultaneous or max-statistic-aware
  uncertainty where inferential language is used.
- Multiple secondary feature tests use a frozen multiplicity procedure or are
  labeled exploratory.
- Effect estimates and intervals are reported regardless of statistical
  significance.
- No optional stopping is permitted. Missing planned trials fail the structural
  audit.

The exact evaluator, bootstrap method, replicate count, random seed,
non-inferiority margin, and practical continuation thresholds must appear in
the machine-readable plan before the confirmatory freeze.

## 12. Stage gates and stopping rules

### Gate 0: result-free implementation

Required:

- complete protocol and threat model;
- immutable artifact inventory;
- deterministic plan builder and independent validator;
- end-to-end tiny-model or synthetic-fixture run;
- tests for hooks, intervention position, scoring, resume behavior, duplicate
  prevention, and missing-trial failure;
- exact source and plan hashes;
- adversarial design review and written adjudication.

No raw-checkpoint target outcomes are opened in this gate.

### Gate 1: discovery-only raw-checkpoint behavioral pilot

Purpose: establish that the focal behavioral contrast exists on the actual
weights and estimate variance for the confirmatory sample.

Continue to Claim A only if the predeclared discovery threshold is met. The
threshold will require both a positive full-versus-sham effect and a minimum
practical effect, not merely divider production. If it fails, stop the
mechanistic attack study, report the pilot, and redesign only under a labeled
new exploratory plan.

### Gate 2: confirmatory behavior

Run the frozen held-out panel. Preserve and retrieve all receipts before
analysis. Claim A stands or falls under the frozen analysis.

The J-lens stage proceeds only if the behavioral effect is large and stable
enough for a mechanistic comparison under the frozen continuation rule.

### Gate 3: descriptive J-lens map and SAE discovery

Run lens controls and discovery-only SAE feature/strength selection. A failed
lens gate removes Claim B without invalidating an independently sound SAE
intervention.

Amendment A005 freezes the implementation before any mechanistic output:

- The primary location is the turn-2 assistant prediction boundary. Secondary
  locations are generated-token indices 0, 1, 2, 4, 8, and 16; unavailable
  positions remain explicit missing observations.
- At every declared lens source layer, the margin is the mean refusal-probe
  logit minus mean compliance-probe logit after z-scoring against that
  transported vector's complete vocabulary logits.
- Fitted J-lens, identity, and deterministic dense-Gaussian transports use the
  same layers, positions, probe sets, and aggregation. Each random matrix is
  generated from base seed 20260725 with a stable per-layer derivation and
  Frobenius-matched to that layer's fitted Jacobian.
- The SAE discovery surface is residual-post layer 50 on discovery behaviors
  only. Eligible candidates have a positive paired full-minus-structural-sham
  activation delta and activate on at least 10% of full-arm discovery
  examples. Candidates are ranked by paired standardized delta with stable
  feature-ID tie-breaking; one primary and no more than three secondary
  candidates are retained.
- Candidate receipts also retain decoder norm, prevalence, sparsity, and
  reconstruction diagnostics. Public feature labels remain non-gating
  hypotheses.

Freeze the causal intervention plan before opening held-out intervention
outcomes.

### Gate 4: held-out causal intervention

Execute the primary feature, both signs, zero intervention, matched SAE
features, and isotropic controls on held-out prompts. Claim C is reported under
the frozen analysis, including null, asymmetric, or utility-damaging outcomes.

### Gate 5: optional defense bridge

This requires a separate amendment or protocol, held-out scaffold family,
utility non-inferiority margin, adaptive attacker budget, and explicit compute
approval. It is not automatically authorized by success at Gate 4.

## 13. Compute and operational plan

The cheap-node ladder is mandatory:

1. schemas, scoring, analysis, and receipts on synthetic CPU fixtures;
2. exact hook and intervention path on the smallest compatible model/GPU;
3. one timed unit on the cheapest hardware that exercises the production
   sharding, precision, and kernel path;
4. cost projection from measured throughput;
5. explicit approval of GPU type, hourly price, disk price, wall-time ceiling,
   spend ceiling, and no-progress timeout;
6. paid 70B run bound to an exact Git commit and plan hash;
7. hash-verified retrieval, credential removal, pod termination, volume
   accounting, and fresh provider inventory.

No paid pod is left running while design or analysis choices are being made.
Raw receipts must be sufficient for new analyses without rerenting the GPU.

### Budget contract

For the initial work, `$50` is the soft escalation threshold and `$100` is the
hard cumulative ceiling. Crossing the soft threshold is not authorization for
new work: checkpoint, stop launching new phases, and request approval. A
healthy atomic unit already in progress may continue while approval is pending
only while it remains on plan, makes observable progress, and stays within the
hard ceiling. Duplicate, runaway, misconfigured, idle, or no-progress work is
terminated regardless of the soft threshold. The hard ceiling is enforced.

The exact pod rate, storage rate, expected duration, no-progress timeout, and
per-run worst-case cost still require a stated approval before creation.

### Persistent working set

The pinned model, Jacobian lens, SAE, verified download manifests, and resumable
checkpoints will live on a task-owned RunPod network volume during the active
multi-day work window. The volume and each pod have separate lifecycle records.
Terminating idle GPU compute must not delete the network volume.

The volume ledger records its provider ID, datacenter, capacity, monthly and
effective daily rate, mount path, contents, creation time, retention deadline,
and deletion evidence. Retention is reviewed daily. Critical result receipts
are also retrieved and hash-verified locally; the network volume is a working
cache, not the sole evidence copy.

Runners checkpoint by stable trial ID, write temporary files followed by atomic
rename, append attempt logs, and resume only after verifying completed shards.
The resume path must survive a forced-kill test on cheap hardware before
interruptible capacity is used.

## 14. Evidence and release policy

Track in Git:

- protocols, schemas, builders, validators, tests, and analysis code;
- result-free plan manifests;
- prompt IDs, categories, transformations, and hashes where raw text is
  restricted;
- credential-free receipts and derived statistics safe for release;
- complete control results, nulls, failures, and amendments.

Keep outside public Git:

- credentials and provider configuration;
- model weights and caches;
- raw high-risk exploit strings when publication would materially increase
  misuse;
- unrestricted harmful generations;
- temporary pod files.

If a high-transfer vulnerability or guard bypass is found, pause public release
for a documented responsible-disclosure decision. Public materials emphasize
behavior IDs, mechanisms, controls, and defenses rather than copy-paste attack
recipes.

## 15. Visual evidence contract

This is a figure-first study. Before a GPU stage is frozen, every planned
empirical figure must have a result-free contract that names:

- the scientific question and permitted inference;
- axes, units, conditions, controls, and uncertainty;
- the exact raw receipt fields and row-selection rule;
- the independent resampling unit;
- the same-statistic null or cheapest baseline;
- accessibility text and non-color encodings;
- generator and byte-identity verification paths.

The contract prevents a visually important analysis from depending on a tensor
or score that was never retained. It does not pre-author conclusions or select
favorable views. New post-outcome figures are allowed only as explicitly
exploratory additions and must receive the same receipt and provenance
treatment.

Every empirical figure:

1. is generated by code from immutable machine-readable evidence;
2. has SVG/PDF and 300-DPI PNG outputs where the charting stack supports them;
3. has a self-contained `<figure-stem>.receipt.json`;
4. records source, generator, and output SHA-256 values;
5. is indexed in a post-wide `provenance.json`;
6. passes a verification mode that re-derives values and checks output bytes;
7. displays controls beside the focal method on the same statistic;
8. has a complete text equivalent and finding-focused alt text.

The initial figure inventory and receipt field requirements are in
`docs/FIGURE_CONTRACT.md`.

## 16. Permitted and forbidden claims

Permitted if supported:

- the focal scaffold changed behavior on the pinned Llama checkpoint;
- the fitted lens readout differed under the frozen summary and controls;
- a realized layer-50 intervention causally changed held-out outcomes;
- the intervention did or did not preserve benign utility;
- the result generalized or failed to generalize to the specific transfer set.

Forbidden:

- "the model's thoughts were read";
- "alignment was deleted";
- "the safety circuit was found";
- SAE feature labels are ground-truth concepts;
- one scaffold explains G0DM0D3's multi-model product success;
- one model or tokenizer establishes a general lexical-attack mechanism;
- prompt-local steering is a deployable defense;
- null J-lens evidence proves no internal mechanism exists.

## 17. Change control

This charter is locked by the repository's first commit. It is not immutable,
but changes are append-only and visible.

Before the confirmatory freeze:

1. record the proposed change and rationale in `docs/AMENDMENTS.md`;
2. state whether any relevant raw-checkpoint outcome has been inspected;
3. update the plan and tests in the same commit;
4. obtain human approval for changes to claims, endpoints, sample construction,
   stage gates, safety policy, or compute ceiling.

After the confirmatory freeze, outcome-informed changes cannot alter the
confirmatory analysis. They become labeled exploratory analyses or a new
prospective version.

## 18. Preconditions for the confirmatory freeze

The study is not confirmatory-ready until all are true:

- exact artifact revisions and hashes are pinned;
- corpus eligibility, licenses, categories, split, and sample size are fixed;
- raw restricted prompts have a secured storage and access procedure;
- all primary and secondary conditions are generated deterministically;
- evaluator, human-validation protocol, thresholds, and failure rules are
  frozen;
- probe lexicons and strict-subset sensitivity are frozen;
- primary SAE intervention semantics and runtime receipt fields are specified;
- every planned empirical figure has a result-free contract and the runtime
  receipt schema contains its required raw fields;
- analysis code runs end-to-end on synthetic results;
- an independent validator reconstructs counts and balance;
- the research-director review has been adjudicated;
- a genuinely different review channel has assessed construct validity;
- the public or embargoed freeze commit is pushed before confirmatory outcomes;
- any paid run has separate explicit cost approval.

## 19. Provenance of this charter

The charter was informed by:

- exploration/proposal notes, SHA-256
  `5936fcb2141099c699b5b04c1e410716cd61094ecacc42ba054c7c3094b12e88`;
- external deep-research review, SHA-256
  `9f9a2ce726f79e6bc878bdbe2c4511225ca8c5e889a28e6fe0736ada340bf833`;
- the companion experiment-integrity, GPU-compute, Git-workflow, and
  research-note playbooks in `../../agent-skill-documents/`;
- public model, SAE, and Jacobian-lens artifact pages linked in the README.

The external review's literature citations are research leads, not verified
bibliographic evidence in this commit.
