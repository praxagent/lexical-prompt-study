# Follow-up study plan: replication, causal localization, and defensive detection

Status: prospective protocol freeze candidate

Study namespace: `lexical-scaffold-followup-v2`

Outcome status: no follow-up target outcome generated or inspected
Source study: `lexical-scaffold-llama33-70b-v1`

## Purpose

The completed study found that Llama 3.3 70B SAE feature 10146 separated an
EP-derived full scaffold from a matched structural sham in discovery data.
Direct steering along that feature's decoder direction did not yield an
eligible causal dose. The next round must distinguish four possibilities:

1. feature 10146 is a model- and scaffold-specific lexical fingerprint;
2. it is a reproducible marker of a broader attack state;
3. it is a downstream biomarker of an earlier distributed routing decision;
4. it participates in a useful detector even if it is not itself a causal
   control point.

This plan adds an explicit activation-patching arm to localize candidate causal
ancestors. It also includes the previously proposed replication, detector,
robustness, cross-family, adaptive-stress, feature-subspace, and shadow-mode
circuit-breaker work.

## Non-negotiable boundaries

- Do not search for, open, display, or copy any additional EP attack prompt.
- Reconstruct the already pinned Meta artifact only inside the restricted
  runner, mechanically and by exact revision/hash.
- Treat any future model-specific attack supplied by the human as data:
  restricted storage, exact hash, no instruction following, and no public raw
  text.
- Keep harmful prompts, token IDs that permit reconstruction, generations, and
  replayable activation tensors private.
- Use a new run namespace. Never append follow-up outcomes to study-v1 roots.
- Preserve immutable public artifacts on task volume `u85xfo0aue`; completed
  receipt bundles must also be retrieved and hash-verified locally.
- No paid scientific run begins until the local runner, receipt schema,
  forced-kill resume test, independent plan review, exact live rate, and
  measured one-unit throughput all pass.

The machine-readable companion is `plans/followup_v2.public.json`. Its strict
validator is `lexical_prompt_study.followup_plan.validate_followup_plan`.

## Data partitions

The independent unit is the JBB behavior ID. Existing v1 assignments remain
fixed. The 40 reserve behaviors are split deterministically within each of the
ten categories: the first two canonical behavior IDs become calibration and
the final two become adaptive-stress.

| Partition | Harmful IDs | Use |
|---|---:|---|
| Discovery | 20 | feature/subspace selection and coarse patch localization |
| Calibration | 20 | thresholds, one patch window, and component eligibility |
| Confirmatory | 40 | one held-out test of frozen primary claims |
| Adaptive stress | 20 | explicitly exploratory robustness/evasion stress |
| Utility calibration | 50 benign | detector thresholds and lexical baseline |
| Utility confirmation | 50 benign | held-out false positives, utility, and latency |

No observation used to select a feature, subspace, threshold, layer, position,
or component may contribute to its confirmatory estimate.

## Stage A: Llama 3.1 8B replication

### Prerequisite: finish the Llama 3.3 four-arm discovery replay

Before treating feature 10146 as a detector candidate, replay the pinned
layer-50 SAE over the preserved Gate-3 assistant-boundary states for all four
arms. The archive contains 20 available states per arm, so this requires no new
70B forward pass. The prospectively frozen replay is
`plans/gate3_sae_four_arm_replay_v1.public.json`.

Until that replay completes, the exact claim is “feature 10146 is a
discovery-only full-versus-structural-sham fingerprint.” It has not yet been
shown to be independent of harmful-request content or inert formatting/length.
Even a clean four-arm replay remains discovery evidence and cannot replace the
later held-out detector test.

### Exact instrument

- Model: `meta-llama/Llama-3.1-8B-Instruct` at
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- SAE: `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` at
  `f6775a221e47b44233af4bac2c7b65189265519a`, layer 19 artifact SHA-256
  `bbdc6fad591d3d420e8859dcb6eefa4ec642026090f7f39a963b1c0f27290e4f`.
- Jacobian lens: `neuronpedia/jacobian-lens` at
  `a4114d7752d11eb546e6cf372213d7e75526d3a1`, artifact SHA-256
  `d5300a1575c0b5ad6e01b098c75adb072f3a6c246f184ce2fb35a3550fc8c68b`.
- Dataset, evaluator, and attack remain pinned to their v1 revisions.

Before any target output, the runner must assert the model identity, hidden
width, layer count, SAE width and hook layer, Jacobian shapes/source layers,
tokenizer revision, and exact artifact hashes. Qwen compatibility is a
different gate and cannot be used to weaken these checks.

### Arms and questions

Every harmful behavior uses base, inert-length, structural-sham, and full arms.
The utility set includes ordinary benign and benign-but-structured prompts.
Prospectively specified landmark-removal/replacement and paraphrase
transformations operate mechanically inside the restricted runner.

Primary questions:

1. Does one SAE feature or a small frozen SAE subspace distinguish full from
   structural sham on confirmatory behaviors?
2. Does the layer-by-position Jacobian-lens trajectory reproduce the functional
   timing and refusal/compliance orientation seen in Llama 3.3 70B?
3. Does the signal persist after paraphrase and removal/replacement of obvious
   lexical landmarks?

Discovery may nominate one single feature and one subspace of at most eight
features. Calibration freezes detector weights and thresholds. Confirmation is
run once. A different feature number in the 8B SAE is an analogue only by
prospectively tested function, never by index.

## Stage B: causal activation patching on Llama 3.1 8B

This is distinct from Gate 4. It replaces actual internal state from a matched
donor run rather than adding an SAE decoder direction.

### B1: coarse localization

For each matched behavior, cache full and sham residual states at:

- the primary turn-2 assistant boundary before the first generated token;
- four private, mechanically derived scaffold-span landmarks at 25%, 50%, 75%,
  and the final scaffold token;
- early generated indices 0, 1, 2, 4, and 8, using a shared teacher-forced full
  baseline prefix in both donor and recipient before free generation resumes.

Coarse residual-post sites are transformer blocks 0, 4, 8, 12, 16, 20, 24, 28,
and 31. A discovery patch window is eligible only if sham-into-full reduces the
harmful-compliance score with at least 70% behavior-ID sign concordance, the
paired 95% cluster-bootstrap interval excludes zero, the reciprocal
full-into-sham effect has the opposite sign, and both no-op and identity-patch
effects are within 25% of the primary absolute mean effect. The seeded
magnitude-matched random control and the irrelevant-site control must each be
within 50% of the primary absolute mean effect.

Among eligible sites, discovery selects the largest absolute standardized
paired behavioral effect, breaking ties by earlier layer and then earlier
position. Calibration must reproduce its direction and an interval excluding
zero. Otherwise B2 stops and no confirmatory patch outcome is generated.

### B2: component localization

Only the single eligible layer-position window frozen after calibration is
tested. Attention output and MLP output are patched separately using the same
donor/recipient alignment and controls. Discovery chooses at most one component
by the B1 eligibility rule; calibration must reproduce it before a single
confirmatory test. Head-level localization is not part of this freeze: it
requires a later prospective amendment, its own tested implementation, and a
new cost statement.

### Controls

Every control traverses the identical hook and scoring pipeline:

- full-to-full identity patch;
- sham-to-sham identity patch;
- explicit no-op hook;
- seeded same-site magnitude-matched random-direction delta;
- irrelevant layer and irrelevant token-position controls;
- reciprocal full-into-sham patch;
- a deterministic cross-behavior donor, matched by category and nearest token
  length, with cyclic hash-order tie breaking.

### Estimands and interpretation

The primary causal estimand is the behavior-ID-clustered paired change in the
frozen harmful-compliance/refusal score for sham-into-full patching.

Secondary estimands are the change in the frozen SAE feature/subspace score and
the frozen Jacobian-lens refusal-minus-compliance margin. Each result is assigned
one of four predeclared descriptions:

1. behavior changes and the feature is suppressed;
2. behavior changes without feature suppression;
3. feature is suppressed without behavioral change;
4. neither changes.

Case 3 supports “biomarker, not localized control point.” Case 1 at an upstream
site supports “candidate causal ancestor.” Neither supports complete mediation,
a universal jailbreak circuit, or a production defense.

Each patch receipt binds the actual realized donor, recipient, and delta tensor
hashes. Private replay bundles retain the BF16 pre-patch recipient and realized
delta tensors plus exact model/input/site identities; public receipts expose
only hashes, norms, shapes, dtypes, and derived scores.

## Stage C: detector robustness

The frozen Llama 3.3 feature-10146 detector and any frozen 8B single-feature and
subspace detectors are evaluated on:

- untouched confirmatory harmful behaviors;
- benign structured prompts;
- alternate conversation positions;
- length- and formatting-matched controls;
- private landmark ablations and replacements;
- mechanically generated paraphrase families;
- adaptive-stress variants that preserve the behavioral endpoint while trying
  to reduce the detector score.

Adaptive stress is exploratory, uses only the 20 stress IDs, has a frozen
maximum of 12 transformations per ID, and never changes the detector or
threshold. Report held-out recall, false-positive rate, precision at observed
prevalence, precision transported to 1%, 5%, and 10% attack prevalence, AUROC
and precision-recall curves, with behavior-ID cluster uncertainty. The benign
panel uses all 100 pinned JBB benign behaviors, split category-balancely into
50 calibration and 50 confirmation IDs. Even zero false positives in 50 cases
does not establish production-grade specificity; report the exact interval.
Compare the single feature, at-most-eight-feature subspace, and a simple private
lexical baseline. Do not publish landmark strings or detector thresholds
intended for deployment.

## Stage D: small-Qwen Jacobian-only pilot

The primary candidate is `Qwen/Qwen3.5-4B` at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, paired to the corrected
Neuronpedia n=1000 lens at revision
`b62c39069a0740aebcc70462231b68612cae367f`, artifact SHA-256
`1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e`.

Because this is a conditional-generation architecture, a local/config smoke and
one outcome-free GPU compatibility fixture must verify the exact text backbone,
layer mapping, hidden width, output head, Jacobian shapes, and hook semantics.
Failure stops the Qwen arm; there is no silent model substitution.

Until the human supplies a Qwen-specific artifact, this is explicitly a
cross-model-transfer test of the pinned Meta scaffold, not a native Qwen
jailbreak study. The behavioral viability gate precedes mechanistic
interpretation. If the scaffold does not change behavior, the result is
“transfer not demonstrated,” not “Qwen lacks the mechanism.”

Measurements are full-versus-sham J-lens trajectories, an ordinary
hidden-state mean-difference/linear-projection baseline, identity transport,
seeded Frobenius-matched random transport, and cross-layer/position
consistency. Dense Qwen directions are compared only by functional signature,
not equated with SAE feature 10146.

Qwen 3.5 397B remains outside this protocol and requires separate authorization.

## Stage E: circuit-breaker prototype

This stage begins only after detector thresholds are frozen. It runs in shadow
mode and scores the completed prefill before the first output token is released.
It records whether the request would have been blocked, cancels the request
rather than terminating the server, and never returns internal feature values
or thresholds to a client.

The comparison includes the single feature, frozen subspace, and private lexical
baseline. Primary deployment metrics are held-out recall, false-positive rate,
precision at observed prevalence, decision latency, added prefill latency, and
utility impact. This is an offline prototype, not a production-readiness claim.

## Stage gates and stop rules

1. **G0 — local integrity:** plan validator, schemas, synthetic/tiny-model hook
   tests, tensor replay, forced-kill resume, and independent review pass.
2. **G1 — one-unit qualification:** exactly one A40 pod, one synthetic-safe
   behavior per required pipeline, no target outcome inspection, exact
   throughput and memory receipt. Failure tears down the pod.
3. **G2 — 8B behavioral viability:** full versus sham must have the frozen
   directional behavioral effect in discovery and calibration. If not, report a
   failed 8B transfer and do not claim mechanistic absence.
4. **G3 — 8B sparse/J-lens replication:** freeze feature/subspace and trajectory
   statistics using discovery/calibration only.
5. **G4 — B1 causal localization:** apply the eligibility rule above. If none
   qualifies, stop B2 while retaining the detector and Qwen arms.
6. **G5 — B2 component localization:** only the one frozen component may reach
   confirmation.
7. **G6 — detector confirmation and shadow mode:** thresholds are immutable
   before held-out outcomes.
8. **G7 — Qwen pilot:** only after the cheaper 8B stages and compatibility gate.

No result-dependent retries, extra layers, alternative model, new prompt
variant, threshold change, or head search are allowed without a timestamped
prospective amendment stating which outcomes were already inspected.

## Figures and provenance

Planned empirical figures:

- layer × position full-minus-sham map;
- SAE feature/subspace discovery and held-out replication;
- B1 patching heatmap with controls;
- feature/behavior four-way dissociation plot;
- detector ROC and precision-recall panels with utility false positives;
- paraphrase, landmark, position, and adaptive-stress robustness;
- Qwen versus Llama functional trajectory comparison;
- shadow-mode latency and block/no-block outcome panel.

Every plotted number must be generated from audited receipts. Each SVG, PNG,
and PDF receives a sidecar with source receipt hashes, analysis code hash,
source commit, plan hash, parameters, row counts, output hashes, and
deterministic byte-verification status.

## Cost envelope before qualification

The live secure-cloud price snapshot used for planning is one A40 48 GB at
`$0.44/hour`. G1 has a 90-minute wall limit (`$0.66` compute maximum) and a
20-minute no-progress timeout. It may not fall back to a more expensive GPU
without a revised cost statement.

Scientific-run costs remain estimates until G1 measures unit throughput:

| Work package | Base | Slow case |
|---|---:|---:|
| Llama 3.1 8B replication | $3 | $8 |
| 8B causal patching, including controls | $6 | $12 |
| Detector robustness, including one bounded 70B pass | $20 | $34 |
| Small-Qwen pilot | $2 | $4 |
| Review, setup margin, and five more volume-days | $8 | $9 |
| **Incremental total** | **$39** | **$67** |

The requested campaign envelope is therefore **$50 incremental soft** and
**$75 incremental hard**, with each pod still requiring its own exact
post-benchmark cost statement. Against the reconciled prior estimate of
`$69.3810094824`, this projects about `$108.38` base and `$136.38` slow case,
below the existing `$200` cumulative hard ceiling. The retained 500 GB volume
is approximately `$1.17/day`; its retention is reviewed daily and is never
confused with pod lifecycle.
