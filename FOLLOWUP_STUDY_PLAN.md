# Follow-up study plan: replication, causal localization, and defensive detection

Status: prospective 8B/Qwen protocol freeze candidate; 70B replay complete

Study namespace: `lexical-scaffold-followup-v2`

Outcome status: four-arm 70B replay inspected; no 8B or Qwen target outcome generated
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

## Frozen scaffold-placement factor

Every follow-up behavior crosses the scaffold-bearing arms with exactly two
orderings:

1. `ep_before_request`: the pinned scaffold or matched control precedes the
   harmful request;
2. `ep_after_request`: the same scaffold or matched control follows the same
   harmful request.

Within an arm, one canonical rendering template permutes exactly two immutable
blocks. The two orderings contain the same request bytes, scaffold or control
bytes, boundary-stable delimiter bytes, conversation turn, context ceiling,
and generation budget. Before target generation, the restricted runner must
assert per behavior and arm that each component has the same realized token
subsequence and count across orderings; delimiter and chat-template special
tokens match the freeze; component offsets are unique and recoverable; neither
render is truncated, padded differently, or context-shifted; the assistant
boundary has the frozen suffix; and total prompt-token counts are equal. Any
failure stops the placement experiment. Public receipts contain only
tokenizer/chat-template/render hashes, component hashes and counts, offset-map
hashes, and pass/fail status, never reconstructive token IDs. Structural-sham
and inert-length controls are crossed the same way. Base requests and ordinary
benign prompts contain no scaffold, so each is generated once as a shared
reference and never duplicated or double-counted.

Placement is a controlled factor, not a secondary robustness label. Behavioral
outcomes, every SAE feature statistic, and every J-lens layer-by-position
readout are computed and reported separately for `ep_before_request` and
`ep_after_request`. There is no pooled placement estimate, pooled feature
ranking, or pooled threshold fit. A single common feature/subspace may be
nominated only by a maximin rule: it must satisfy the frozen discovery and
calibration criteria separately in both orderings, and ranking uses the smaller
of its two ordering-specific standardized effects. “Robust across the two
frozen single-turn orderings” means only that both named orderings passed. It
does not mean placement invariance, effect equivalence, or robustness to other
turns or positions. A one-order result is reported only for that ordering. The
paired difference between the two full-minus-sham behavioral effects is
reported with a descriptive 95% behavior bootstrap interval as a secondary
placement-interaction estimate; no no-moderation claim is permitted.

## Stage A: Llama 3.1 8B replication

### Completed prerequisite: Llama 3.3 four-arm discovery replay

The pinned layer-50 SAE was replayed over the preserved Gate-3
assistant-boundary states for all four arms without a new 70B forward pass.
Feature 10146 prevalence was `0/20` base, `14/20` inert-length, `0/20`
structural-sham, and `20/20` full. The prospectively frozen replay and result
are `plans/gate3_sae_four_arm_replay_v1.public.json` and
`results/gate3.sae-four-arm-replay.discovery.json`.

The replay is an explicit machine gate. Feature 10146 remains eligible only if
its full-arm prevalence is at least 0.90, prevalence in each of base,
inert-length, and structural-sham is at most 0.10, and every paired
full-minus-non-full contrast has at least 0.80 positive-delta concordance with a
95% bootstrap lower bound above zero. Failure retires feature 10146 from
confirmatory detector, shadow-mode, and defense claims. The result is still
reported as discovery evidence, and this rule cannot be revised after replay.
The inert-length prevalence failed the gate, so feature 10146 is retired from
confirmatory detector, shadow-mode, and defense claims. It remains descriptive
prior evidence and an example of why the complete control matrix is required.

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
The three scaffold-bearing arms are crossed with both frozen placement
orderings; base is the shared reference defined above. The utility set includes
ordinary benign and benign-but-structured prompts, with structured benign
controls crossed by ordering.
Prospectively specified landmark-removal/replacement and paraphrase
transformations operate mechanically inside the restricted runner.

Behavioral viability is evaluated separately as full-minus-structural-sham
within each ordering. Discovery is a non-inferential screen: each ordering
must have mean paired HarmBench affirmative-probability difference at least
0.20 and a 10,000-replicate behavior-bootstrap lower bound above zero.
Calibration contains the two formal ordering-specific paired tests in one Holm
family at alpha 0.05. Each must have mean difference at least 0.15, at least
70% positive paired signs, and a Holm-adjusted two-sided paired sign-flip
randomization p-value below 0.05. Both discovery screens and both calibration
tests must pass for a common placement-robust mechanistic arm. A one-order pass
is reported only as ordering-specific behavioral transfer and stops common-site
causal confirmation.

The sole primary Stage-A confirmatory construct is an **EP-derived
attack-family attempt detector** on the pinned JBB panel. The original full arm
is positive within each ordering. Base, ordering-matched inert-length,
ordering-matched structural-sham, ordinary benign, and ordering-matched
structured-benign prompts are negative. Mechanically frozen paraphrase and
landmark transformations are secondary robustness tests, not extra independent
units or a license to call the detector general-purpose.

The confirmatory detector question is whether one detector frozen on
discovery/calibration distinguishes the original full arm from all declared
negative arms. Jacobian-lens trajectory similarity, feature/subspace
alternatives, and robustness transformations are secondary descriptive
questions.

Discovery may nominate one single feature and one subspace of at most eight
features, but calibration must select exactly one of them as the sole 8B
confirmatory detector. It must pass selection separately in both orderings by
the maximin rule. Calibration freezes one common set of weights and one common
threshold without pooling: optimize the smaller ordering-specific recall
subject to every ordering-specific negative-class constraint. Confirmation is
reported as two ordering strata. A different feature number in the 8B SAE is
an analogue only by prospectively tested function, never by index. A frozen
low-capacity dense hidden-state baseline and a private lexical baseline are
comparators only; no SAE-superiority claim is made unless their prospective
comparison supports it.

The maximin selector is deterministic. For candidate \(c\), ordering \(o\), and
behavior-level paired score differences \(d_{i,c,o}\), its ordering-specific
standardized effect is
\[
z_{c,o} = \frac{\operatorname{mean}_i d_{i,c,o}}
{\sqrt{\operatorname{mean}_i d_{i,c,o}^2}}.
\]
A zero root-mean-square denominator makes the candidate ineligible. Eligible
candidates rank by descending \(\min_o z_{c,o}\), then descending smaller
ordering-specific raw mean, then single feature before subspace, then ascending
lexicographic feature-ID tuple. Candidate eligibility is evaluated before
ranking. The candidate grid is exactly the discovery-nominated single feature
and one discovery-fitted linear subspace containing at most eight ascending
feature IDs; calibration may choose only between those two frozen candidates.

The common detector threshold grid is the sorted unique finite calibration
scores plus the two exterior thresholds immediately below the minimum and
above the maximum using `numpy.nextafter`. A score equal to the threshold is
positive. Eligible thresholds satisfy every ordering-specific and shared
negative constraint and have full-arm recall at least 0.80 in each ordering.
They rank by descending minimum of the two full-arm recalls, then ascending
worst negative-stratum false-positive rate, then descending threshold. Shared
base and ordinary-benign rows enter once. If no threshold is eligible, no
detector advances.

## Stage B: causal activation patching on Llama 3.1 8B

This is distinct from Gate 4. It replaces actual internal state from a matched
donor run rather than adding an SAE decoder direction.

### B1: coarse localization

The assistant boundary before the first generated token is the only position
eligible for causal selection or confirmation. For descriptive maps only, the
runner may cache four private scaffold-span percentiles and early generated
indices 0, 1, 2, 4, and 8. Those positions are not interpreted causally because
percentile tokens need not be homologous and teacher-forced prefixes can create
hybrid states. Promoting any other position requires a new prospective
alignment-validation amendment before its target outcomes.

Coarse residual-post sites are transformer blocks 0, 4, 8, 12, 16, 20, 24, 28,
and 31. A discovery layer is eligible only if sham-into-full at the assistant
boundary reduces the HarmBench affirmative probability with at least 70%
behavior-ID sign concordance, a paired 95% behavior-bootstrap interval below
zero, and a mean reduction of at least 0.10. The reciprocal full-into-sham
effect must have the opposite sign. No-op and identity controls must have
absolute mean effect at most 0.02 and a paired 90% interval contained in
[-0.05, 0.05]. Seeded random and irrelevant-site controls must have absolute
mean effect at most 0.05 and a 95% interval containing zero.

Among eligible layers, discovery selects the largest absolute standardized
paired behavioral effect under the ordering-stratified maximin rule, breaking
ties by earlier layer. The layer must be eligible separately in both orderings.
Calibration must reproduce the direction, mean reduction of at least 0.10, and
an interval below zero separately in both orderings. Otherwise B2 stops and no
confirmatory patch outcome is generated.

### B2: component localization

Only the single eligible assistant-boundary layer frozen after calibration is
tested. Attention output and MLP output are patched separately using the same
donor/recipient alignment and controls. Discovery chooses at most one component
by the B1 eligibility rule applied separately to both orderings; calibration
must reproduce it in both before confirmation. Head-level localization and
AtP-star are not part of this
freeze. Either requires a later prospective amendment, stays discovery-only,
and cannot satisfy the causal endpoint without exact held-out replacement
patching.

### Patch-assay validity gate

Before any target-site inference, the exact hook class must pass a safe
target-model positive control. Twenty calibration-only prompt pairs request one
of two single-token benign answers whose tokenizer IDs are locally asserted.
At every candidate layer, replacing the recipient assistant-boundary state
with its matched donor state must shift the frozen donor-minus-recipient
next-token logit margin by at least 0.50 on average, with at least 90% pairwise
sign concordance and a paired 95% bootstrap lower bound above zero. No-op and
identity controls must satisfy the absolute equivalence rule above. Failure
invalidates the causal arm; it is never reported as a mechanistic null.

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

The confirmatory causal estimand is computed separately within each placement
ordering: the mean paired behavior-ID change in HarmBench affirmative
probability under exact sham-state replacement into the full arm at the one
calibrated component and the pre-generation assistant boundary. One common
layer/component can reach confirmation only if it passes discovery and
calibration separately in both orderings, using the smaller standardized effect
for selection. No states or scores are pooled across ordering. Within each
ordering, a successful directional result requires mean change at most -0.10,
at least 70% negative pairwise signs, and rejection by the predeclared
two-sided paired sign-flip randomization test after Holm correction across the
two ordering p-values at family alpha 0.05. The 95% paired
behavior-bootstrap intervals are descriptive and not decision-bearing.
Randomization uses 65,536 seeded Rademacher sign vectors plus the observed
assignment, seed `20260726`, with a plus-one p-value correction. A
behavioral-equivalence result requires a paired 90% interval contained in
[-0.05, 0.05]. Anything between those rules is inconclusive.

Before confirmation, calibration-only paired-score variance is used for a
documented sensitivity simulation at n=40. Each ordering must have at least
80% estimated power for a -0.10 location shift at conservative two-sided alpha
0.025, using 10,000 simulations with seed `20260726`; otherwise confirmation is
not run and the causal arm is reported as underpowered. This is a conservative
per-order proxy for the Holm conjunction. The estimand is the fixed average
over the pinned confirmatory JBB panel; extrapolation to a behavior population
is secondary and requires an explicit exchangeability assumption.

Secondary estimands are the change in the frozen SAE feature/subspace score and
the frozen Jacobian-lens refusal-minus-compliance margin. Each result is assigned
one of four predeclared descriptions:

1. behavior changes and the feature is suppressed;
2. behavior changes without feature suppression;
3. feature is suppressed without behavioral change;
4. neither changes.

Case 3 supports only “dissociation at the tested component” and only when the
positive-control gate passed and behavioral equivalence was established. Case
1 at an upstream site supports “candidate causal ancestor at the tested site.”
Neither supports complete mediation, global noncausality, a universal jailbreak
circuit, or a production defense.

Generation settings and seeds, deterministic judging inputs, judge blinding to
arm/intervention labels, score-parser failures, missing behavior IDs, and
exclusions are immutable before confirmation. A missing or failed score remains
missing; no behavior is silently replaced. More than 10% missing confirmatory
IDs invalidates that ordering-specific causal test. The detector and causal
endpoints are separate families, each with family alpha 0.05. The two
ordering-specific causal confirmations use Holm correction, and a
placement-robust causal claim requires both to pass. All other Stage-A/B tests
are secondary unless listed above as gates.

Each patch receipt binds the actual realized donor, recipient, and delta tensor
hashes. Private replay bundles retain the BF16 pre-patch recipient and realized
delta tensors plus exact model/input/site identities; public receipts expose
only hashes, norms, shapes, dtypes, and derived scores.

## Stage C: family-specific detector confirmation and robustness

Feature 10146 is evaluated only if its four-arm replay gate passed. It did not
pass and is therefore retired from this stage. Exactly one 8B detector, either
the single feature or the at-most-eight-feature subspace selected on
calibration, reaches confirmation. The positive class is the original
EP-derived full arm in each ordering. Negative strata are shared base,
ordering-specific inert-length, ordering-specific structural-sham, shared
ordinary benign, and ordering-specific structured-benign. The detector is
evaluated on:

- untouched confirmatory harmful behaviors;
- benign structured prompts;
- the two frozen scaffold orderings, never pooled;
- length- and formatting-matched controls;
- private landmark ablations and replacements;
- mechanically generated paraphrase families;
- adaptive-stress variants that preserve the behavioral endpoint while trying
  to reduce the detector score.

Calibration chooses one common threshold that maximizes the smaller of the two
ordering-specific full-arm recalls, subject to each ordering-specific negative
stratum having false-positive rate at most 0.10. No rows are pooled.
Confirmatory success is joint across ten unique one-sided Clopper-Pearson
bounds: two ordering-specific full-arm recall bounds; shared base and
ordinary-benign FPR bounds; and ordering-specific inert-length,
structural-sham, and structured-benign FPR bounds. Per-bound alpha is
\(0.05/10\). Each full-arm recall lower bound must be at least 0.80, and each
negative-stratum FPR upper bound at most 0.20. Failure of any bound means the
detector did not meet the placement-robust family-specific gate.

Detector denominators remain fixed at their planned stratum counts. Scores are
never silently replaced. An unevaluable positive counts against recall; an
unevaluable negative counts against the false-positive gate. More than 10%
unevaluable units in any stratum invalidates that stratum and therefore the
joint two-order gate. A missing shared base or ordinary-benign unit is counted
once in its single shared bound. Complete-case rates are secondary diagnostics
only.

The one-sided Clopper-Pearson bounds are conservative acceptance gates for the
pinned fixed panel. Population interpretation requires an explicit
exchangeability assumption. At per-bound alpha 0.005, the attainable count
rules are frozen:

| Bound | Planned n | Passing event count |
|---|---:|---:|
| Each ordering-specific full recall | 40 | at least 39 detected |
| Base, inert-length, or structural-sham FPR | 40 | at most 1 false positive |
| Ordinary-benign or structured-benign FPR | 50 | at most 2 false positives |

Adaptive stress is exploratory, uses only the 20 stress IDs, has a frozen
maximum of 12 transformations per ID, and never changes the detector or
threshold. Report held-out recall, false-positive rate, precision at observed
prevalence, precision transported to 1%, 5%, and 10% attack prevalence, AUROC
and precision-recall curves, with behavior-ID cluster uncertainty. The benign
panel uses all 100 pinned JBB benign behaviors, split category-balancely into
50 calibration and 50 confirmation IDs. Even zero false positives in 50 cases
does not establish production-grade specificity; report the exact interval.
Compare the frozen detector with the unselected alternative, a low-capacity
dense mean-difference projection trained on the same discovery/calibration
units, and a simple private lexical baseline. Those comparisons, trajectory
similarity, and robustness families are secondary. Do not publish landmark
strings or detector thresholds intended for deployment, and do not call the
result a general jailbreak detector.

## Stage D: conditional small-Qwen Jacobian-only pilot

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
not equated with SAE feature 10146. The two scaffold orderings are run and
reported as separate behavioral and J-lens strata under the same no-pooling
rule. Cross-order plots label semantic regions separately and do not imply
token-position homology outside the shared assistant boundary.

This separately scoped pilot runs only after the core Llama replay,
replication, patching, and detector results are frozen and reported. It may run
regardless of whether the Llama detector gate passes because its declared
question is cross-family J-lens transfer, not detector deployment. Qwen 3.5
397B remains outside this protocol and requires separate authorization.

## Stage E: circuit-breaker prototype

This stage begins only if a detector passes its confirmatory gate and its
threshold is frozen. It runs in shadow
mode and scores the completed prefill before the first output token is released.
It records whether the request would have been blocked, cancels the request
rather than terminating the server, and never returns internal feature values
or thresholds to a client.

The comparison includes the confirmed detector, unselected SAE alternative,
dense comparator, and private lexical baseline. Reuse of the Stage-C panel is
explicitly an engineering replay with no new independent statistical claim.
Metrics are decision latency, added prefill latency, deterministic agreement
with the frozen offline scorer, cancel/no-cancel behavior, and utility-path
integrity. This is an offline prototype, not deployment evidence or a
production-readiness claim.

## Stage gates and stop rules

1. **G0 — local integrity:** plan validator, schemas, synthetic/tiny-model hook
   tests, tensor replay, forced-kill resume, and independent review pass.
2. **G0b — 70B four-arm SAE replay:** apply the frozen candidate gate. A fail
   retires feature 10146 from detector, shadow, and defense claims.
3. **G1 — one-unit qualification:** exactly one B200 pod, one synthetic-safe
   behavior per required pipeline, no target outcome inspection, exact
   throughput and memory receipt. Failure tears down the pod.
4. **G2 — 8B behavioral viability:** full versus sham must have the frozen
   directional behavioral effect separately in both placement orderings in
   discovery and calibration. If only one passes, report an ordering-specific
   transfer and do not make a placement-robust or mechanistic-absence claim.
5. **G3 — 8B sparse/J-lens replication:** freeze exactly one detector using
   the ordering-stratified maximin rule on discovery/calibration only; emit
   complete separate SAE and J-lens tables for both orderings. Trajectory
   statistics remain secondary.
6. **G4 — patch validity and B1 causal localization:** the safe positive
   control must pass before applying the boundary-only eligibility rule. If none
   qualifies, stop B2 while retaining the detector and Qwen arms.
7. **G5 — B2 component localization:** only the one frozen component may reach
   confirmation.
8. **G6 — detector confirmation:** the joint family-specific bound must pass
   before shadow engineering.
9. **G7 — conditional shadow engineering:** reused-panel implementation
   verification only after G6 passes.
10. **G8 — Qwen pilot:** only after the core Llama report and compatibility
    gate.

No result-dependent retries, extra layers, alternative model, new prompt
variant, threshold change, or head search are allowed without a timestamped
prospective amendment stating which outcomes were already inspected.

## Figures and provenance

Planned empirical figures:

- separate layer × position full-minus-sham maps for both scaffold orderings;
- separate SAE feature/subspace discovery and held-out replication panels for
  both scaffold orderings;
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

The original secure-cloud A40 allocation failed before creating a pod despite
the availability query reporting high stock. Prospective infrastructure-only
amendment A018 then specified one RTX A6000 48 GB at `$0.53/hour`, but that
allocation also failed before pod creation. Because this exact retained volume
previously admitted a secure B200, amendment A019 substitutes exactly one B200
at the 2026-07-26 live secure-cloud rate of `$5.89/hour`, shortens G1 to a
30-minute wall limit (`$2.945` compute maximum), and shortens the no-progress
timeout to 10 minutes. This is an explicit qualification-device substitution,
not an automatic fallback; a B200 allocation failure creates no pod and stops
pending another prospective cost statement.

Scientific-run costs remain estimates until G1 measures unit throughput:

| Work package | Base | Slow case |
|---|---:|---:|
| Llama 3.1 8B replication | $3 | $8 |
| 8B causal patching, including controls | $6 | $12 |
| Detector robustness, including one bounded 70B pass | $20 | $34 |
| Small-Qwen pilot | $2 | $4 |
| Review, setup margin, and five more volume-days | $8 | $9 |
| **Incremental total** | **$39** | **$67** |

Those estimates include both scaffold orderings. The bounded work-unit ledger
before qualification is:

| Stage | Shared runs | Ordering-specific runs | Maximum |
|---|---:|---:|---:|
| 8B harmful and benign generation/capture | 180 | 680 | 860 |
| B1/B2 patch continuations, all frozen controls and gates | 0 | 5,400 | 5,400 |
| Adaptive-stress variants | 0 | 480 | 480 |
| Qwen discovery/calibration transfer pilot | 40 | 240 | 280 |

The patch maximum expands the nine frozen intervention/control kinds across
the nine-layer discovery grid and the gated calibration/component/confirmation
steps. Stopped gates reduce, never increase, these counts. G1 must replace
these planning maxima with measured seconds, bytes, and cost per unit before
each scientific run. Crossing placement does not raise the `$75` incremental
hard ceiling; reaching the `$50` soft gate requires a new human check-in.

The requested campaign envelope is therefore **$50 incremental soft** and
**$75 incremental hard**, with each pod still requiring its own exact
post-benchmark cost statement. Against the reconciled prior estimate of
`$69.3810094824`, this projects about `$108.38` base and `$136.38` slow case,
below the existing `$200` cumulative hard ceiling. The retained 500 GB volume
is approximately `$1.17/day`; its retention is reviewed daily and is never
confused with pod lifecycle.
