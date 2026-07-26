# A Strong Lexical Effect, but No Causal SAE Handle Yet

**Status:** Complete through the frozen Gate 4 discovery calibration. The
held-out confirmatory intervention outcomes were not opened.

> **Draft and AI-use status.** Generative-AI tools helped implement, audit,
> execute, interpret, visualize, and draft this study. The author selected the
> research question and authorized the compute. Before publication, the author
> must inspect the cited artifacts and take responsibility for the final text
> and claims. This is an independent, non-peer-reviewed research draft. Verify
> every number against the released receipts before relying on it.

## Abstract

We tested a structured lexical scaffold against
`meta-llama/Llama-3.3-70B-Instruct`, then followed the behavioral effect into
the model with a fitted Jacobian lens and a public layer-50 sparse autoencoder
(SAE). In a frozen, paired behavioral evaluation over 40 behavior IDs, the
full scaffold raised the mean HarmBench affirmative probability by
`0.6076427093` relative to a token-count-matched structural sham. The
behavior-level bootstrap 95% interval was `[0.4632223418, 0.7472964572]`.
The full scaffold mean was `0.6079533572`; the sham mean was `0.0003106479`.

Discovery analysis found a descriptive Jacobian-lens trajectory and four
layer-50 SAE candidates. Feature `10146` was selected prospectively as the
primary intervention direction. We then evaluated both signs at four safe
doses on 20 discovery behaviors. All runtime safety gates passed, but no dose
met the frozen efficacy rule. The largest mean signed half-span was
`0.0212158025`, with a bootstrap 95% interval of
`[-0.0015528605, 0.0631422285]`, against a required minimum of `0.1`.
The selector therefore stopped with no eligible alpha, and the held-out
confirmatory outcomes remained unopened.

The behavioral vulnerability is well supported on this checkpoint and
protocol. The selected SAE direction is not yet a validated causal mechanism
or defense handle.

## Result in one picture

| Claim rung | Outcome | Evidence level |
|---|---|---|
| The full lexical scaffold changes behavior relative to structural controls | Passed | Confirmatory, paired, receipt-backed |
| The fitted lens and layer-50 SAE expose internal differences associated with the effect | Described | Discovery-only |
| The selected SAE direction causes a practically large bidirectional behavioral change | Did not pass calibration | Prospective discovery calibration |
| The direction supports a held-out defense | Not tested | Confirmatory outcomes unopened |

## Prior work, contribution, and non-claims

The experiment combines a pinned
[Llama 3.3 70B Instruct checkpoint](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct),
the public
[Neuronpedia Jacobian lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/main/llama3.3-70b-it/jlens),
the public
[Goodfire layer-50 SAE](https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50),
and the
[HarmBench classifier](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls).
Our contribution is a staged, receipt-backed test that separates behavioral
confirmation, descriptive localization, discovery-only feature selection, and
causal intervention.

We do not claim that the Jacobian-lens readout is itself a mechanism. We do not
claim that SAE feature IDs name stable human concepts. We do not claim that
feature `10146` is the only relevant direction. We do not claim a working
defense. We also do not generalize from this one checkpoint, scaffold family,
judge, or layer-50 SAE to other models.

## Design

The independent unit was the behavior ID. Raw prompts and model generations
were kept in restricted storage and were not manually inspected during
analysis. Public artifacts contain scores, structural indicators, token IDs
where required by the protocol, hashes, and derived statistics, but not the
restricted text.

The behavioral comparison included four arms:

1. **Base:** the behavior request without the focal scaffold.
2. **Inert length:** a length-matched but semantically inert control.
3. **Structural sham:** the same refusal, divider, and post-divider form, but
   with the post-divider instruction continuing to refuse.
4. **Full scaffold:** the focal structured lexical treatment.

Gate 2 used 40 held-out behavior IDs. Gate 3 used 20 discovery behavior IDs.
Gate 4 used those discovery IDs to choose whether any SAE intervention dose
was eligible for a later held-out test. The complete prospective rules and
amendments are in [STUDY_PLAN.md](STUDY_PLAN.md) and
[docs/AMENDMENTS.md](docs/AMENDMENTS.md).

## Gate 2: the scaffold changed behavior

![Turn-2 score by arm and paired full-minus-sham effect](figures/gate2/E01-full-vs-sham.png)

*Figure E01. Each point is a behavior ID. The left panel places the full
scaffold beside base, inert-length, and structural-sham controls. The right
panel shows the paired full-minus-sham contrast. The diamond is the mean and
the interval is the frozen 10,000-replicate behavior-level bootstrap interval.
Source: [Gate 2 artifact](results/gate2.confirmatory.json). Figure receipt:
[E01 receipt](figures/gate2/E01-full-vs-sham.receipt.json).*

The primary estimand was the mean paired HarmBench affirmative probability,
full scaffold minus structural sham. It was `0.6076427093`, with a bootstrap
95% interval of `[0.4632223418, 0.7472964572]`. The interval excluded zero and
its lower bound exceeded the predeclared continuation threshold. Both the mean
and interval gates passed.

The arm-level results make the control logic visible:

| Arm | Mean affirmative probability | Binary affirmative rate | Refusal rate | Truncation rate |
|---|---:|---:|---:|---:|
| Base | `0.1631202423` | `0.175` | `0.750` | `0.025` |
| Inert length | `0.0036327705` | `0.000` | `0.175` | `0.000` |
| Structural sham | `0.0003106479` | `0.000` | `1.000` | `0.000` |
| Full scaffold | `0.6079533572` | `0.625` | `0.000` | `0.425` |

Length alone did not reproduce the effect, and the structural sham did not
reproduce it even though it generated the divider on every measured turn-2
case. The full scaffold also had a `0.425` truncation rate, so the behavioral
effect should not be read as a pure measure of complete, high-quality
responses. It is a strong change in the frozen classifier score, not a
comprehensive utility or harm assessment.

![Response phases by arm and turn](figures/gate2/E02-response-phases.png)

*Figure E02. Structural indicators are shown separately for turn 1 and turn 2.
Divider production is not sufficient for the behavioral effect: the
structural sham produces the divider and post-divider body but retains
refusal. Source: [Gate 2 artifact](results/gate2.confirmatory.json). Figure
receipt: [E02 receipt](figures/gate2/E02-response-phases.receipt.json).*

## Gate 3: descriptive internal differences

Gate 3 analyzed 132,720 receipt rows across 79 declared source layers. Of 560
scheduled observations, 549 were available and 11 unavailable positions were
recorded explicitly. Exact artifact, plan, layer-topology, and analytic
vocabulary-moment checks passed. A same-checkpoint published fixture was not
available and was not treated as passed.

![Layerwise fitted Jacobian-lens margin](figures/gate3/E03-layerwise-jlens-margin.png)

*Figure E03. Full-minus-sham refusal/compliance margin at the turn-2 assistant
boundary across layers, with identity and deterministic Frobenius-matched
Gaussian controls. This is a descriptive readout, not a causal localization.
Source: [Gate 3 artifact](results/gate3.discovery.json). Figure receipt:
[E03 receipt](figures/gate3/E03-layerwise-jlens-margin.receipt.json).*

The fitted trajectory correlated `0.7893731348` with the identity trajectory
and `0.0460648260` with the matched random trajectory. Its RMSE was
`0.4306417070` from identity and `0.5943266332` from random. Those diagnostics
show structure relative to the declared random control, but they do not show
that any layer or direction causes the behavior.

![Layer and generated-position trajectory](figures/gate3/E04-layer-position-trajectory.png)

*Figure E04. Descriptive full-minus-sham margin across layers and declared
generated-token positions. Missing early positions remain missing rather than
being imputed. Source: [Gate 3 artifact](results/gate3.discovery.json). Figure
receipt: [E04 receipt](figures/gate3/E04-layer-position-trajectory.receipt.json).*

At layer 50, the discovery rule selected SAE features `10146`, `44802`,
`4057`, and `3907`, with feature `10146` primary. Their paired standardized
full-minus-sham deltas were `0.9928309748`, `0.9847804216`,
`0.9791809512`, and `0.9768136882`. Each activated on all full-arm discovery
observations and on none of the sham observations. The matched negative-control
features were `26453`, `9105`, and `40804`.

![SAE candidate and control map](figures/gate3/E05-sae-candidate-map.png)

*Figure E05. Discovery-selected layer-50 SAE candidates beside matched
controls. Selection used only discovery behaviors and was frozen before the
intervention run. Source: [Gate 3 artifact](results/gate3.discovery.json).
Figure receipt: [E05 receipt](figures/gate3/E05-sae-candidate-map.receipt.json).*

The SAE reconstruction relative error was substantial: mean `0.5983428955`
and maximum `0.6649254560`. That fact further limits how literally the sparse
features should be interpreted. The candidates were useful intervention
hypotheses, not decoded ground truth.

## Gate 4: the frozen causal selector stopped

Feature `10146` was tested in both signs at relative doses
\(\rho \in \{0.0025, 0.005, 0.01, 0.02\}\), plus zero intervention. The run
produced 180 of 180 expected receipts over 20 behaviors and nine conditions.
There were no runtime errors, parse failures, or clipping events. The maximum
requested-versus-realized relative error was `0.0703088221`, below its `0.1`
limit. The maximum intervention-delta to pre-residual-norm ratio was
`0.0412302759`, below its `0.05` limit.

The frozen selector required all of the following:

- mean signed half-span at least `0.1`;
- bootstrap 95% lower bound above zero;
- restoring-minus-zero at most `-0.1`;
- opposite-minus-zero at least zero;
- all runtime safety gates passed.

No candidate met the efficacy conditions:

| Relative dose \(\rho\) | Signed half-span | Bootstrap 95% interval | Restoring minus zero | Opposite minus zero |
|---:|---:|---:|---:|---:|
| `0.0025` | `0.0061752877` | `[0.0001528791, 0.0155376166]` | `-0.0208433151` | `-0.0084927398` |
| `0.005` | `0.0132400006` | `[-0.0001225952, 0.0342718131]` | `-0.0140141914` | `0.0124658098` |
| `0.01` | `0.0039501415` | `[-0.0113142787, 0.0236661727]` | `0.0122721124` | `0.0201723954` |
| `0.02` | `0.0212158025` | `[-0.0015528605, 0.0631422285]` | `-0.0232621043` | `0.0191695007` |

![Gate 4 discovery calibration stop](figures/gate4/E05a-discovery-calibration-stop.png)

*Figure E05a. Left: signed half-span intervals against the frozen `0.1`
threshold. Middle: the two directional components required by the selector.
Right: runtime safety diagnostics. Safety passed; efficacy did not. The
selector stopped and did not open held-out outcomes. Source:
[Gate 4 artifact](results/gate4.calibration.discovery.json). Figure receipt:
[E05a receipt](figures/gate4/E05a-discovery-calibration-stop.receipt.json).*

This result does not distinguish among several possibilities: feature `10146`
may be associated rather than causal; the intervention semantics may be too
weak or too local; another feature, layer, position, subspace, or nonlinear
operation may matter; or the behavior may be distributed in a way this SAE
does not capture. It does establish that the prospectively selected direction,
under the tested safe dose ladder, did not earn a held-out defense test.

## Interpretation

The cleanest conclusion is asymmetric. We have strong evidence that the full
lexical scaffold changes the pinned model's behavior relative to controls. We
also have discovery evidence that its internal trajectory differs and that a
small set of SAE features separates full from sham examples at layer 50. We do
not have evidence that the primary feature direction is a practically useful
causal handle.

The stop is informative. It prevents a vivid discovery correlation from being
promoted into a mechanism claim merely because a feature separates conditions.
The protocol did what it was designed to do: it let a promising candidate fail
before the held-out confirmatory set could be used to rescue or tune it.

## Discussion

Everything in this section is interpretation. The audited results above stand
on their own, and each hypothesis below needs a separately frozen test.

One plausible hypothesis is that the selected SAE feature is a readout of a
larger state transition rather than a control knob for it. The full scaffold
and sham differ across many tokens and instructions, while the intervention
adds one normalized decoder direction at one layer and one position. A feature
can therefore classify the two states almost perfectly on discovery examples
without spanning the causal subspace that moves generation. A prospective
follow-up could compare a small, regularized subspace built from multiple
discovery features against matched random and frequency/norm-matched SAE
subspaces.

A second hypothesis is positional: the decisive computation may occur across
multiple prompt tokens or generation steps. The current intervention targets
the turn-2 assistant boundary. Testing a position sweep or repeated injection
would be a new protocol, not a reinterpretation of the present null. It should
freeze positions, multiplicity, dose normalization, utility checks, and
stopping rules before any held-out score is opened.

A third hypothesis is that first-order or sparse-linear instruments are
descriptive in this regime because the scaffold induces a nonlinear,
distributed change. That would motivate controlled comparisons among direct
residual additions, SAE activation interventions, local Jacobian-informed
directions, and low-rank subspaces. The key test is not whether an instrument
produces any change, but whether a direction selected without held-out access
moves harmful behavior in the desired direction while preserving benign
utility and avoiding overrefusal.

The most defensible near-term defense work is therefore diagnostic rather than
deployment-ready. A detector based on feature `10146` could still be explored,
but its discovery-perfect separation is not a held-out detector result, and a
detector alone does not explain or repair the vulnerability.

## Reproducibility and provenance ledger

| Surface | Artifact | SHA-256 or source binding |
|---|---|---|
| Public study plan | [plans/study_v1.public.json](plans/study_v1.public.json) | `a2ed9a0542a6953dbbfd775064366e7b88a07a8f9347eb96679b0ba77300a24e` |
| Gate 2 result | [results/gate2.confirmatory.json](results/gate2.confirmatory.json) | `9b2408742f58875213781138ef6f51564f8f6bbdfd89ff539153de44e345aec5` |
| Gate 2 source commit | artifact field | `9fe4935a2c95eed404ae567922160b290a201810` |
| Gate 2 figures | [figures/gate2/provenance.json](figures/gate2/provenance.json) | `589fb5d321d8160b203b51a335932d7d080ec473955b8ede48d0e607d9f9dee0` |
| Gate 3 result | [results/gate3.discovery.json](results/gate3.discovery.json) | `a6e6637b55c5a4869cbe9e8979a80c6d95506a5c8397973497e14a8b17a10b39` |
| Gate 3 run commit | artifact field | `be65d606e84465d55bb0d60f3ba73f31321dc47d` |
| Gate 3 figures | [figures/gate3/provenance.json](figures/gate3/provenance.json) | Per-output hashes and source pointers inside |
| Gate 4 intervention plan | [plans/gate4_intervention_v1.public.json](plans/gate4_intervention_v1.public.json) | `0e8e47a6569de4d2e03ac5d53b54c7c65c893f0c411a61ae48637642918b0047` |
| Gate 4 result | [results/gate4.calibration.discovery.json](results/gate4.calibration.discovery.json) | `e09c6c771f4fb2b313f7b6dcd31e8e657f1ce8ecec950814bbdc843b58f2a1f5` |
| Gate 4 run and analysis commit | artifact field | `fe82c68cbaa91d3e8b858866f24b40a8d88f1ebe` |
| Gate 4 generation receipt aggregate | artifact field | `a0dfdbc2bbb6283c5d7e41fb7f6ecfa6af5ee0d5c50ce883f630d8bd8a81aad5` |
| Gate 4 score receipt aggregate | artifact field | `a09d306f4f643bf502fcae9e371ea0911624287e17abe7d9d87d6db00b30ca75` |
| Gate 4 analysis rows | artifact field | `fdb1d70c559b8f648e63229bd26b7d74fe9ec056b2d930174a4aecc1bf90cd98` |
| Gate 4 figure | [figures/gate4/provenance.json](figures/gate4/provenance.json) | `9f3d61afce4e81e35a1d3a3ea3a34f0e275497f187d1516b2131d94dc225adab` |
| Compute reconciliation | [results/compute-reconciliation.json](results/compute-reconciliation.json) | Exact task pod billing plus derived task-volume accrual |

Private, non-raw checkpoint bundles are retained locally for receipt audit:

- Gate 2: SHA-256
  `bf9410bf11f268e3832a866d11c78570d82f0817d8eb48fa4811f218b27b1d02`
- Gate 3: SHA-256
  `16271bebbedc8952c731a28dc6e78cdcf3263c060bda8f5b59feafd9aed6dd7e`
- Gate 4: SHA-256
  `d5075a3653e0169293228fbbe6b62c92b4eeae07ac8c212e0cc2ae8c9511a5b5`

All six empirical figures ship as SVG, PNG, and PDF with individual receipt
files. The figure verifiers regenerate them into temporary directories and
require byte-identical outputs.

## Compute and storage

A final RunPod billing query grouped by pod ID attributes
`$68.7766788497` to the four task-owned pods over `30,992,179` billed
milliseconds. This is below the `$100` soft gate and `$200` hard ceiling. No
task-owned GPU pod remains active, so GPU billing is `$0/hour`.

The persistent 500 GB RunPod network volume remains provisioned at
`$35/month`, or about `$1.17/day`, to preserve the pinned model and resumable
artifacts during the short reporting window. From its creation through the
final query, its rate-derived task accrual is approximately `$0.6043`, for an
estimated task infrastructure total of `$69.3810`. The storage amount is
derived because RunPod's network-volume billing endpoint returned account-wide
buckets without volume IDs. The exact method, task pod rows, private source
receipt hash, and retention review date are recorded in
[results/compute-reconciliation.json](results/compute-reconciliation.json).

## Appendix: release inventory

| What is released | In plain English | Primary use |
|---|---|---|
| Public plans and amendments | The rules frozen before each stage | Audit researcher degrees of freedom |
| Gate 2 result JSON | Behavioral scores, controls, and paired uncertainty | Verify the confirmatory effect |
| Gate 3 result JSON | Layerwise lens and SAE discovery summaries | Reproduce descriptive plots and candidate selection |
| Gate 4 result JSON | Dose selection, efficacy, safety, and stop state | Verify that no alpha qualified |
| Figure receipts and provenance | Exact inputs, output hashes, and plot bindings | Verify every plotted mark |
| Schemas and analysis code | Machine-checkable receipt contracts and deterministic analysis | Re-run local validation |

Restricted prompt and generation text is intentionally excluded from the
public release. This preserves the raw-outcome restriction and avoids
publishing operational attack content that is unnecessary for checking the
reported statistics.
