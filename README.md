# Lexical Prompt Study

This repository contains a staged white-box study of how a structured lexical
scaffold changes refusal and compliance behavior in
`meta-llama/Llama-3.3-70B-Instruct`, whether a fitted Jacobian lens localizes
that change, and whether a public layer-50 sparse autoencoder (SAE) supports a
causal intervention. A controlled follow-up replicates the behavioral and
internal readouts on `meta-llama/Llama-3.1-8B-Instruct`, with the lexical
scaffold placed both before and after the harmful request and never pooled.

The governing design is [STUDY_PLAN.md](STUDY_PLAN.md). The external proposal
review and our point-by-point adjudication are recorded in
[docs/REVIEW_ADJUDICATION.md](docs/REVIEW_ADJUDICATION.md).

## Current status

**Study v1 reached its prospectively frozen Gate-4 stop rule.** Gate 2
confirmed a large behavioral scaffold effect, Gate 3 produced a receipt-backed
Jacobian-lens map and discovery-only SAE candidates, and Gate 4 found no
eligible causal intervention strength within the frozen efficacy and safety
rules. Held-out intervention outcomes were never opened.

The integrated, figure-led account is
[RESEARCH_REPORT.md](RESEARCH_REPORT.md). It separates the confirmed
behavioral result, discovery-only internal readouts, and the Gate-4 causal
calibration stop.

The public plan, restricted companion plan, artifact inventory, category-balanced
splits, four tokenizer-matched arms, evaluator, thresholds, probe sets, decoding,
resume rules, and compute policy were fixed before their affected outcomes.
Implementation and numerical-semantics corrections are recorded in
[docs/AMENDMENTS.md](docs/AMENDMENTS.md).

The Gate-4 discovery calibration evaluated 20 behaviors at zero and both signs
of four residual-norm-scaled doses. All 180 intervention receipts passed the
frozen runtime gates, but no dose met the frozen behavioral efficacy criteria.
The largest signed half-span was `0.0212` (95% bootstrap interval
`[-0.0016, 0.0631]`) against a required `0.1`; therefore the confirmatory
intervention panel was not run. The non-raw receipt-backed artifact is
[results/gate4.calibration.discovery.json](results/gate4.calibration.discovery.json);
the byte-verified stop-gate figure is
[figures/gate4/E05a-discovery-calibration-stop.svg](figures/gate4/E05a-discovery-calibration-stop.svg).

The Llama 3.1 8B follow-up passed its ordering-specific behavioral discovery
and calibration screens. A discovery-frozen SAE rule selected feature `6779`
on calibration, with full prevalence `20/20` in both orderings and partitions,
base/inert-length prevalence `0/20`, and order-dependent structural-sham
activation. The detector threshold is intentionally unfitted because the
ordinary-benign and structured-benign calibration negatives have not yet
been generated. The result is therefore an internal replication candidate,
not a detector, circuit breaker, causal mechanism, or defense. See
[results/g3.followup-mechanism.public.json](results/g3.followup-mechanism.public.json)
and the three byte-verified figures under
[figures/followup-g3](figures/followup-g3).

The follow-up coarse causal-localization arm also completed. It scored all
`1,800` predeclared residual-post patch continuations across both scaffold
placements, five instrument-valid layers, nine conditions, and 20 discovery
behaviors. No layer passed both restoring and reciprocal effect gates, so the
eligible set is empty and calibration remains closed. The result does not rule
out finer tokenwise, component, feature, or subspace circuits. See
[results/g4.followup-patch-discovery.public.json](results/g4.followup-patch-discovery.public.json)
and the byte-verified
[E12 causal-stop figure](figures/followup-g4/E12-followup-causal-localization-stop.svg).

Prior OpenRouter observations described in the source proposal are exploratory
and are not evidence about the raw checkpoint used by this study.

## Repository rules

- No paid GPU is launched without a timed cheap-path test and explicit approval
  of hourly rate, wall-time ceiling, and spend ceiling.
- No raw high-risk attack prompts, credentials, model weights, or unrestricted
  generations are committed.
- Every empirical figure is generated from immutable receipts, has a
  self-contained per-figure receipt, and is indexed by a post-wide provenance
  manifest that can re-derive every plotted value.
- Confirmatory outcomes are never used to select features, steering strengths,
  probe lexicons, thresholds, or analysis choices.
- Every substantive plan change is recorded in
  [docs/AMENDMENTS.md](docs/AMENDMENTS.md) before the affected outcome is
  inspected.
- A negative behavioral gate, failed lens calibration, or null steering result
  is a valid result. No stage is forced to continue.

The planned visual evidence surfaces and the raw fields they require are fixed
in [docs/FIGURE_CONTRACT.md](docs/FIGURE_CONTRACT.md). Compute authorization,
checkpointing, and persistent-volume rules are in
[docs/COMPUTE_POLICY.md](docs/COMPUTE_POLICY.md).

## Pinned public artifacts

The complete file-level hashes and licenses are in
[plans/artifacts.v1.json](plans/artifacts.v1.json). Principal revisions are:

- [Meta Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct),
  revision `6f6073b423013f6a7d4d9f39144961bfbfbc386b`;
- [Neuronpedia Jacobian lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/main/llama3.3-70b-it/jlens),
  revision `a4114d7752d11eb546e6cf372213d7e75526d3a1`;
- [Goodfire layer-50 SAE](https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50),
  revision `128ee921ecd1b8b3a87d776cbcc357c0855da134`;
- [HarmBench Llama-2 13B classifier](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls),
  revision `bda705349d1144fa618770bea64d99ce54e3835b`.

The machine-readable public plan has SHA-256
`a2ed9a0542a6953dbbfd775064366e7b88a07a8f9347eb96679b0ba77300a24e`;
the artifact manifest has SHA-256
`7da57a4047175fc2896a623dbb87566ec6abad394ce965026567efcb73946308`.

## Local verification

Use the frozen uv environment:

```bash
uv sync --frozen --extra dev
uv run lexical-study validate-plan \
  --public plans/study_v1.public.json \
  --private private/plans/study_v1.private.json \
  --artifacts plans/artifacts.v1.json
uv run ruff check .
uv run pytest -q
```

The private plan is intentionally ignored and is required only for authorized
execution. Public plan validation, schemas, analysis code, and tests remain
reviewable without publishing attack strings or unrestricted generations.
