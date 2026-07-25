# Lexical Prompt Study

This repository contains a staged white-box study of how a structured lexical
scaffold changes refusal and compliance behavior in
`meta-llama/Llama-3.3-70B-Instruct`, whether a fitted Jacobian lens localizes
that change, and whether a public layer-50 sparse autoencoder (SAE) supports a
causal intervention.

The governing design is [STUDY_PLAN.md](STUDY_PLAN.md). The external proposal
review and our point-by-point adjudication are recorded in
[docs/REVIEW_ADJUDICATION.md](docs/REVIEW_ADJUDICATION.md).

## Current status

**Locked study charter, outcome-free for the raw checkpoint, not yet a
confirmatory freeze.**

The first commit fixes the claim boundary, threat model, stage gates,
independent unit, controls, primary estimands, safety boundary, and change
process. It does not claim that the final machine plan, prompt corpus, artifact
revisions, or compute budget have been frozen. Those require implementation,
outcome-masked validation, an adversarial plan review, and an explicit costed
go-ahead.

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

## Public artifact candidates

These are feasibility anchors, not frozen revisions:

- [Meta Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [Neuronpedia Jacobian lens for Llama 3.3 70B Instruct](https://huggingface.co/neuronpedia/jacobian-lens/tree/main/llama3.3-70b-it/jlens)
- [Goodfire Llama 3.3 70B layer-50 SAE](https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50)

The model revision observed during planning was
`6f6073b423013f6a7d4d9f39144961bfbfbc386b`. All three artifacts must be
resolved to immutable revisions and verified as mutually compatible before a
paid run.
