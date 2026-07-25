# Adjudication of the external proposal review

Date: 2026-07-25

Source: `~/Downloads/deep-research-report.md`

SHA-256:
`9f9a2ce726f79e6bc878bdbe2c4511225ca8c5e889a28e6fe0736ada340bf833`

Disposition: **directionally useful, not authoritative**.

The review correctly identifies missing design elements, but it silently
reframes a focused mechanism study as a three-year PhD dissertation and then
criticizes the proposal for not already being that dissertation. We adopt the
construct-validity and evaluation corrections without accepting the scope
expansion as this repository's first study.

## Accepted

### Write an explicit threat model

Agreed. The proposal implied white-box researcher access and text-only attacker
access but did not state them cleanly. The charter now separates attacker
capabilities from researcher instrumentation and excludes tools, retrieval,
weights, and infrastructure compromise.

### Define the attack construct operationally

Agreed with narrowing. The review's proposed definition is useful as a broad
program umbrella. This study uses the narrower term **structured lexical
scaffold attack** for role framing, semantic inversion, response-format
prescription, and a fixed follow-up.

### Use independent prompts, not seeds, as the inferential unit

Strongly agreed. Repeated decodes from one prompt characterize stochasticity;
they do not create independent evidence. Base behavior ID is now the unit for
splits, bootstrap intervals, and power calculations.

### Replace the single phishing probe with a panel

Agreed. The single OpenRouter probe was an existence check. Discovery,
confirmatory, utility, and transfer sets will be disjoint and will span
predeclared behavior categories.

### Measure robustness, overrefusal, utility, and systems cost

Agreed where an intervention is proposed as a defense candidate. A safety
effect that destroys benign usefulness is not a defense. The causal stage now
includes benign instruction following, overrefusal, degeneration, latency, and
memory outcomes.

### Pin evaluation and runtime choices

Agreed. Model revisions, chat templates, decoding parameters, judges,
thresholds, attack budgets, random seeds, and failure rules belong in the
machine plan and receipts.

### Use paired statistics and respect clustering

Agreed. The primary analysis is paired by base behavior, and no token, turn,
position, paraphrase, seed, or feature is promoted to an independent sample.

### Treat high-transfer attack artifacts cautiously

Agreed. Public releases should contain IDs, hashes, schemas, controls, and safe
transform descriptions when raw strings would materially increase misuse.
High-severity findings receive a responsible-disclosure decision before public
release.

### Keep the mechanistic contribution narrow

Agreed. Even a positive result would explain a subset of format-bifurcating
scaffolds on one pinned model, not jailbreaks generally.

## Accepted with modification

### Add taxonomy, defense, and mechanism tracks

Useful as a long-range research program, but not as parallel obligations now.
This repository is Track M, the mechanism study. A taxonomy benchmark or
defense-stack comparison begins only after the focal effect and causal result
justify it.

### Evaluate multiple models and tokenizers

Necessary before a broad defense claim, unnecessary for the first
model-specific mechanism claim. Cross-model and cross-tokenizer transfer is a
sequel gate. Adding weakly aligned GPT-J now would consume effort without
clarifying refusal-to-compliance dynamics in the chosen aligned checkpoint.

### Use standardized jailbreak and utility benchmarks

Agreed in principle, subject to live verification of licenses, evaluator
validity, construct fit, and release policy. Benchmark names in the review are
a literature shortlist, not an automatically adopted stack. A condition-masked
human validation subset remains necessary because evaluator scores are not
ground truth.

### Study defenses as a stack

Agreed for eventual deployment research. It does not answer the current causal
question. Surface detectors, smoothing, prompt separation, and output guards
will be cheapest-prior baselines in a later defense study, not confounds mixed
into the white-box mechanism experiment.

### Center interpretability only in the mechanism paper

Agreed, and this repository is precisely that mechanism paper. Interpretability
therefore remains central here while staying subordinate to behavioral
evidence and causal controls.

### Test both directions of SAE steering

Agreed, with a split-sample correction. Feature, polarity, and magnitude can be
selected only on discovery data. Bidirectionality is then tested on held-out
base behaviors with zero, matched-feature, and isotropic controls.

## Rejected or deferred

### Expand immediately into a dissertation-scale benchmark

Rejected for this study. Breadth before the raw-checkpoint behavioral effect is
known would multiply models, attack families, judges, and defenses around an
unverified premise. A sharp null on the exact checkpoint would make much of
that work irrelevant to the proposed mechanism.

### Treat all text-mediated attacks as one lexical class

Rejected as too broad for causal interpretation. Optimized suffixes,
imperceptible Unicode perturbations, role-play, multi-turn persuasion, indirect
injection, and format bifurcation can share a text channel while relying on
different mechanisms. The umbrella may organize future taxonomy work, but it
cannot define a homogeneous treatment here.

### Include GPT-J as a required legacy baseline

Deferred. A model without the focal modern safety/refusal regime is not the
cheapest serious baseline for a claim about how an aligned model transitions
from refusal to compliance. It may be useful for detector false-positive or
cross-tokenizer work later.

### Implement three defense stacks now

Rejected as premature. The present intervention must first demonstrate
held-out causality and benign-utility preservation. Production stacks answer a
different systems question and require adaptive threat models of their own.

### Adopt the 5-20 TB storage estimate

Rejected as unsupported. Storage will be derived from a measured unit, planned
tensor retention, shard count, compression, and restart policy before paid
compute. Full residual dumps are not retained by default.

### Treat watermarking or media provenance as part of the defense

Rejected for the focal problem. Provenance can help incident response but does
not prevent a prompt from changing the model's behavior.

### Rely on the review's legal timeline

Rejected pending primary-source verification. The review says on 2026-07-25
that a broader EU enforcement phase "began on August 2, 2026," a date still in
the future. Legal claims will not enter the study protocol without current
primary sources and appropriately qualified review.

### Treat the review's citation placeholders as verified references

Rejected. The report contains internal citation handles rather than a
resolvable bibliography. Each consequential source must be opened, assigned
its actual role, and pinned before it warrants a claim in a paper.

## Resulting decision

The first study remains narrow:

1. establish the full-scaffold versus structural-sham behavioral contrast on
   independent held-out behaviors using the exact raw checkpoint;
2. use the Jacobian lens as a controlled descriptive readout, not a causal or
   mental-state claim;
3. select SAE features and steering strengths only on discovery data, then
   test both signs causally on held-out behaviors;
4. require benign-utility preservation and transfer before using the word
   "defense";
5. stop cleanly when a gate fails.

This design incorporates the review's strongest criticism without allowing its
dissertation framing to dissolve the first answerable question.
