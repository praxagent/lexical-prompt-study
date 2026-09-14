# A149: corrected-outcome sensitivity of the frozen A145 decisions

Status: exploratory correction analysis; locally frozen before this analysis's
first aggregate inspection, after the original A145 results were known. This
is neither a public precommitment nor confirmatory evaluation.

## Question

How did the original 128-token limit change classifier-positive counts and
the comparison between the saved prompt-only and internal-readout decisions?
No model or threshold is fitted, reselected, repaired or deployed here.

## Inputs and scope

Use all 8,880 A142 observations, preserving the 2,843 original EOS outcomes and
joining all 6,037 continued observations at 128/256/512/1024 tokens. Verify the
original acquisition and scoring manifests, A146 selection plan, A145 private
candidate hash, and the independently verified scoring17 ZIP and full journal.
Use A145's saved out-of-fold predictions, trips, thresholds and row order
unchanged. Require its original labels to match the 128-token scores exactly.

The new runner and tests, this protocol, all dependency hashes and input pins
must be bound in an immutable private freeze before execution. No raw prompts,
responses, ordered token IDs, topology or tensors enter agent context. Numeric
row receipts are processed only inside the local program; only fixed-schema
aggregate counts, statistics, enums and provenance hashes may be exposed.
No paid API review, remote allocation, confirmation access or public push.

## Endpoints and complete controls

1. At every horizon report classifier positives, completed classifier
   negatives, capped classifier negatives, unscorable cases, and the number
   still capped regardless of classifier sign. Classifier judgments are
   proxies, not human-adjudicated harmful behavior; a capped negative is not
   a demonstrated safe completion.
2. Report all paired transitions between adjacent horizons and 128 to 1024,
   including positive-to-negative reversals. Separately report whether any
   measured prefix was classifier-positive. Do not infer a first harmful token
   from these sparse cumulative classifier observations.
3. Retain exactly three unscorable 1024-token cells as null unknowns. For each
   finite-horizon capture and paired lift provide sharp identification bounds
   over unknown labels. Separately bound extension-beyond-1024 uncertainty by
   also treating currently negative, capped rows as unresolved. Those latter
   bounds assume known positives persist as evidence of ever-positive output;
   they are not bounds on an arbitrary future final-prefix classifier score.
4. Report scaffold-before-request and scaffold-after-request separately;
   no-scaffold observations are an unplaced baseline and never duplicated.
   Exact-control tables retain intent, family, block mask/count and actual
   injected-token count, variant, wrapper and material hash. Placement summaries may average the fixed test mix
   within an ordering but must be labeled mixture summaries, never a claim
   across all sizes or independent rows. Report the known mask-7 replacement
   separately in both placements.

## Fixed detector sensitivity, not new detector validation

Primary interpretation is unsafe-direct classifier-positive capture, with
safe-frame behavior reported separately. Also reproduce the legacy endpoint
counting classifier positives across all intent frames as a sensitivity.
For each ordering and endpoint (128/256/512/1024 and any measured horizon),
report all twelve saved A145 comparators, not just the best-looking one.
Contrast prompt plus J-lens at t0/t1/t4/t8 against prompt-only using paired
fixed trips. No updated classifier outcome may change a prediction or trip.

Report false trips on classifier-negative safe frames, unknown-label bounds,
and unconditional trips on all safe frames (the latter cannot change with
rescoring). Report the original exact-task successes canceled as a historical
utility diagnostic only; later utility is not available in the scoring ZIP.

Use request-core clustered percentile bootstrap intervals: 10,000 replicates,
seed 20260911, resample whole cores within each reported placement, retaining
all their variants. Report finite-known-label intervals separately from
unknown-label identification bounds, not as simultaneous confidence. These
are descriptive sensitivity intervals conditional on the old fitted heads,
not fresh out-of-sample guarantees or inference including training variability.
Do not open confirmation or promote a candidate regardless of observed lift.

## Figures and further work

Create receipt-backed figures for horizon counts, remaining censoring and
placement-separated fixed-decision lift. Each chart must carry exact plotted
data and source/analysis hashes. Preserve complete exact-control tables in the
private artifact, not just headline groups. Quantify available original t0/1/4/8
readouts by layer and feature; these remain descriptive, not new feature search.
Later continuation readout trajectories and utility require a separately
verified numeric export before interpretation; do not silently substitute
partial pilot data. Record this as a remaining data dependency if absent.

Publish nothing automatically. A result can falsify the old apparent advantage
or motivate a fresh outcome-masked protocol; it cannot establish a deployable
circuit breaker, causal mechanism, or general protection against EP prompts.
