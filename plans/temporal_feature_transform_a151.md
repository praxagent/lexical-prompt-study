# A151: precise retained-state transforms for A150 Stage B

Status: pre-transform numerical clarification. A150 Stage A aggregates have
been inspected; the richer retained-state and same-prefix comparisons have not
been fitted or inspected. This annex does not add candidate families, change
labels, modify the sealed confirmation set, or authorize new model inference.
The parent protocol is `temporal_corrected_a150.md`.

## Deterministic residual transform

Use residual-post layers 8, 16, 19, 24 and 30, in that order. Verify every
retained file's SHA-256, trial identity, generated-prefix identity, model
revision, layer set, shape and BF16 dtype before numerical use. Deserialize
only the pinned task-owned payload, with `weights_only=True` and CPU mapping;
never execute arbitrary pickle code or load target weights.

Convert each 4,096-element BF16 vector to float64. Compute its Euclidean norm
in float64; reject zero, nonfinite or malformed vectors. Let `u = x / ||x||`.
Generate five independent `(32, 4096)` matrices in layer order using NumPy
PCG64 seed 2026091201 and equiprobable signs scaled by `1/sqrt(32)`. Freeze
the exact matrix bytes and hashes, generation code, NumPy version and dtype
before transforming retained states. The concrete RNG draw operation is part
of that executable freeze; the seed alone is not a matrix identity.

Export `P @ u` (32 float64 values) and `log(||x||)` (natural logarithm) per
layer. Do not renormalize the projected vector. The parent protocol's residual
difference means `P @ u_current - P @ u_previous`, not a separately normalized
raw difference. Local analysis derives this difference from the exported
coordinates using the immediately prior planned available landmark within
the same acquisition regime. It never uses a future state or substitutes an
outcome-chosen peak, EOS position or window. All retained positions may be
transformed for accounting; fitted comparison landmarks remain those in A150.

## Same-prefix text and utility

No prompt or generation text is displayed to the research agent. Programs
inside the quarantine decode exact saved generated-token prefixes using the
Llama 3.1 8B Instruct tokenizer, model revision
`0e9e39f249a16976918f6564b8830bc894c89659`, offline and with remote code disabled.
Pin tokenizer JSON, tokenizer config, special-token map, model config and
library versions in the execution receipt. Do not infer tokenizer identity
from a directory name or borrow another Llama checkpoint's hashes.

Use the same `skip_special_tokens=True` decoding and cleanup semantics as the
historical acquisition. Require every complete original generation's decoded
hash to match its original receipt, and every exported landmark's token-prefix
hash to match the corresponding internal readout. This is a comparison with
externally emitted text plus generated-token count; skipped special tokens are
not silently presented as decoded words. Never re-encode text to manufacture
token identity.

Apply the original 256-dimensional signed byte 3/4/5-gram hashing and L2
normalization to the decoded emitted prefix. Bind the exact original hashing
source and test its synthetic parity before use. Original t0/t8 prefix
features use the complete original cohort where those landmarks exist;
continuation features use only the correctly selected continuation cohort.
An EOS before a landmark is unavailable, not a later repeated observation.
Current token count is available; final generation length and EOS/cap status
are never predictors.

Reuse the original three fixed safe-response expectations and refusal parser,
with source hashes and expectation hashes checked against the original numeric
receipts. Recover exact-task utility and parser outputs during the same
quarantined decode. These are mechanical outcomes, not human-adjudicated
harmfulness or semantic utility. No evaluator call or harmful-core text is
needed for this computation.

## Bounded acquisition and analysis gates

Recover features on CPU next to the retained data, exporting only an opaque
private numeric-array bundle plus hash/coverage metadata. Raw tokens, prompt
text, generations, original tensors and hashed feature vectors do not enter
agent context. Do not download model weights. Reject symlinks, path traversal,
unbounded archives, incomplete coverage, source drift and nonfinite arrays.

The additional CPU job requires its own immutable source/input/transform
freeze, passing synthetic tests, exact pod identity, zero duplicate inventory,
at most $0.16/hour CPU plus the existing storage reserve, a $0.25 allocation
ceiling and at most 30 minutes. Measure representative throughput and preserve
a transfer/teardown reserve; stop if the remaining work does not fit. Never
extend a lease or change consumed code while it runs. The existing $70 round
ceiling remains cumulative; a per-job cap is not a fresh budget.

Verify copied numeric arrays and coverage locally after exact compute teardown
before fitting Stage B. Preserve missing/selected-cohort counts, numerical
acquisition regimes, both placements, and the A150 fixed model set. Time a
representative local fit before a large local sweep; resume immutable fitting
checkpoints. Human audit, fresh held-outs, latency, sequential false alarms
and causal intervention remain separate requirements before a defense claim.
