# A146: repair censored generation outcomes before further detector work

Status: prospective exploratory repair; no confirmatory detector evaluation.

The user approved a fresh **$15 total hard cap, including a $2 pilot cap**.
This does not authorize another project's compute or release of restricted data.
An exact private execution bundle binds the original acquisition, topology,
scorer, model, SAE, Jacobian lens and source commit. Selection uses the original numeric
finish reason only, never classifier labels or internal feature values.

## Frozen measurement

- Preserve original results. Continue every eligible original length-limited
  response from its exact saved prompt tokens and exact 128 generated tokens.
  Do not regenerate that prefix, insert a new instruction or re-tokenize it.
- Use the same pinned Llama 3.1 8B Instruct, BF16/eager, greedy policy. Re-prefill
  the prefix because the original key/value caches were not saved. Before any
  pilot row, audit four fixed saved prefixes: trusted Hugging Face generation
  receives the exact saved prompt-plus-128-token sequence, and its next eight
  tokens/EOS must match the adapter continuing that same sequence in two cache
  chunks. Verify that both paths leave the entire saved input unchanged.
  This conditional-continuation gate does not regenerate the first 128 tokens
  or require historical batched output to equal fresh singleton output. Preserve
  floating-point limitations explicitly; the repair does not claim identity to
  hypothetical uninterrupted decoding on the historical hardware/batch.
- Observe 128, 256, 512 and 1,024 total generated-token horizons, stopping at EOS.
  EOS is not counted as response text, matching the original convention.
  Persist token-exact, hash-bound restricted checkpoints after every horizon.
- At extended horizons use the pinned HarmBench scorer on the entire cumulative
  response, asserting its context window before inference. Report each horizon
  separately and cumulative ever-positive status separately. A negative score
  on a still-truncated response remains censored, not a demonstrated safe end.
  Carry original EOS outcomes forward without further target-model generation.
- Preserve generation-phase parser and exact-safe-task outcomes. Keep request
  intent, scaffold family, dose/block mask and both scaffold placements separate.
  Do not describe classifier positives as adjudicated harmful behavior.
- Retain raw BF16 residuals at layers 8, 16, 19, 24 and 30, and the existing
  Jacobian-lens / SAE readouts at generated lengths 16, 32, 64, 128, 256, 512,
  1,024 and actual EOS where available. Historical positions are replayed from
  saved tokens, not claimed as original cached states. These are exploratory
  retained measurements; no detector fitting or confirmation is authorized.

## Pilot and expenditure gates

Select 64 rows using a fixed hash order, reserving each available family ×
placement and balancing intent × placement × block count. The deliberately
balanced pilot is descriptive, not a representative population estimate.

Local schema, crash recovery, exact-prefix, EOS, context-limit, deadline and
ownership tests must pass before allocation. Launch exactly one task pod on
the retained volume. Record the exact ID, rate, deadline and source archive.
Request provider-side `terminateAfter` and run an independent local deadline
guard; terminate immediately at completion or unrecoverable failure. Neither
guard may mutate unrelated pods. A provider deadline request is not called a
verified backstop without receipt evidence that it was accepted.

The pilot's conservative worst-case allocation, boot, loading and scoring
allowance must fit $2. Check live progress, not merely process existence. Project
full cost using measured stratum-weighted generated-token/runtime data with a
safety margin, loading/scoring/storage reserves and the already spent pilot.
Freeze a full-scope authorization only if that projection fits $15. Otherwise
stop paid compute and report the bottleneck; do not silently increase budget,
drop expensive strata, shorten the agreed horizon or claim a completed repair.

All new outcomes, raw states and row-level receipts stay private. Public release
of aggregate results requires explicit user authorization. This protocol does
not reopen confirmation, select a new detector, or authorize deployment.
