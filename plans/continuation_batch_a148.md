# A148: bounded batched continuation repair

Status: internally frozen exploratory correction, not confirmatory evaluation.
This supersedes only the numerical execution path and spending envelope of
A146/A147. It does not authorize public release of outcomes. The private source
commit and source manifest precede new batched outcomes; they are not a public
precommitment or formal registration. Unpublished earlier result commits prevent
an unapproved push of this branch.

## Question and unchanged measurements

Did the 128-token cap leave the safety assessment materially unresolved, and
what later internal trajectories accompany the continued responses? This repair
can describe that fixed corpus under the pinned continuation path. It cannot
validate a detector, establish causality, reopen confirmation, or authorize an
automatic circuit breaker. Negative, positive and still-censored outcomes are
all retained. A classifier positive is not human-adjudicated harmful behavior.

Keep A146's exact original prompts, saved 128-token prefixes, model revision,
BF16/eager greedy settings, scorer, SAE, lens, token horizons and readouts.
Continue the 6,037 originally length-capped rows; carry the other 2,843 original
EOS outcomes unchanged. The 64-row outcome-independent pilot selection remains
identical. Preserve before/after placement, intent, family and block-count
labels separately. There is no adaptive prompt construction or attack search.

The target executor alone reads restricted text and tokens. Agents see code,
counts, timings, numeric summaries and non-reconstructive hashes. Exact text,
ordered token IDs, model generations and raw residuals stay on the retained
study volume. No operational payload enters a review packet or public artifact.

## Changed numerical path, declared before execution

The first partial singleton pilot passed four conditional-prefix parity checks
and completed 12 of 64 rows before its deadline; it was not scored. Timings
showed model forwards dominated cost. These partial results remain an immutable
A147 diagnostic attempt. A148 starts a fresh namespace from the original
saved 128 tokens, never from the newly generated singleton continuations.

Use fixed batches of four physical slots with exact left-padded token IDs and
attention masks. Freeze deterministic batch membership and slot order; pilot
groups remain unchanged when full collection adds the remaining rows. Keep
finished slots in their original physical positions and pad the final short
group using the frozen adapter rule. EOS padding is computational only: it
does not create an observation or extend the recorded response beyond EOS.

Trusted Hugging Face greedy generation advances exact saved prefixes to total
generated-token horizons 256, 512 and 1,024, or the first configured EOS. The
EOS token is not counted as response text. Preserve a group-level immutable
token checkpoint before diagnostics; crash recovery must resume the same group
and exact tokens without substituting another row or changing slot placement.
Interruptions during an uncommitted segment may repeat that segment under the
same layout. An existing committed token checkpoint is never overwritten.

Before pilot acquisition, compare the adapter with independently constructed
trusted Hugging Face generation conditional on the same four exact prefixes,
same physical layout and same padding. Compare new-token IDs and EOS, including
a split continuation boundary. Verify saved prefixes are untouched. A failure
stops paid target acquisition; it cannot be waived using approximately equal
logits. Small synthetic CPU tests cover unequal lengths, mixed EOS, tail slots,
context overflow, token checkpoints and interruption/resume before allocation.

This is a batched conditional continuation, not proof of bitwise identity to a
singleton run or the historical uninterrupted batch. Generation phase readouts
are explicitly prefix replays, not recovered historical KV-cache states. Replay
must use the exact generated sequence and the frozen valid-token position.

Preserve SAE feature 6779, the frozen eight-feature comparator, all 31 layer
J-lens readouts, and BF16 residuals at layers 8, 16, 19, 24 and 30 at generated
lengths 16, 32, 64, 128, 256, 512, 1,024 and actual EOS when available. Do not
drop instrumentation to improve throughput. Score each cumulative horizon with
the pinned HarmBench evaluator; retain parser/utility outcomes and context
validation. Report per-horizon and ever-positive summaries distinctly. A
negative response still at the cap is censored, not demonstrated safe.

## Cost gate and ownership

The user approved **$70 total for this continuation round, including prior
spending**. Explicit authority ID: `a148-user-70`. The operator allocates at most
**$8 cumulative pilot/setup**, inclusive of earlier failed startup and singleton
pilot costs. The conservative prior reserve is $2.00 including additional
retained-volume time and rounding; actual billing is not yet settled.

Permit one task-owned pod at a time, on the existing study volume. Never touch
another project's pod. Quote the actual available GPU before allocation; use
at least 80 GB of GPU memory and the retained CUDA 13 runtime with a compatible
driver. Maximum quoted GPU rate is $6.79/hour. Each allocation binds exact ID,
rate, image digest, source manifest and absolute deadline; its worst-case rent
plus storage/teardown reserve must fit both remaining stage and round budgets.

The next pilot allocation has a 40-minute maximum including startup, at most
$4.70 including reserve at the maximum rate. If startup makes no progress for
five minutes, or execution fails without a diagnosed safe recovery, retrieve
sanitized diagnostics and terminate the exact owned pod. Request provider
expiry and run a separately live-verified local deadline guard. A requested
provider expiry is not described as an independently verified provider stop.
Record deletion and verify exact lookup plus fresh inventory.

The full gate requires all 64 pilot rows, all required readouts and scoring,
and unique per-batch wall-time receipts. Do not sum duplicated batch time as
though each slot ran independently. Separate generation time from diagnostic
and I/O overhead. Only complete pilot cohorts with at least 128 measured decode
steps qualify for throughput extrapolation; no qualifying cohort means no full
projection. Normalize their generation time conservatively to the full
896-step extension ceiling and take the slowest. Add the largest fixed
diagnostic/I/O overhead across all pilot cohorts, scaled to four replay passes
(one for historical sites and one per new horizon). Never multiply one-step
EOS setup overhead by 896. Apply a factor of at least 1.5 to the projected
remaining fixed groups and scoring. Project scoring conservatively for all three new
horizons of every remaining row; add explicit validation/loading, storage,
already-spent pilot and retrieval/teardown reserves. Report missing pilot strata
without inventing their means. This slow-case planning estimate is not a
statistical confidence bound or a guarantee; an exact-ID deadline remains
necessary. If the estimate does not fit the remaining $70, do not launch the
full run or silently shrink the scientific sample.

## Reporting and next decision

Hash and verify the private safe receipt bundle before interpretation. Preserve
every original and successor attempt, including failures. Report completion,
remaining censoring and compute/storage costs before any claim of progress on
detection. Further detector fitting or causal localization requires its own
prospective design and unopened evaluation data; this repair does not turn an
exploratory signal into a validated defense. No result publication is implied
by the budget approval.
