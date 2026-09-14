# Prospective threshold representation repair, 2026-09-14

Status: prospective local engineering specification, written before implementation
and synthetic validation. This specifies an opt-in helper, not a replacement for any consumed
A145/A150 analyzer, saved decision, threshold, fit, or result. No study data are
opened or fitted by this work. Adoption in an experiment requires a fresh
source/input/protocol freeze and a separately identified result.

## Observed limitation

On the migrated Python 3.12.14 / NumPy 2.5.1 environment, the existing perfectly
separating synthetic nested-logistic test selected an inner-validation threshold
of 0.9664072495463042. Outer-refit positives scored 0.966407249546304: a difference
of 2.220446049250313e-16. The frozen exact `score >= threshold` comparison rejected
all positives. This is an observed numerical-boundary limitation, not evidence
that any historical empirical result is wrong. Do not rewrite that historical
implementation or add an outcome-selected tolerance to its decisions.

## New, explicit rule

Use `incremental_threshold_v2.py` only through explicit import. Inputs are
float64 probabilities in [0, 1], with negative infinity reserved for unavailable
scores. Reject NaN, positive infinity, invalid probabilities, overlapping
positive/safe-negative labels, or empty required endpoints.

Enumerate exactly the same attainable finite-score partitions as the old rule.
Tied scores remain one group. Between distinct adjacent score groups, use their
float64 midpoint as the cutoff. If rounding places the midpoint on the lower
score, use the upper score: adjacent representable scores may have no interior
float64 value, and the rule must not pretend otherwise. The all-finite partition
uses cutoff zero; no-trip uses an explicit null cutoff, including at score one.
Missing scores never trip. Continue to use exact `>=`, with no fuzzy tolerance.

Select maximum positive capture, then minimum safe-negative trips, then the
highest cutoff (no-trip is most conservative). Check the fixed 2% inner-validation
safe-negative budget by integer counts: `false_trips * 50 <= safe_negatives`.
Positive and safe-negative denominators include unavailable rows, matching the
historical objective. Record available endpoint counts separately so missingness
cannot disappear. This training constraint is not a guarantee on held-out FPR.

Use the same existing ridge-logistic fit, training-only preprocessing and
request-grouped nested fold structure. Neither held-out scores nor labels can
select the cutoff. Store each fold's cutoff, adjacent score bounds, boundary kind,
and calibration counts. Callers remain responsible for binding request identities,
placement-specific inputs and the full experimental protocol.

The new cutoff has the same decisions on its calibration scores as the selected
old partition, but can change decisions for new scores inside an observed gap.
That is an explicit prospective algorithm change, not byte-identical historical
reproduction. Midpoints reduce avoidable sensitivity when a genuine score gap
exists; they cannot solve overlapping classes, close ties, calibration shifts,
or general floating-point portability.

## Verification contract

Document the legacy one-ULP failure with a controlled synthetic inner/outer
scoring fixture. Verify the new rule with real synthetic logistic fitting and
tests covering held-out-label isolation, exact ties between classes, adjacent
float64 values, zero/one probabilities, unavailable scores, invalid inputs,
permutation invariance, and exact safe-negative budget boundaries. Compare
selected calibration membership against the frozen selector on synthetic cases.
No target-model inference, new study fit, threshold sweep on empirical data,
confirmation access or historical artifact mutation is part of verification.

For CPU-only test collection, import optional PyTorch inside the single test
that constructs tensors. Its absence skips that tensor test while retaining all
NumPy/provenance tests in the same module. Do not install dependencies or change
the frozen model runtime for this engineering repair.
