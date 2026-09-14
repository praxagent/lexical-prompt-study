# A153: repair and execute-test the metadata probe wrapper

Status: operational correction after CPU24 stopped, before any Stage-B feature
export or fitting. No scientific specification in A150 or A151 changes.

CPU24 staged its source and tokenizer packet but produced no runtime-probe
receipt. Its generated probe command imported `runpy`, then invoked the existing
pod-identity helper, which requires `os` in the command's global namespace.
Locally executing those exact frozen wrapper bytes reproduced `NameError` for
the missing name `os` before the probe could run. This is a wrapper defect, not
evidence of a library-version mismatch or an experimental result. The outer
controller retained only the exception type; its final `not_copied` error did
not preserve the original failure location.

The exact task-owned CPU24 pod was deleted and independent inventory confirmed
zero study-owned pods and the retained volume. CPU24's rounded-up compute cost
was $0.01. The closed-operation base is now $25.62, plus the existing cumulative
storage planning reserve and later actual compute, within the $70 round cap.
The failed closure, source bytes and operation namespace remain immutable.

## Bounded correction

Use a new CPU26 namespace. Import `os` explicitly in the generated probe wrapper.
Before allocation, execute that actual generated command locally with the real
frozen identity-helper source, a synthetic pod-ID environment variable and a
stubbed probe entrypoint. Verify that ownership is checked and the intended
probe entrypoint is reached; also test rejection of a different pod identity.
Do not replace the identity helper with a constant-return mock in these tests.

Preserve a bounded operational-stage failure code, exception type and message
hash before the generic parent controller handles any failure. Do not expose
raw stderr, tracebacks, environment values, prompts or experiment data.

Keep A152's metadata-only runtime qualification, exact Python/Torch/Transformers
pins, NumPy-2.x/five-matrix byte-identity gate, recorded actual versions, and
persisted exact qualified runtime specification before experimental reads.
Reuse the frozen metadata-probe implementation; change the scientific exporter
only to use the new destination namespace and test source equivalence.

Retain the same single-pod CPU-only $0.25/30-minute cap, measured throughput
gate, startup and no-progress limits, transfer reserve and independent teardown
guardian. No install, model inference, GPU, paid review, public release,
confirmation access, or mutation of another project's resources is authorized.
Cross-bind the new source/input freeze, runtime qualification, exact ownership,
transfer and independent local closure before any fitting. The same seven
feature families, folds, endpoints and threshold rule remain fixed.
