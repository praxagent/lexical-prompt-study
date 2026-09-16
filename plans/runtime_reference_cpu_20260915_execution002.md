# CPU reference execution 002: text-log compatibility correction

Status: prospective replacement execution; no execution-002 model calls or outcomes.
Preserve the original frozen execution 001, including its failure and zero trial
attempts/results. It stopped during model loading, before generation: Transformers'
weight-loading progress writer passed text to the runner's binary private log,
raising `TypeError`.

Change only the live runner's private `execution.log` open mode from exclusive
binary creation to exclusive UTF-8 text creation. This accommodates standard
stdout/stderr and progress writers while retaining private logs and one-shot
output ownership. A synthetic main-entry regression must emit both printed text
and Unicode progress through the redirected streams, complete the fake run, and
verify that neither message reaches public stdout/stderr.

All scientific design and model settings remain those in
`runtime_reference_cpu_20260915.md`, SHA256
`1e2a3074f282acfa4e15331b42b19e6c4ca05e786e8826a93f6dec412aeb8467`:
the same 32 already-observed development inputs, original-weight CPU FP32 SDPA,
four threads, identical native tokens/seed/greedy 64-token generation, fixed order,
48 GiB memory gate, 7,200-second external bound, no per-trial retries and no
held-out execution. This is an infrastructure correction before any reference
outcome, not a prompt, model, cohort, metric or interpretation change.

Freeze the corrected source/tests and this amendment before launch. Use a new
private execution-002 output directory; never overwrite, resume or edit the
frozen failed execution 001. Bind this amendment hash in the replacement run's
reference-protocol field and retain the original design document/hash alongside it.
