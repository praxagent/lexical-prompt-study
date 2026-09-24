# A191: supplied versus computed answers

This experiment asks whether supplying correct answers changes exact task
success relative to requiring calculation, under one shared response contract.
It compares two input packages; it does not isolate an internal mechanism.

Eight fresh cores each contain two additions of signed two-digit integers. Each
core appears twice: one prompt has missing answer fields, while the other
supplies the correct sums. Operands, order, expected answers and instructions
remain identical within each pair. Core order is deterministic; pairs stay
adjacent, with each arm scheduled first on four cores. The public namespace is
`a191-answer-availability-v1`.

The instruction requires exactly two comma-separated integers. Success requires
terminal EOS and the correct ordered values across the whole response within
64 emitted tokens, including EOS. The oracle accepts ASCII whitespace around
fields, optional signs and leading zeros. A completed capped response fails;
an interrupted or unattempted endpoint remains unknown.

The primary estimand is supplied minus computed success over eight fixed pairs.
Report sharp bounds over unknown endpoints, each arm's fixed eight-case rate
bounds, and joint pair counts. A primary point estimate requires all sixteen
endpoints. Never replace it with a completed-subset comparison.

Supplied-answer qualification is false if any supplied endpoint is a known
failure, true if all eight are known successes, and otherwise unknown. This flag
does not stop scheduled generations or filter the primary comparison. All-success
results describe a ceiling-limited assay. Persistent supplied-answer failures
close this exact-output assay branch without another parser or wording loop.

The selected runtime is the pinned Llama-3.2-3B-Instruct revision
`0cb88a4f764b7a12671c53f0838cd831a0843b95`, CPU FP32, eight intra-op threads and
one inter-op thread. Use deterministic math attention, full-prefix greedy
recomputation, and no cache, autocast, quantization or offload. Two repeated
technical calls precede at most 1,024 scientific calls. Record actual LM body
entries separately from nested decoder entries, which are not additive.

Qualification uses invented cases and a tiny untrained model before the selected
checkpoint is loaded. A new endpoint coordinator reuses the pinned CPU FP32
forward primitive and its receipts, with its own eight-thread preflight and
stricter budget. It performs no prefix capture, detector fitting or encoder work.

Allow one twelve-hour resource queue and one six-hour native attempt, with
60 seconds for cleanup. Readiness requires 36 GiB of available host memory and
64 GiB of free scratch space. Stop at a sampled 24 GiB owned-RSS ceiling, an
8 GiB host-memory floor, a 3 GiB raw-output ceiling or a 16 GiB scratch reserve.
Only task-owned processes may be terminated. These checks are not reservations.

Freeze reviewed source, runtime assets, tests, output destination and code-backup
commit before launch. Keep prompts, responses, token arrays and per-case labels
outside Git. After the attempt, independently verify the saved evidence without
model or tokenizer calls, preserve missingness, report aggregate findings and
archive the immutable study. No old cohort, parser or consumed verifier is rerun.
