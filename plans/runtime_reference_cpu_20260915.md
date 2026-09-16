# CPU FP32 reference diagnostic, 2026-09-15

Status: prospective; no reference model calls or outcomes. A164 stopped after its
32-cell development baseline failed. Preserve that failed gate and all prior
artifacts; this diagnostic does not reopen A164 or authorize held-out execution.

Run exactly the same 32 already-observed development prompts, in frozen plan order,
once each. Reuse the checkpoint, tokenizer, native system/user rendering, prompt
token IDs, EOS tokens, seed, greedy decoding and 64-token limit. Use the original
unquantized weights in CPU FP32 with SDPA and four CPU threads. Do not dequantize
NF4 weights, change wording, add examples, select cases, retry failures or tune
configuration after outcomes. No new held-out, scaffold, API or GPU calls occur.

Before model loading, validate the original model-file hashes, development plan
and shared configuration hashes, frozen source lineage and concrete NF4 receipts.
Recompute native token IDs and re-score NF4 responses privately. Freeze the new
script/protocol hashes, prior numeric record and diagnostic hashes, native-input
hash and explicit reference settings. Require at least 48 GiB available RAM; the
FP32 weights alone occupy approximately 32 GB. Launch under an external 7,200-
second timeout, retaining private immutable attempts, results, error locations,
response tokens and elapsed times. Each trial runs once; unresolved infrastructure
failures remain missing rather than task errors. SIGTERM stops this process; the
reference runner starts no model subprocesses.

Report per-arm strict correctness, both-selector-correct world pairs, exact/
other-selector/other-answer/format categories, literal A/B label returns, cap and
infrastructure counts, and generated-token lengths. Compare paired cell changes
and category transitions, with all 32 cells in best/worst binary bounds. Report
AB/BA presentation crossed with active selector as four fixed descriptive strata.
No significance testing, optional stopping or automatic continuation is planned.

This is a post hoc runtime diagnostic, not a new confirmatory task result. CPU
FP32 versus GPU NF4/BF16 changes weight precision, arithmetic precision, device
and the SDPA numerical backend together. Improvement would localize a runtime-
dependent difference; it would not isolate quantization as the cause. Similar
failure would not prove that every runtime is correct. The original failed A164
baseline remains part of the record regardless of this comparison.
