# A185: finite native prefix integration qualification

A185 tests whether the pinned local 8B CPU FP32 runtime, native tokenizer/template, a counted cache-free greedy producer and qualified A183/A184 retention compose correctly on two fixed harmless fixtures. Implementation and synthetic qualification are selected; native execution is selected conditional on reviewed frozen code, provenance, complete preparation and resource readiness. This protocol creates no additional human-approval gate. It does not select a detector acquisition, scientific representation, endpoint, semantic challenger or evaluation population.

Only the finite work below is in scope. No warmup, retry, replacement fixture, site search, paid service, original held-out access or historical payload recovery is allowed. A183 and A184 remain unchanged dependencies: `landmark_evidence.py` SHA `302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14`, and `prefix_only_capture.py` SHA `5b7041eaa062d4cdd6297fab062af9c59c11159bc6c593c28d8af36cf25b4c69`.

## Fixed native fixtures

Use this exact system message for both fixtures:

```text
Follow the user's instruction.
```

Fixture 0 has this exact user message:

```text
Reply with the single word READY.
```

Fixture 1 has this exact user message:

```text
Write the integers from 1 through 20 separated by single spaces.
```

Each message pair is system then user, with no extra reminder or suffix. Visit fixture 0 then fixture 1. These requests exercise native availability and retention; their content correctness is not scored. Unexpected wording or early termination never authorizes a changed fixture.

Before launch, freeze the native chat template and rendering parameters, tokenizer/model manifests, exact messages, finite plan and source/runtime bindings. The supervised worker verifies the assets and invokes the pinned CPU loader once, including its tokenizer; loading must produce zero model-body entries. It then prepares and verifies the native rendering and prompt IDs and durably publishes their bound preparation evidence before the first target entry. The core rechecks native preparation equality with tokenizer operations only before its execution claim. Verify template tokenization against encoding the exact rendering with no extra special tokens. Reject payload special-token injection and any prompt longer than 128 tokens; do not truncate, pad, alter values or repair rendering. Retain decoder-returned prefix bytes with special-token stripping and cleanup disabled; a partial-byte prefix need not re-encode to the original IDs.

The native model has 32 blocks and hidden width 4096. Technical layer index 31 is fixed solely to test the last block's pre-normalization residual. This is not evidence that this layer is useful for detection, nor selection of a future scientific readout. Synthetic model qualification may use smaller geometry while preserving a last-block test; it does not alter the native choice.

## Counted greedy generation

Use the actual pinned CPU reference loader, not the older GPU quantized engine or the older cache-enabled 64-token generation helper. The 16-token policy is new and explicit; no closed ancestor configuration is edited or silently reinterpreted.

For each fixture, sample at most 16 token IDs by deterministic first-index argmax of the model's final-position logits. Every sampled token, including EOS, requires one full-prefix forward. At step k the complete input is exactly the native prompt followed by the k previously sampled non-EOS IDs. Use one batch, all-one attention, sequential positions, no past cache, `use_cache=False`, evaluation/inference mode and no autocast. Disable hidden-state and attention output collectors. Stop immediately after the first ID in the frozen native EOS set or the sixteenth sampled ID. No forward follows EOS or the cap; no softmax, temperature, sampling, processor or generation fallback changes the argmax rule.

For every step retain exact complete input IDs, the full finite native FP32 last-logit row as lossless little-endian bytes, the selected ID, and bound attempt/entry/completion receipts. The complete vocabulary row is necessary to independently replay both the maximum and the lowest-index tie rule. Chosen-token logits or a digest alone are insufficient. Reject nonfinite or wrong-dtype outputs instead of casting or choosing another token. Preserve the entire sampled stream, including terminal EOS when sampled, with its exact stop reason and generation-source provenance.

Saved-logit replay establishes the selection arithmetic from retained logits, not an independent reexecution of the model. Actual loader, dispatch and raw-output bindings remain necessary. The new producer and ledger require independent analytical/tiny-untrained qualification before native execution; A184 alone does not qualify generation.

## Four landmark slots and one reference each

After each fixture's generation, visit t0 then t8. All four fixture/landmark slots exist prospectively. Here t counts observed non-EOS output IDs: t0 uses only the prompt; t8 uses the prompt plus exactly its first eight non-EOS IDs. EOS before eight makes t8 unavailable. EOS after eight or a cap at or beyond eight leaves t8 reached. Do not substitute a shorter prefix. No later sampled IDs or EOS enter a capture/reference input even when later evidence is retained privately.

For each available slot, call A184 exactly once, then execute one independent same-input normalization-reference forward. The adapter captures the last block output at the last supplied token. The reference uses owned final-norm pre/post hooks that return None and are removed afterward. It is a reference with capture-only hooks, **not a literally hook-free baseline**. No extra unhooked baseline is included in the budget. Disable `output_hidden_states` and `output_attentions` in model configuration and reference calls; do not use lazy hidden-state collectors.

Retain the adapter-returned full native FP32 last-logit row and the reference's norm-input vector, norm-output vector and full last-logit row. Require finite exact shapes/dtypes and lossless storage. Compare adapter residual bytes exactly with the reference norm-input last-token bytes, and compare adapter/reference last-logit bytes exactly for unchanged-output behavior. Keep post-normalization values separate; they are not the adapter target. No tolerance change, alternative site or extra reference pass repairs disagreement.

An unavailable t8 retains its A183 unavailable evidence and zero adapter/reference target entries. If a hard failure prevents a slot from being visited, retain that planned slot as unattempted rather than treating it as early EOS or deleting it. This is prefix replay after generation, not a claim about online detector latency.

| Forward-entry ceiling | Count |
|---|---:|
| Two generations, at most 16 entries each | 32 |
| A184 capture at each of four available slots | 4 |
| One norm-hook reference at each of four available slots | 4 |
| Total, with no warmup or extraction extras | 40 |

An entry is observed by the owned model-entry prehook before the forward body, not proof of body completion. Record attempted work, entries, completed forwards and usable published artifacts separately. A184 dispatch accounting uses its own bound receipts, without adding a foreign hook that would violate its hook audit. If a failed adapter lacks sufficient dispatch evidence, retain the known lower bound and mark the exact capture/total entry count unknown; do not silently record zero. EOS-producing steps count. Nested decoder invocations within one model call are not additional top-level entries. The ceiling covers all target forwards; unreachable slots reduce actual entries without changing planned coverage.

## Failure, publication and interpretation

A hard infrastructure, input-validation, interruption or forward/publication failure stops further execution. Retain the failed/ambiguous operation and every later planned slot as unattempted; never resume or retry. A fully completed adapter/reference comparison mismatch instead records false and continues the remaining fixed schedule. No generated content or technical comparison result selects more fixtures or changes the schedule.

Usable completed adapter evidence requires a bound normal successful writer return and subsequent whole-directory A184 validation. The opaque mutable forward output is not itself immutable evidence; explicitly snapshot and bind the required logits before comparison/export. An A183 child bundle alone is insufficient. A readable result or successful loader cannot prove the original caller observed normal return, rule out a lost later failure marker, or certify an in-progress final fsync. Preserve such ambiguity; never retroactively promote file presence into process success. Receipts, raw bytes, dependency/source bindings and the returned completion observation must agree.

A technical pass requires both t0 comparisons and at least one actually reached t8 comparison, all available completed comparisons passing exactly, successful cleanup and complete accounting without failed, ambiguous or unexplained missing operations. If both t8 slots are unavailable, report unexercised t8 coverage rather than expanding the fixtures. Known mismatch is a failed technical check; an interrupted or incomplete execution is not a zero difference. All four planned slots and their reasons remain visible. There are no population confidence intervals or scientific response-quality scores.

Any A183 manifest descriptors for labels, groups and comparators must explicitly identify this no-fitting apparatus scope. Landmark outcomes remain missing/not measured, and comparator descriptors bind no fitting or calibration. A185 does not fit, predict or publish a prediction-pair packet; these descriptors do not imply prediction rows were acquired. Descriptor hashes are not evidence that a scientific endpoint or comparator was qualified. Preserve common-time predictor isolation; neither future outputs nor technical outcome labels enter a text-only feature view.

## Core API and caller responsibilities

`compile_plan(...)` accepts explicit model/tokenizer/template provenance, vocabulary and native geometry, EOS IDs, protocol/test hashes and a runtime-binding hash. It fixes the two fixtures, four slots and finite policies above. `prepare_inputs(plan, tokenizer)` renders and checks the native inputs; this preparation invokes the tokenizer but no target forward. The injected core does not load a model or tokenizer.

`run_qualification(model, tokenizer, *, plan, prepared, runtime_binding, directory)` consumes a new exclusive evidence directory. It rechecks the native preparation and actual model constraints, then runs the fixed schedule. On normal completion it returns the replayed summary, four retained records and terminal hash. Hard failure preserves available failure evidence and rethrows; a partial root cannot become qualified. `load_qualification(directory, plan)` replays the complete saved logits, selection arithmetic, comparisons, source/input bindings and accounting without target or tokenizer calls. It does not authenticate execution from hashes alone. The outer frozen caller must separately bind its observed normal runner return, loader provenance and supervised terminal state; a complete saved root alone cannot replace those observations.

## Runtime and evidence prerequisites

Use offline CPU FP32 with SDPA, four threads, no CPU autocast and no CUDA. Verify the actual loader/model/tokenizer/template/runtime provenance; opaque model pins alone do not authenticate loaded weights. Require at least 64 GiB available before loading. A cooperative A185 file lock prevents another cooperating A185 launcher from consuming the same run; it does not reserve RAM against other jobs. The supervisor samples owned-session RSS every 0.2 seconds and stops the owned run when the sampled total exceeds 64 GiB. This is a sampled stop threshold, not an instantaneous kernel allocation cap. A host available-memory reading below 8 GiB also stops the owned run conservatively. Apply a 1,800-second worker wall limit including model loading, with 60 seconds termination grace. Inspect and signal only the newly created owned session; do not interrupt other jobs. Supervisor TERM, INT and HUP must unwind through owned-session cleanup; the worker also binds a parent-death signal before loading. Resource readiness and a final resource recheck precede execution; no throughput assumption extends the deadline.

Before launching, independently qualify producer stop/argmax/entry/retention semantics, adapter/reference equality and failure handling on invented analytical and tiny-untrained fixtures. Review actual native/CPU integration and the resource controller, replay the exact final tests, verify code backup, and freeze source, fixtures, declared identities, schedule, plan and allowed inventories before launch. Bind the actual native preparation inside the supervised worker before any target entry, as specified above. No target call is a preparation warmup.

Raw prompt/rendered bytes, IDs, outputs, logits, residuals and per-item receipts remain private outside the public repository. A terminal verifier replays complete retained evidence and accounting without additional target forwards; archives include the actual payloads, not only hashes. Public reports contain aggregate technical coverage and whole-artifact bindings only. Completing A185 establishes this finite integration result; any detector acquisition still needs its own population, outcome/uncertainty policy, representation, matched semantic challenger, grouped evaluation and resource design.
