# A186: matched-prefix prediction of finite arithmetic completion

This is a prospective, fixed local prediction pilot. It asks whether one frozen internal representation adds predictive value to the same prompt-plus-prefix text pipeline on fresh request cores in a held-out CSV format. It does not reopen the earlier lookup-generation studies or test safety, intervention efficacy, a population of attacks, or learner-independent hidden information. Acquisition is permitted only from the reviewed, qualified, pushed and frozen implementation with the declared local resources. No new permission or human rating is a study prerequisite.

## Fixed public tasks and schedule

There are 48 fixed cores, each with twelve additions. For core `c` in 0–47, item `i` in 0–11 and side `s` in 0–1, hash the exact ASCII string `matched-prefix-arithmetic-v1|{c}|{i}|{s}` with SHA-256. Interpret the full digest as a big-endian integer modulo 180. Map residues 0–89 to `z-99` and residues 90–179 to `z-80`. This yields nonzero signed two-digit operands. The expected outputs are their twelve ordered integer sums. No outcomes alter difficulty, assignment, or sample size.

The system instruction is exactly:

```text
For each of the twelve data rows, add its two integers. Preserve their row order. Output exactly twelve comma-separated base-10 integer sums, with no other text.
```

The user message joins twelve data rows with one ASCII newline and has no trailing newline. The formats are:

| Format | Exact row representation | Header |
|---|---|---|
| `plain` | `{a} {b}` | None |
| `labeled` | `a={a}; b={b}` | None |
| `csv` | `{a},{b}` | `a,b` followed by one newline |

| Core indices | Split | Formats | Rows |
|---|---|---|---:|
| 0–31 | `fit` | plain, labeled | 64 |
| 32–47 | `evaluation` | plain, labeled, csv | 48 |

All variants of a core share one group. Observation IDs are `core_{c:02d}_{format}`; core and group IDs are `core_{c:02d}`. These 112 rows comprise 48 fixed groups, not 112 independent samples. CSV is unseen formatting on fresh request cores; individual operand pairs can recur across cores, so this is not necessarily unseen arithmetic facts.

Run in ascending digest order of `matched-prefix-arithmetic-v1|schedule|{c}|{format}`, with core index and format string as tie breakers. Each row retains its digest and zero-based sequence index. Split and outcomes never choose execution order.

Input-only strata are frozen before labels. Count positive-positive, negative-negative and opposite-sign pairs. A same-sign pair has a units carry when the two absolute units digits sum to at least ten. An opposite-sign pair has a units borrow when the units digit of the larger absolute operand is less than that of the smaller; equal magnitudes do not borrow. Also count zero sums. These six per-core counts are retained as string-valued manifest-compatible metadata. They describe arithmetic structure, not empirically established difficulty, and do not select a subgroup or replacement core.

## Common-time native acquisition

Use the pinned local target with CPU FP32, SDPA and four threads. The native manifest binds model assets, tokenizer, chat-template bytes, runtime, source and provenance; declared pins alone do not prove runtime identity. The sole capture site is final block 31, width 4096, in the 32-block model. It is a fixed technical choice, not a discovered effective site. There is no site, rank, precision or landmark search.

Native preparation verifies the fixed payloads, special-token policy, template rendering and tokenized rendering against an independent encode of that rendering without added special tokens. Use the frozen template date `26 Jul 2024`. The native assistant probe round trip verifies the generation boundary and terminal EOS. Each prompt must have at most 256 tokens and prompt length plus the 64-token generation cap must fit the model context. No truncation, padding, reformulation or alternative fixture is allowed on failure. Persist the actual native-prepared artifact before model execution; do not confuse a source-derived plan with native-tokenizer evidence.

Generate greedily with cache-free full-prefix forwards, sequential positions, all-one attention and first-index argmax. Save every finite FP32 last-row raw logit vector needed for model-free argmax replay, emitted IDs, stop reason, source/input bindings and actual dispatch/completion evidence. Cap at 64 sampled tokens including terminal EOS; stop immediately after EOS or cap. Count the EOS-producing call. There are no warmups, retries, extra reference forwards or t0 captures.

The only landmark is eight observed **non-EOS** output tokens. For every reached row, the unchanged qualified A184 adapter makes one separate prefix-only forward on exactly `P + Y[:8]`, with no target suffix or cache, and captures the last-input-token post-block residual. That extra boundary forward is necessary after sampling the eighth token and is included in cost. Preserve native finite FP32 bytes without casting. Successful capture requires the adapter's normal writer return and whole-directory validation; file presence alone cannot establish that the caller observed normal completion. Runtime hook cleanup and failed-after-hook semantics remain those of A184. All hidden-state/attention collection is disabled outside the explicitly owned capture hook.

EOS before eight leaves the t8 prefix unavailable even if the finite task label is known. EOS immediately after eight leaves the boundary available; that later EOS is not a feature. A missing or failed capture is distinct from prefix unavailability. Retain every planned row and all null/unattempted slots. Actual native decoding keeps `skip_special_tokens=False` and `clean_up_tokenization_spaces=False`; exact decoded UTF-8 may include replacement characters and need not re-encode to the original token IDs.

## Finite outcome and conservative prefix check

A completed response succeeds only if it terminates with frozen native EOS by the 64-sampled-token cap and contains exactly twelve comma-separated integer fields, each equal to its corresponding sum. Around each field permit only ASCII space, tab, carriage return, line feed, vertical tab and form feed. After stripping that whitespace, each field must match `[+-]?[0-9]+` in its entirety. Leading zeros, optional plus and negative zero are permitted by integer value. Unicode digits, Unicode whitespace, prose, empty fields, extra commas and other syntax do not qualify.

A completed cap without EOS is label 0 even if the visible arithmetic is correct. A completed wrong, partial, malformed or extra response is 0. Infrastructure failure, interruption and unattempted generation have missing labels; partial text cannot become an observed label. A cap-zero is failure to complete under this finite policy, not inability with more time or a safety conclusion. Preserve this endpoint separately from A183's derived censoring metadata: absent terminal EOS with fewer than 64 retained content tokens is censored; a completed cap is not. A183 stores instrument evidence and does not establish label truth by itself.

At t8, close a field **only at an observed comma**. Flag a wrong or malformed closed field. The twelfth comma already implies an extra field and therefore a known irreversible failure. Never judge an open tail wrong, even if it contains an incomplete sign, wrong-looking numeral or prose; never consult later delimiters, later text or EOS. The shared prefix features are the observed comma count divided by twelve, without clipping, and this Boolean flag. They are derived from the prompt and observed prefix only. The unnormalized count is also retained as provenance for those features.

Both comparators apply a success-probability-zero override separately **only if that comparator already has a valid learned prediction** and the flag is true. A missing learned prediction stays null. Preserve raw feature, fit and prediction availability and the override indicator. Report flagged fractions and unflagged label/class/core support separately in fit and evaluation. Overall class support can arise solely from already-exposed failures; if no unflagged evaluation errors remain, do not claim prediction of still-unrevealed errors. Even an unflagged prefix is not certified unknowable by other visible cues.

## Frozen feature and fitting pipelines

Both predictors receive the same exact rendered prompt, observed t8 IDs and decoded UTF-8 bytes, plus the two common prefix features. Text-only feature packets exclude future suffix, final length, stop reason, labels, later parser state, internal values and runtime outcomes. Labels and row/group joins are separate. Acquisition preserves lossless text/internal evidence through unchanged A183/A184 contracts; native generation and capture provenance require the separately qualified acquisition adapter.

The semantic encoder is `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, with all six local assets authenticated before execution. Run offline on CPU FP32/eager with four threads. Encode two exact documents: `Prompt:\n` plus rendered prompt, and `Assistant prefix:\n` plus decoded prefix. No lowercasing beyond the pinned tokenizer, additional normalization, suffix or stop metadata is supplied.

Tokenize without truncation. Partition into contiguous chunks of at most 510 content tokens and add native CLS/SEP. Each document has at most eight chunks; reject oversize data without fallback. Execute one chunk per encoder forward, with batch size exactly one and no deduplication, including repeated prompt documents. L2-normalize each CLS vector, combine by content-token-count-weighted averaging, then L2-normalize. The two 384-dimensional embeddings contribute 768 dimensions. Fixed headers make empty decoded text an explicit nonempty document. Unavailable prefixes do not receive fabricated documents or embeddings.

Add a 256-dimensional signed prefix-byte hash. Enumerate all overlapping 3-, 4- and 5-byte n-grams of the exact decoded-prefix UTF-8. Hash `bytes([n]) + gram`; digest byte 0 chooses the bin; add +1 when digest byte 1 has low bit zero, otherwise −1. Accumulate occurrences and L2-normalize; a zero vector remains zero. No Unicode normalization or case folding enters this hash. Together with the two common prefix features the text pipeline has 1,026 features.

The augmented pipeline adds sixteen PCA components of the actual final-block vector. Convert retained FP32 values to float64 for feature processing, L2-normalize nonzero input vectors, center using eligible fitting rows only, and use full SVD with whitening disabled. Retain sixteen components even when rank deficient. Orient each component so its largest-absolute coordinate is positive, with lowest-index tie breaking. Report numerical rank at `eps64 * max(n_rows,n_features) * largest_singular_value`; do not substitute a different rank. Preserve the fitted transform and its fitting-group bindings.

Fit one unweighted L2 logistic model per comparator with `C=1`, `solver=liblinear`, `tol=1e-6`, `max_iter=1000`, and seed `20260915`. There is no calibration fit, tuning grid or threshold search. Both use the same eligible fitting cohort: reached t8, valid paired features and known label. If fewer than seventeen fitting rows or fewer than four distinct fitting cores contributing to either class remain, the fits are unavailable. Report every excluded planned fitting slot. Nonconvergence or unavailable required transform has no alternate solver, rank, encoder or observed-subset repair. The four-core rule is an engineering guard, not statistical power.

## Fixed estimands, missingness and interpretation

Let text and augmented probabilities be `p` and `q`, and outcome be `y`. The paired Brier improvement is `d=(p-y)^2-(q-y)^2`; positive favors the augmented pipeline. The primary averages the sixteen CSV evaluation rows equally. The separate secondary averages the two known formats equally per evaluation core, then averages the sixteen cores. Preserve both probabilities and their individual null reasons.

Any missing required label or probability keeps that fixed-cohort point null, even if its bounds collapse. Use these sharp row bounds and the same fixed weights:

| Known quantities | Row interval |
|---|---|
| p, q, y | `[d,d]` |
| p, q | Min/max of d at y=0 and y=1 |
| p, y | `[(p-y)^2-1, (p-y)^2]` |
| q, y | `[-(q-y)^2, 1-(q-y)^2]` |
| p only | `[min(p^2,(1-p)^2)-1, max(p^2,(1-p)^2)]` |
| q only | `[-max(q^2,(1-q)^2), 1-min(q^2,(1-q)^2)]` |
| Neither probability | `[-1,1]` |

Retain existing probabilities when their pair is missing. There is no observed-subset primary. These bounds express missing-data possibilities, not confidence intervals. Report finite group-weighted results and coverage, not repeated-format independence or population confirmation.

Prespecify a descriptive reached, unflagged evaluation analysis with actual row/core support and retained missing probability/label slots within that subgroup. Average the retained subgroup formats within each represented core, then average represented cores equally; a core with more retained formats gets no greater total weight. A separately hash-bound analysis-only `landmark_reached` map covers every observation: true means reached, false means known unreached (including early EOS), and null means unknown. Known unreached rows are nonmembers, not unknown members. A reached row whose common prefix/flag is unavailable still has unknown subgroup membership. The production wrapper supplies this complete map; it never enters feature arrays or fit eligibility. Unknown membership makes the diagnostic unresolved; an empty subgroup is undefined. It does not replace the primary. Evaluation class collapse is reported without new cores or changes to difficulty, landmark or cohort.

A resolved positive primary supports only this finite two-pipeline comparison on the sixteen CSV core slots. Zero or negative supports no gain here. Null, class collapse or unavailable fits are not equivalence or absence of internal information. Observed t8 recognition is not necessarily prediction of a future unseen mistake. None of these outcomes establishes safety detection, general text-model inferiority or a mechanism. No adaptive follow-up, replication or search is selected by the outcome.

## Source interfaces and retained evidence

`matched_prefix_tasks.build_roster()` (alias `compile_roster`) constructs the exact schedule. `validate_roster` reconstructs and rejects any altered key, type, value, row or order. Full row keys are `observation_id`, `core_index`, `core_id`, `group_id`, `split`, `format`, `operands`, `expected_sums`, `strata`, `messages`, `schedule_sha256`, and `sequence_index`. `prediction_roster` projects only observation ID, core index, format and split after validation.

`oracle_response(text,expected_sums)` checks grammar/arithmetic only. `prefix_check(text,expected_sums)` and `prefix_features(core_index,prefix_text)` return exactly `completed_fields`, `completed_fields_normalized`, and `prefix_known_error`. `outcome_for_generation(row,generation,decoded_text,eos_token_ids)` validates the retained stop-stream rules and returns `status`, `label`, `reason`, `terminal_eos`, `capped`, and `censored`. Acquisition binds exact decoding, vocabulary, attempts and provenance; the pure task scorer cannot prove those occurred. Its incomplete status is `missing`/`not_measured`, not an inferred negative.

The acquisition module owns `compile_plan`, `validate_plan`, `prepare_inputs`, run/load and A183 manifest construction. Its plan accepts explicit provenance (model, tokenizer and chat-template hashes), geometry (vocabulary, hidden width, layer count and context limit), EOS IDs, runtime/protocol/test hashes and two prospective A183 comparator descriptors. In this pre-acquisition capture manifest, comparator `model_sha256` and `transform_sha256` bind the frozen algorithm/configuration descriptors; they do not claim hashes of future fitted coefficients or a future fitted PCA transform. The manifest is used for capture evidence, not to publish A183 paired predictions falsely attesting those future objects. Actual learned coefficients, the fitted transform and their hashes belong to the separate post-fit A186 prediction envelope.

The plan retains all rows and is reconstructed on validation. Its manifest records format and the task-strata hash; full counts remain in each bound row. `run_acquisition(model,tokenizer,*,plan,prepared,runtime_binding,directory)` and model-free `load_acquisition(directory,plan)` preserve complete or failed schedule receipts. `feature_views(directory,plan)` requires finished replay and the caller's separately observed normal acquisition return; it exports separate semantic packets, internal packets and labels. A failed root is retained without subsequent encoding.

Prediction owns separate strict feature packets, two fit artifacts, paired probabilities and fixed-cohort analysis. Its common-time packet includes exact rendered-prompt/prefix text, corresponding IDs, completed field count and the common error flag; semantic vectors and internal vectors occupy distinct named fields. Labels enter fitting/analysis separately, never the common feature validator. Row IDs are join provenance, not additional model inputs. Final source/test/protocol, roster, native rendering, model/encoder assets, runtime, endpoint, transform definitions and group manifests are bound before execution; fitted objects are bound after their actual returns. Synthetic qualification covers independent constructor enumeration, input-only strata, strict ASCII and EOS/cap edges, comma-boundary causality, type-preserving tamper rejection, common-time feature isolation, fixed denominators, missingness and lifecycle. This qualification does not itself supply acquired detector performance.

Raw prompts/renderings, logits, generated text/IDs, vectors, labels, row-level predictions and model assets remain in private evidence storage, outside public Git. Public source, invented fixture tests and this protocol contain no acquired outputs or private model paths.

## Operational ceiling and stopping

One target load is released before one encoder load, followed by at most two fits. Use four threads and CPU FP32. Require at least 64 GiB available before load and 64 GiB free scratch. Enforce sampled owned RSS at 64 GiB and an 8 GiB host available-memory floor; sampled thresholds do not reserve RAM or impose an instantaneous kernel allocation cap. The entire acquisition, encoding and fitting pipeline has a 24-hour deadline plus 60 seconds termination grace. Raw run storage is at most 8 GiB, with 32 GiB reserved for staged copies and archives.

There are at most `112*64 + 112 = 7,280` target forwards, including reached t8 captures; at most 224 semantic documents, eight chunks each and 1,792 encoder entries; and at most two fit attempts. No warmup, extra t0/reference calls, retries, precision fallback, downloaded asset or paid service is selected. These are conservative ceilings, not throughput promises. Any hard infrastructure, resource or interruption failure stops remaining operations and preserves explicit unattempted slots. A completed negative outcome, prefix error or class collapse does not change the acquisition schedule. Freeze, resource review and one-shot accounting precede the one authorized local run.
