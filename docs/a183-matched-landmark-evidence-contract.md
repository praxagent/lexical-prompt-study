# A183: matched-landmark evidence retention contract

A183 prepares a small, model-free evidence validator and private writer. It qualifies retention and alignment on invented fixtures, not a detector, acquisition population or target-model capture. No target model, real tokenizer, encoder, fitting procedure, old corpus or held-out panel is used in this phase. A later acquisition must make its own scientific and resource choices before execution.

The intended comparison gives a text predictor and a text-plus-internal predictor the identical model-visible prompt and emitted prefix at one declared time. Internal states of a fixed deterministic model are functions of that input; a later comparison concerns the performance of specified representations and learners, not information theoretically unavailable from text. A183 does not claim that any observed internal feature adds predictive value.

## Scope and minimal artifacts

The [landmark_evidence module](../src/lexical_prompt_study/landmark_evidence.py) validates a fixed manifest, observation/landmark evidence, outcome references and paired predictions; it writes immutable private bundles and constructs restricted predictor views. Generation, capture hooks, tokenization, representation extraction, fitting, metrics, scheduling and recovery are outside this module. It must not silently invoke those operations to repair evidence.

| Artifact | Required information and separation |
|---|---|
| Manifest | Schema `matched-landmark-v1`; the full Cartesian product of the ordered observation list and sorted unique landmark list, with both comparator slots for every pair; fixed group/split assignments; model, tokenizer, template and source bindings; declared generation, decoding, boundary and representation settings; endpoint and fit/calibration lineage. Later scientific choices are explicit values, never inherited historical defaults. |
| Observation and landmark | Exact prompt and emitted-prefix bytes and IDs; generation/attempt provenance; independent prefix and capture availability; lossless declared internal bytes and complete boundary metadata when available. Retain the sampled output ID stream, including terminal EOS when sampled, privately as separate generation evidence, never as predictor input. |
| Outcome | Instrument and target-definition bindings, horizon, applicability, known label or explicit unknown, disagreement/uncertainty and censoring. Outcome availability does not redefine an earlier landmark. |
| Paired predictions | Every planned comparator slot for every fixed observation/landmark, with shared target/group/split identity; model, transform, training, calibration and input bindings; finite probability or an explicit missing reason. |

Private raw bytes, tokens, internal values, outputs, labels and row predictions stay outside the public repository. Public examples and qualification data are synthetic. Safe summaries contain counts and whole-artifact digests, not payloads or per-item identifiers.

## Exact common-time input

Retain the original message structure, exact rendered prompt UTF-8 bytes, exact native prompt token IDs, and their separate content hashes. Bind the model, tokenizer, chat template, renderer and source versions. Do not reconstruct prompt IDs by retokenizing a prettified string or replace the native rendering with concatenated message text.

For this new contract, t is the count of observed generated token IDs before terminal EOS; EOS is not counted. This is a prospective convention, not a reinterpretation of any old token-eight cohort. Other non-EOS IDs still count as IDs, not words or Unicode characters. A valid declared generation stream has at most one terminal EOS and no tokens after it. The EOS ID set and stop metadata are explicit.

At t=0 the prefix is empty and the boundary is the last prompt token, absolute index `len(prompt_ids)-1`; the observed prompt can make t0 available even when generation is unattempted. At t>0 the boundary is `len(prompt_ids)+t-1`. A capture forward's declared complete input must be exactly the prompt followed by the first t observed non-EOS IDs, ending at that boundary. A longer completed answer, later suffix or terminal EOS is forbidden in that capture input. The boundary state predicts the next token after the observed prefix. Sampling the last prefix token does not itself prove that it was subsequently fed to the model for capture.

Retain the exact prefix IDs and the tokenizer-returned decoded UTF-8 bytes with `skip_special_tokens=False` and `clean_up_tokenization_spaces=False`. IDs are authoritative for exact token identity. Arbitrary generated prefixes may end inside a byte-token sequence and decode with replacement characters; do not require decode/re-encode identity, normalize Unicode or whitespace, or discard such rows as malformed text. A separately derived display view cannot replace the retained bytes or IDs.

A pure validator can check exact declared IDs, byte hashes, lengths, slicing and identities. It cannot prove the tokenizer actually returned the submitted bytes or the model consumed the submitted input. Real native correctness and capture-boundary execution require a later, separately qualified adapter audit.

## Internal payload and provenance

The initial serializer supports only finite IEEE-754 float32 source values stored as little-endian float32, without conversion. Source and storage dtype must both declare float32. Shape, byte length and value finiteness must agree; reject unsupported source dtypes rather than silently downcast or relabel precision. This implementation restriction does not select a future model, precision or scientific readout. Any wider dtype support requires its own explicit lossless qualification.

Bind each capture to the observation, landmark, attempt/owned-forward receipt, prompt and prefix hashes, complete capture-input IDs and length, model/source versions, zero-based layer and site, absolute token index, shape, dtype, endianness and content hash. V1 fixes declared attention to all ones, explicit zero-based sequential position IDs and no cache; their arrays must cover the exact complete input. Preserve the full declared representation before any lossy transformation. A derived feature references that exact source plus its fixed transform; it cannot masquerade as the source vector.

Exact candidate-free prefix-only input is the declared initial boundary. A future streaming-cache adapter would need separate proof of equivalent causal history and accounting. The storage module provides neither a capture adapter nor execution truth merely because metadata is internally consistent.

## Availability, outcomes and fixed coverage

Keep prefix availability, capture availability, generation stop reason and outcome availability distinct. Reaching t is decided from the observed prefix, not survival to a later outcome horizon.

| Condition | Required accounting |
|---|---|
| EOS before t non-EOS IDs | Landmark unreached; no shortened substitute prefix or internal vector. |
| EOS after exactly t or more non-EOS IDs | Landmark remains reached; later EOS is excluded from that predictor view. |
| Cap at or beyond t | Landmark remains reached; cap is not EOS and does not establish a complete safe output. |
| Interruption or infrastructure failure | Preserve reached prefixes when valid, with explicit separate missing capture/outcome reasons. Do not recode the stop as early EOS. |
| Valid prefix, unavailable capture | Text input remains available; internal input and corresponding prediction may be missing. Retain the planned paired row. |
| Unknown or disputed outcome | Preserve the label uncertainty and target binding without removing otherwise valid input evidence. |

The manifest enumerates observations and landmarks, not just a total count. Every observation-times-landmark slot belongs to the retained bundle and prediction cohort, including fit and calibration groups. Exports and validators reconcile every planned slot, including unreached landmarks, unattempted work, failed captures, failed fits and unknown labels. Missing values are typed nulls with reasons, never zero vectors, zero probabilities or absent rows. Retained bytes with an invalid or absent final commit are orphans, not complete observations. No automatic recomputation, substitute source, favorable subset, later-survivor cohort or observed-subset paired result is allowed.

V1 generation statuses are `completed`, `interrupted`, `infrastructure_failed` and `missing`; a completed stop is either sampled terminal EOS or a full cap with no terminal EOS. Prefix status is `available` when enough non-EOS IDs exist, `unreached` for a completed early EOS, otherwise `missing`. Capture status is independently `completed`, `infrastructure_failed`, `missing` or `unavailable`; an unavailable prefix requires an unavailable capture and no payload. Available text may therefore coexist with an absent capture.

V1's outcome is a submitted binary instrument label, not an implemented judge. Status is `known`, `unknown`, `not_applicable` or `missing`. A known submitted label is exact integer 0 or 1 with declared applicability true; it does not mean full-horizon truth is established. Censoring is defined mechanically as **no terminal EOS and observed non-EOS length below the declared horizon**. Retain the label and censoring independently, including a submitted negative (0) with censoring; the module must not discard or coerce instrument evidence. Such a negative does not establish absence over the declared horizon or safety. A completed EOS or reaching the fixed horizon removes this particular censoring flag, without establishing indefinite safety. Unknown reasons distinguish uncertainty, disagreement and censoring; missing means not measured. Disagreement is retained as a Boolean or null. Instrument, target and uncertainty-policy hashes bind declarations; the module does not execute them, prove applicability, or adjudicate label truth. A later prospective endpoint, uncertainty and adjudication policy must decide whether each retained label is usable.

A future study must freeze the comparison cohort and missing-prediction/outcome policy before fitting. A183 validates the evidence required for that policy; it does not compute a metric, choose imputation or turn partial availability into a complete-cohort result. Automated judgments may have explicit uncertainty and disagreement; this contract imposes no human-rating prerequisite and does not convert a proxy label into safety or task-success ground truth.

## Predictor isolation and paired predictions

`predictor_view` accepts only a bundle already returned by successful `validate_landmark` or `load_landmark`. Its dataclass type alone does not establish that a caller-constructed value is valid. This lightweight view does not revalidate an arbitrary forged bundle; writers and the complete-cohort validator do revalidate it against the manifest.

The return value contains only `features` and `features_sha256`, with no provenance or join envelope. Text features are exactly `rendered_prompt_utf8`, `prompt_token_ids`, `prefix_utf8` and `prefix_token_ids`. The internal comparator receives those same fields plus `residual_fp32le` and `residual_shape`. Text views require an available prefix; internal views also require a completed capture, otherwise construction raises. Typed missingness remains in the separate retained bundle/prediction slots rather than entering the feature packet. No observation/group ID, bundle/outcome hash, later output, final length, stop reason, label, applicability, uncertainty or future readout enters either feature packet; internal values never enter the text-only packet. Feature hashes bind only the allowed content.

Splits are exactly `fit`, `calibration` and `evaluation`. Every variant, placement, frame and landmark derived from one request core has the same preassigned independent-group identity and split. Each comparator's fit/calibration group lists must exactly equal the corresponding manifest split groups; neither may select a convenient subset. Reject shared-core leakage or disagreement between observation, prediction and split manifest. Group/stratum names cannot be inferred from labels after acquisition. Fitted transforms and thresholds carry training/calibration bindings; reject declared fitting or calibration cohorts that include evaluation groups under a manifest declaring disjoint roles. This checks declared lineage, not the truth of an unseen training execution.

Require exact keyed comparator coverage and verified row alignment. All fit/calibration rows retain both probability-null slots with eligibility and null reason `not_evaluation`; they are not silently omitted or treated as out-of-fold evaluation predictions. Evaluation eligibility depends on prefix availability and, for the internal comparator, capture availability, not on whether the outcome is known. Each probability is a finite real number in [0,1], or null with a reason and fit status; Boolean numbers, NaN and infinity are invalid. Retain probabilities themselves, not only aggregate metrics or hashes. Bind the actual model and transforms, common text input, optional internal input, target and group/split. A comparator may have a known prediction while its partner is unavailable, but the pair remains explicitly incomplete. Each frozen comparator has one consistent fit status across every planned row, including non-evaluation and unavailable-input slots. Fit status is `fitted`, `failed` or `not_attempted`; an eligible row with null probability records `prediction_failed`, `fit_failed` or `fit_not_attempted` respectively. A non-evaluation or unavailable-input slot instead records that eligibility reason. A row-specific inference failure preserves comparator status `fitted` and records `prediction_failed`; it does not relabel that model as globally failed. Input feature bindings remain available even on non-evaluation rows when the corresponding view exists. Key permutation may change serialization order only if the exact join and all bindings remain valid.

## Small validation and publication API

The small API is:

- `validate_manifest(manifest)` returns a defensive validated manifest copy.
- `validate_landmark(manifest, observation, generation, prefix, capture_metadata, capture_bytes, outcome)` returns an immutable bundle.
- `commit_landmark(directory, bundle, manifest)` and `load_landmark(directory, manifest)` publish or revalidate that bundle.
- `predictor_view(bundle, comparator="text")` returns only features and their hash; comparator is `text` or `text_internal` and its validated-bundle precondition applies.
- `validate_prediction_pairs(manifest, bundles, rows, keyed=False)` returns an immutable retained packet. Bundles always follow manifest observation/landmark order. Default prediction rows must have that order; `keyed=True` requires a unique complete key join and restores manifest order.
- `commit_prediction_pairs(directory, packet, manifest, bundles)` and `load_prediction_pairs(directory, manifest, bundles)` publish or revalidate the full packet.

There is no automatic fitting, scoring, recovery or scheduling API.

## Immutable publication and finite synthetic qualification

Validation copies mutable buffers defensively. Writers require a new absolute destination directory, use private file/directory creation modes, reject symlinks and special files, and refuse reuse of an existing or partial directory. They write and fsync temporary single-file contents, publish each destination with an exclusive link, and flush directory state. A claim and child files precede the final hash-bound receipt; no destination is overwritten. They reject overwrite, duplicate keys, duplicate observations and unbound extra artifacts. Callers must choose private storage outside the public repository.

This is receipt-last publication across multiple paths, **not an atomic multipath transaction**. A failed/interrupted write may leave partial children or an incomplete receipt; their existence does not establish successful durable publication. A successful commit return is the writer's completion boundary. Loading independently checks exact inventory, canonical JSON, every child hash and full semantic revalidation before accepting evidence. Never accept an in-progress directory or receipt based solely on its presence, and do not resume or overwrite partial publication. A completed missingness record is legitimate when its schema explicitly declares absent payloads; a success record missing a required payload is not. Exact archive verification must include retained bytes and paired predictions, not only their digest receipts.

Finite invented fixtures must cover t0 and positive-t boundaries; Unicode, whitespace and replacement-character preservation; exact prefix slicing; future-token insertion; identity swaps; byte/dtype/shape corruption and nonfinite values; mutable buffers; early EOS, post-landmark EOS, caps and interruptions; distinct missing prefix/capture/outcome cases; child-publication failure before final commit; symlink/overwrite rejection; group leakage; duplicate, reordered or missing predictions; and Boolean/nonfinite probability rejection. Changing future output or labels must leave the predictor content unchanged, while changing an observed input changes its binding. Paired export must preserve every planned slot and exact probabilities under a verified key reorder.

Qualification success establishes that the module rejects or preserves these declared synthetic cases. It is not proof of real model capture, native decoding, label validity, generalization, calibration or useful detection. A later capture adapter needs its own causal-boundary and cleanup tests before any acquisition decision.

Still unselected are the acquisition population and number of independent cores, landmark count, model/precision, layer/readout, semantic challenger, endpoint/instrument/horizon, uncertainty policy, fit budgets and compute/storage schedule. No encoder search, acquisition, fit, old-data recovery or paid service is selected by A183. The only selected work is implementation, synthetic qualification and review of this retention apparatus.
