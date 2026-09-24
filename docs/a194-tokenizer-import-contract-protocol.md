# A194: guarded tokenizer import and continuation contract

This is a new, prospective tokenizer-only instrument. It asks whether the pinned AutoTokenizer satisfies the complete special-inventory and continuation contract when a generic model registry is distinguished from concrete model implementations. It does not retry A193, use its fixture payloads, score an answer or make a scientific performance claim.

## Scope and prerequisites

Use the local tokenizer for Meta Llama-3.2-3B-Instruct revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, with local files only and remote code disabled. No pretrained language model, encoder, fit, download or paid service is allowed. A193 must be independently closed before either new native phase. Reviewed pure tests precede a separately frozen cold-import check; that successful check, public source backup and a verified private prepared backup precede the selected tokenizer attempt.

Freeze each phase's sources, installed library bindings, runtime versions, limits, destination and exclusive claim before it starts. Each is one bounded attempt, without a resource queue or retry, at most 600 seconds plus 60 seconds cleanup grace. Both use the inherited sampled limits: at least 4 GiB available host memory before launch, at most 4 GiB owned RSS, and a 2 GiB available-host-memory stop floor. Sampling does not reserve RAM or impose an instantaneous kernel allocation limit. Bind the inherited scratch-space and raw-output limits in the reviewed runner. Stop on any guard, binding, infrastructure or resource failure; signal only the owned process session.

## Exact import and execution boundary

The only permitted `transformers.models.*` module containing a `modeling_` name component is the exact generic registry `transformers.models.auto.modeling_auto`. Authenticate its installed source path and whole-file SHA-256, prospectively pinned to `b439f15d7b0f1ed3e1339c0e71cea0ebc1bac4a87ff89596d39d78e4cd1b0fd4`. This exception is not a prefix or wildcard allowlist. All concrete architecture implementation imports remain forbidden, including similarly named descendants or lookalikes. Generic configuration definitions, documentation processing and Torch dependency imports are not themselves model execution.

Independently block AutoModel factory `from_config` and `from_pretrained` entries, PreTrainedModel construction/load/forward entries, and GPU initialization. An allowed registry import therefore does not authorize invoking its factories. The import-only mode additionally blocks tokenizer constructor and loader entries; selected mode permits the single selected tokenizer-load path. Guard counters describe observed prohibited entry attempts, not completed body executions or all possible nested library activity. Require zero prohibited entries for success and restore the owned import/profile guards before publishing success.

On a failure, retain the source-bound finite stage and guard codes, original exception type, partial operation counts, cleanup status and a safe public failure classification. Retain a blocked module's full name privately only after the guard validates its bounded dotted-identifier form. Do not serialize arbitrary exception text or object representations. An unknown module or library exception remains explicitly unknown; do not infer a historical import name from a later static diagnosis.

## Phase 1: cold AutoTokenizer import only

Start a fresh process under the pinned Python/runtime and corrected guard. Its separate frozen package contains source/runtime provenance and no selected tokenizer asset path or asset manifest. It may perform exactly one owned `from transformers import AutoTokenizer` import, recording the import entry and normal return separately. It must not call the factory, construct/load a tokenizer or model, read selected tokenizer assets, encode, decode, render a template, or initialize a GPU. Reading pinned installed library source and package metadata for provenance is allowed.

Require normal import return, unchanged source/runtime bindings, restored guards, zero tokenizer-constructor/loader and model/factory/GPU entries, and verified owned-process cleanup. Preserve a failed attempt and its partial accounting without rerunning it. This phase tests actual cold dependency integration only. It neither qualifies the selected assets nor tests the native continuation contract. The selected tokenizer freeze must bind the authenticated successful import-qualification receipt; a successful exit code alone is insufficient.

## Phase 2: fresh selected tokenizer fixtures

Load the selected tokenizer exactly once. Both contexts use this exact system message:

```text
Return exactly two comma-separated integers and no other text.
```

The two user messages, in order, are:

```text
Copy this ordered pair: 23,-4.
```

```text
Preserve the order of this pair: 23,-4.
```

Each context uses candidate texts `23,-4` and `24,-4`, in that order. These are syntax fixtures without correctness labels. Use template date `24 Sep 2026`. There are two open prompts and four closed prompt/candidate paths. No input, loader, revision or limit is replaced after any check.

Collect named IDs from `all_special_ids` and backend-added IDs from `added_tokens_decoder` whose live objects expose exact Boolean `special=True`. Preserve both sources and validate their union using the unchanged qualified inventory helper. Use the authenticated full vocabulary bound including added tokens. Reject Boolean, invalid or out-of-range IDs and malformed snapshots. Require every declared generation EOS in the union. Resolve `<|eot_id|>` explicitly, require its membership in that union and the declared stop set, and check its single-ID encode and exact decode. Collect the inventory again after all fixtures and require exact canonical equality.

For each context separately encode system, user and both candidate payloads with `add_special_tokens=False`; none may introduce any union-special ID. Do not deduplicate payloads. Render the open native template with `add_generation_prompt=True`, separately encode it, and compare with direct `apply_chat_template(..., tokenize=True, return_dict=False)` IDs. For each candidate, use `add_generation_prompt=False` and require the closed rendering to equal the open rendering plus candidate plus `<|eot_id|>`. Compare separately encoded closed text with direct template IDs. All six direct calls explicitly set `return_dict=False`.

Require the complete open-prompt IDs as the closed sequence's prefix. Its remaining target must contain ordinary candidate IDs followed by exactly one EOT and no other special ID. Decode target and full sequence with `skip_special_tokens=False` and `clean_up_tokenization_spaces=False`; require exact candidate-plus-EOT and closed-rendering roundtrips. Candidate target IDs must match across contexts. Each prompt is at most 256 tokens; each target including EOT is at most 16.

The successful selected phase has exactly 40 explicit caller-level wrapper operations:

| Operation | Count |
| --- | ---: |
| Selected tokenizer load | 1 |
| Inventory collection | 2 |
| Explicit EOT token-to-ID conversion | 1 |
| Separate encode | 15 |
| Rendered template | 6 |
| Direct tokenized template | 6 |
| Decode | 9 |

The encodes comprise one EOT literal, eight payloads, two open renderings and four closed renderings. Decodes comprise one EOT ID, four targets and four full renderings. Record entry and normal return separately. Nested library calls are not independently counted by this ledger. The separate cold-import operation is not one of these 40 selected-phase calls. Language-model construction/load/forward and GPU entries must remain zero.

## Evidence and interpretation

Authenticate relevant tokenizer/config assets, template, runtime and sources before and after the selected attempt. No model-weight authentication is inferred from checking tokenizer assets. Retain exact fresh renderings, IDs, inventories and bounded diagnostics privately; expose only approved aggregates and whole-artifact bindings. Require successful normal return, complete expected checks, verified postflight bindings and owned-process cleanup before qualification.

A cold-import pass establishes dependency integration under this guard. A selected-phase pass establishes the declared inventory and continuation properties only for the pinned tokenizer and fresh fixtures. Neither identifies A193's unretained blocked import, validates model weights, establishes answer preference or repairs a closed study. A failure leaves subsequent checks unresolved. Any later scientific use requires its own new prospective design, fresh inputs and frozen adapter; A194 selects no such experiment.
