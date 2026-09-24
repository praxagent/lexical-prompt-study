# A193: native special-token and continuation contract

This is a prospective tokenizer-only instrument qualification. It asks whether the pinned tokenizer supplies a complete declared special-token inventory and valid canonical assistant continuations through the selected API. It makes no scientific performance claim and does not replay any prior task roster, prompt, candidate, generation, score or consumed qualification.

## Fixed scope and fresh inputs

Load the local tokenizer for Meta Llama-3.2-3B-Instruct revision `0cb88a4f764b7a12671c53f0838cd831a0843b95` exactly once, with local files only and remote code disabled. Freeze the actual tokenizer assets, generation-EOS metadata, template, runtime, helper/runner/protocol sources and destination before execution. No pretrained language model is loaded or constructed. No forward, generation, encoder, fit, download or paid service is allowed.

Both contexts use this exact system message:

```text
Return exactly two comma-separated integers and no other text.
```

The two user messages, in order, are:

```text
Copy this ordered pair: 17,-6.
```

```text
Preserve the order of this pair: 17,-6.
```

Each context has the same two candidate texts, in order: `17,-6` and `18,-6`. These are fresh syntax fixtures with no correctness labels or preference scores. Use the native template date `24 Sep 2026`. Candidate text contains no special token. A closed assistant completion must append exactly the native string `<|eot_id|>` through the selected template. There are two open prompts and four closed prompt/candidate constructions. No fixture is replaced or modified after any native check.

## Inventory contract

Collect the tokenizer's named special IDs from `all_special_ids` and its backend-added entries from `added_tokens_decoder`. The latter's objects must expose an exact Boolean `special` flag; do not infer it from truthiness or reinterpret an arbitrary configuration dictionary as a live AddedToken object. The qualified special-ID set is the union of named IDs and backend-added IDs whose flag is true. Preserve both sources separately, their overlap and the combined set in a defensive canonical snapshot. Named special metadata alone is not claimed to enumerate all backend specials.

Reject Boolean, noninteger, negative or out-of-vocabulary IDs, malformed mappings or flags, and inconsistent snapshot contents. The valid ID bound includes added tokens and is authenticated from the pinned configuration; do not substitute a tokenizer's base-vocabulary-only size. Validate every declared generation EOS against the union. Resolve `<|eot_id|>` explicitly with `convert_tokens_to_ids`, require its ID to belong to both the declared generation-stop set and the combined special set, require encoding that literal alone to yield that single ID, and require decoding the single ID to return the literal exactly. Other EOS IDs need not appear in named metadata when they are valid backend-added specials.

Collect and validate the inventory before fixture operations and again afterward. Require identical canonical inventory snapshots. The reusable helper performs inventory validation only; native token identity, rendering, roundtrips and actual asset/runtime identity are responsibilities of this separate caller. Invented API tests must include the case of two named specials and three generation EOS IDs whose extra EOS is explicitly backend-special.

## Native continuation checks

For each context, separately encode the system, user and both candidate payloads with `add_special_tokens=False`. Require valid nonempty token sequences and no token from the combined special inventory. Do not deduplicate repeated payloads: this fixed control has eight payload encodes.

Render the open system/user prompt with `add_generation_prompt=True`. Separately encode that rendering and require exact equality to the direct `apply_chat_template(..., tokenize=True, return_dict=False)` result. For each candidate, render the messages with the assistant candidate and `add_generation_prompt=False`; require the closed rendering to equal open rendering plus candidate text plus `<|eot_id|>`. Separately encode it and compare exactly with the direct tokenized closed template. All six direct template calls explicitly set `return_dict=False` and must return a list of token IDs; they do not depend on the library's default return container.

The closed token sequence must start with the complete open-prompt sequence. Its remaining target must be nonempty, end in exactly one explicit EOT ID and contain no other special token. Decode the target and full sequence with `skip_special_tokens=False` and `clean_up_tokenization_spaces=False`; require equality to candidate-plus-EOT and the entire closed rendering, respectively. The target IDs for each fixed candidate must be identical across the two contexts. Require each open prompt to have at most 256 tokens and each target including EOT at most 16. These checks test the frozen native API contract; no model is asked to produce the candidate.

## Explicit operation accounting

The successful schedule has these exact caller-level operations:

| Operation | Count |
| --- | ---: |
| Selected tokenizer load | 1 |
| Inventory collections, each reading named IDs and backend-added metadata | 2 |
| Explicit EOT token-to-ID conversion | 1 |
| Separate encodes with special-token insertion disabled | 15 |
| Template calls returning rendered text | 6 |
| Direct template calls returning token IDs | 6 |
| Decode calls with both stripping and cleanup disabled | 9 |
| Language-model construction, model loading and forwards | 0 |

The 15 encodes are one EOT literal, eight payloads, two open renderings and four closed renderings. The nine decodes are one EOT ID, four targets and four full renderings. Direct template calls may internally tokenize, and metadata properties may call other library methods. Those nested operations are not independently counted by this caller-level ledger. Do not report these counts as all backend operations. Record an operation's entry separately from its successful return; an exception must not be counted as a completed check.

## Failure, provenance and interpretation

The preceding failed study must be independently closed before this new tokenizer attempt starts. Use one exclusive attempt, at most 600 seconds plus 60 seconds cleanup grace, without a resource queue. Check host-memory headroom before loading and retain observed process-memory peaks; the exact resource thresholds are bound in the reviewed runner before execution. Thresholds are sampled, not reservations. Any guard, binding, infrastructure or timeout failure ends this attempt. Preserve completed and unresolved checks, partial operation-entry/return counts, the exact stage, a runner-owned literal guard/failure code, and the original exception type. The code must come from a source-bound finite enumeration; never expose an arbitrary exception message, token payload or unexpected object representation. An unclassified library exception receives an explicit generic operation-exception code with its stage. No retry, fallback inventory, suppressed guard, different tokenizer revision or replacement fixture is allowed within it.

Check bound source, relevant assets and runtime before and after the attempt. Retain exact fresh renderings, token arrays and inventory snapshots privately, and publish only the approved aggregate checks, counts and whole-artifact bindings. Require zero language-model constructor/load/forward entries. Reject imports of architecture implementation modules that could instantiate a language model. Generic model configuration metadata or a tokenizer dependency importing Torch is not itself language-model execution: report actual imports accurately rather than claiming Torch was absent. No GPU initialization or computation is intended or needed.

A pass establishes only that this pinned tokenizer and these fresh fixtures satisfy the declared inventory and continuation contract. It does not identify an unretained earlier exception, validate language-model weights, establish answer preference, prove free-generation formatting competence or authorize reopening a failed attempt. A fail is an instrument result with unresolved subsequent checks. Any later application requires its own separately recorded prospective design and frozen sources.
