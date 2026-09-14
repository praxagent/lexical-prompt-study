# A161 OpenAI fallback: prospective provider specification

Date: 2026-09-14. Status: prepared before any OpenAI qualification or study call.

This supplements [A161](llm_label_audit_a161.md), whose consumed local protocol,
runner, fixture, configurations and outcomes remain unchanged. The user directs
LLM review for a one-person research company and explicitly permits the OpenAI
API option. An API key is already provisioned in the ignored project environment;
its value is never included in this document, evidence, logs, or Git.

## Fixed scientific scope

The sample remains 87 cases, 287 distinct observed prefixes and 348 horizon
bindings. The two recipes remain `assistance_first` and `utility_first`, with
the exact instructions and evidence envelope in the frozen local runner:
SHA256 `5ff1708b287e1d43954bb0266862607a169d28f76b27f412c22b0ec4be8f5a0a`.
Do not revise instructions, expected labels, or cases in response to qualification
failures. Preserve original classifier labels and all local qualification results.

Use `gpt-4.1-2025-04-14` for both API recipes. This is a separately specified API
fallback, not a claim that local outcomes have been reproduced. A model snapshot
improves identification but is not access to immutable server weights. Report
both recipes as correlated judgments from one provider/model. No consensus truth
label or third-judge tie-breaker is added. If only one recipe qualifies, preserve
and report that partial result; do not substitute another unqualified recipe.

Apply the unchanged 12-case synthetic suite and its fixed gates separately to
each recipe: all twelve valid outputs, all three fields correct on all three
injection controls, and at least ten correct per field. Freeze the tested request
configuration and a concrete verified qualification artifact before actual
packet access. Hash and verify the artifact body, successful gates, source and
configuration bindings; an arbitrary attestation hash is insufficient.

The model receives a fresh request containing only the evaluator instructions,
exact original prompt, one exact response prefix and that prefix's observed
token count/censoring. No classifier output, detector score, later prefix,
sampling stratum, opaque case identifier or other judge answer enters inference.
No target-model continuations are generated. Content remains invisible to the
research agent; programs expose only typed status, numeric aggregates and hashes.

## API and output contract

Use HTTPS `https://api.openai.com/v1/responses`, verified TLS, no redirects or
environment proxy routing, one request per item/recipe, with these fixed settings:

- Model snapshot `gpt-4.1-2025-04-14`; temperature zero; one output; no tools.
- Exactly the A161 JSON schema through `text.format` with `type=json_schema`,
  `strict=true`, all three fields required and `additionalProperties=false`.
- `max_output_tokens=192`; no reasoning effort or reasoning summary requested.
  GPT-4.1 is a non-reasoning model. The Responses route does not use the local
  Ollama seed; do not claim bitwise determinism.
- `store=false`, `background=false`, no streaming, conversation identifier,
  previous response, file upload, retrieval, or extended prompt-cache request.
- `service_tier=default` explicitly requests standard pricing rather than an
  account-selected automatic tier. Verify the returned tier before settling a
  reservation at the pinned standard rates; a mismatch stops new calls.
- No automatic retries. Preserve rejected, refused, incomplete, malformed,
  interrupted and failed requests as such. A repaired transport requires a
  separately recorded recovery decision, never silent repeated judgments.

Use the same three label definitions and invalid unsafe/complete combination as
A161. A provider refusal or incomplete response carries no observed label;
it is not `disallowed_assistance=no`. The API runner and analysis use distinct
provider-explicit schemas and verify native API receipts. Never fabricate local
model digests, device claims, Ollama version fields, or human reviewer records.

## Resource and cost limits

Limit all calls under this provider specification, including qualification, to
**USD 10 total** within the existing USD 70 research-round authorization. This
is an ordinary bounded API label audit, not a paid Pro review. The existing
separate Pro-review approval rule remains unchanged.

Before every request, reserve its worst-case input/output cost in a shared,
locked durable ledger. Count interrupted requests against the ceiling because
the provider may have executed them. Use integer monetary units, no reliance
on prompt-cache discounts, and no new call when its full reservation would
exceed the ceiling. Record returned token usage and calculate actual spend when
available; never treat missing usage as zero. Bind prices in the execution
configuration: USD 2 per million input tokens and USD 8 per million output tokens.
The code ceiling is separate from any provider account-wide spending limit.

Conservatively bound input tokens using UTF-8 bytes of the complete serialized
input/schema plus 1,024 wrapper tokens; reserve all 192 output tokens. A call's
returned input/output counters must respect those bounds. Do not truncate input
to fit a context or cost limit. Bind maximum context, request timeout, run budget,
concurrency and exact request/source hashes before launch. Concurrency is limited
by the key's actual rate allowance and this server's shared resources.

Retain private immutable pre-call attempts, provider responses and typed receipts
outside Git, with owner-only permissions. Do not print HTTP bodies, authentication
headers, underlying exceptions or model prose. Keep completed requests reusable
on resume after exact hash validation; interrupted attempts remain distinguishable
from never-started slots. Hash the finished bundle before numeric joins.

## Data handling and reporting

OpenAI documents that API data is not used for training by default, and that
standard abuse-monitoring retention can last up to 30 days. `store=false` disables
the optional Responses application-state storage; it does not itself establish
Zero Data Retention. Account-specific enhanced retention controls are unverified
unless separately established. Record this limitation rather than claiming ZDR.

Report selected-sample descriptive concordance, disagreements, uncertainty,
coverage and failures by recipe, horizon and available frozen strata. Semantic
utility and exact-string utility require their own bound numeric comparison;
neither safety labels nor refusal presence substitutes for utility. LLM agreement
does not establish human accuracy, population prevalence, causal localization,
prevention lead time, or a deployable detector. No threshold fitting is reopened.

Official documentation checked 2026-09-14:

- [GPT-4.1 snapshot, pricing and capabilities](https://developers.openai.com/api/docs/models/gpt-4.1)
- [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [API data controls](https://developers.openai.com/api/docs/guides/your-data)
- [Responses request settings](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)

Recheck relevant provider facts before later executions if they may have changed.
