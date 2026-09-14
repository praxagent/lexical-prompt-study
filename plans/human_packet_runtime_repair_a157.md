# A157: repair runtime qualification before selected human-packet retrieval

Status: prospective implementation-repair annex for main-agent review. This
document authorizes no allocation by itself. Source, synthetic tests, exact
inputs, lifecycle closure and main-agent review must pass before a new operational
freeze can be issued. No outcome reselection, new inference, detector fitting or
human review is performed by this plan.

## 1. What failed, and what has not been established

The single-use A156 CPU30 operation stopped at `recover30_runtime_package`, before
reading selected response payloads. Its CPU compute has been closed after
78.088558 seconds from allocation to verified absence; its rounded-up compute
estimate is $0.01. The resulting closed compute base is
**$25.68**, before the continuing storage planning reserve. Preserve the failed
execution, bounded operational events, ownership, teardown and cost-closure
receipts, the consumed source/compiler/operational freezes, and any partial
namespace exactly as they are. The next freeze must bind their complete hashes;
do not overwrite them or replay CPU30. Exact closure SHA-256:
`853ce60bdbfae64334eae7b2fe5f05c26d20416ca6c42b2357208d916f9c8cfe`.
Exact teardown SHA-256:
`6b3206072dea2e51f21fbe758f52a25611a70226e74e33f78e1e5964cbbd0761`.

The source audit identifies a mismatch in **what the check measures**: A156 uses
`importlib.metadata.version(...)`, whereas the prior qualified CPU28 values were
recorded from imported modules' version attributes. Distribution metadata and an
imported module's version string are distinct observations and may differ, for
example in build suffixes. A generic `runtime_package` failure does not identify
which package failed or prove that this semantic mismatch was the sole cause.
Do not assign a package-specific cause without its recorded evidence. A157 makes
both observations explicit and tests the acceptance rule prospectively.

This is an implementation repair, not an experiment redesign or permission to
relax the runtime until a run succeeds.

## 2. Preserve the scientific and human-review contract

The governing human-review protocol remains A156, SHA-256:

`b5d73d4290793242b9dae92322b2e99274f6b40120a3db1bb12f2dea3c35ce58`.

Bind this annex as an **additional** provenance input named
`runtime_repair_plan_sha256` in the compiler freeze, request provenance,
operational execution and qualification/bundle lineage. Do not replace the A156
`plan_sha256` identity. Preserve the A155 sampling freeze and selected set:

- Sampling freeze:
  `81739caf1ae5e3112bb3628675ffdd2cc62dd7ceb78b573b2d5c7d9e402cbfbe`.
- Selection:
  `f714051f957cf5817a6b8b020eb35e16ff82e8dc0c5c8c19af672898d70fdcbd`.
- Exactly 87 selected cases, 348 requested case–horizon slots, and 287 distinct
  within-case prefix reviews; seven original-EOS cases and 80 continued cases.
- Requested horizons 128, 256, 512 and 1,024 generated tokens, with the original
  EOS mappings, saved token IDs, token/text hashes and generation-receipt lineage.
- The same fixed prospective timing sample and twice-worst-observed throughput
  gate; no replacement of inconvenient, missing, slow or mismatched cases.

Only implementation-source identities, operational namespace references and
explicit repair provenance may change in the new compiled request. A numeric-only
equivalence verifier must confirm the complete case set, review IDs, case order,
source-artifact identities, original/terminal bindings, prefix hashes, slot maps
and EOS aliases against the consumed A156 request. Report only counts, pass/fail
and hashes. A mismatch stops the operation; it does not trigger reselection.

All A156 presentation and interpretation rules remain in force: exact text,
earlier-prefix judgments locked before later context, per-rater isolation,
exposure accounting, independent ratings where feasible, private alias mappings,
and no population-prevalence, first-harm or prevention claim from incomplete or
contaminated annotations. No reviewer access is granted by a successful export.

## 3. Freeze one runtime criterion and record both version channels

Use the same retained offline interpreter, tokenizer files and imported-module
version requirements as the previously qualified runtime:

| Component | Required observation |
| --- | --- |
| Python | `3.12.3` |
| Interpreter SHA-256 | `a92f0f95e883390c7256b2e441484aac06b1002dbe1d924141a77c8d82f96223` |
| NumPy imported-module version | `2.5.1` |
| PyTorch imported-module version | `2.13.0+cu130` |
| Transformers imported-module version | `5.14.1` |

The interpreter path remains
`/workspace/runs/jlens-signal-falsification-a127/.venv/bin/python`.
Preserve all four tokenizer file SHA-256 values frozen by A156 and the exact
historical decoding behavior. Record the tokenizer backend distribution and
imported-module versions as separate observations when available; do not invent
an exact historical backend pin that was not previously established. Backend
observations remain subject to the frozen source contract and all 348 exact
prefix token/text identity checks.

Before reading **any selected response payload**, execute a bounded runtime
qualification step that:

1. Binds the exact owned pod, allocation-origin clocks, worker/source hashes,
   request hash, A156 protocol hash and this repair-annex hash.
2. Verifies the interpreter identity and exact tokenizer file inventory/hashes.
3. Records, separately for each relevant package, its distribution-metadata
   version and imported-module version. It must preserve both strings rather
   than substitute one for the other or strip build suffixes.
4. Applies the exact imported-module criteria above. Distribution metadata is
   recorded evidence, not an undeclared replacement acceptance criterion.
   Missing imports or required observations fail closed; there is no install,
   download, version-search or alternate-environment fallback.
5. Atomically persists a private qualification receipt containing the complete
   observations, the exact criterion, and an explicit accepted/rejected result
   **before** content access. An error produces a rejected receipt when storage
   permits, with fixed error enums and a message hash, never raw tracebacks.
6. Allows the selected-member inventory/retrieval phase only after the accepted
   receipt is durable and hash-bound into execution, checkpoints and the final
   bundle. A rejected qualification cannot lead to a content read.

No model is constructed or called. Keep GPU visibility disabled and enforce
offline-only operation. Importing the pinned libraries for version qualification
is not permission to initialize CUDA, load model weights, run inference, invoke
an automated reviewer or call the OpenAI API.

## 4. Fresh namespace, same bounded operation

Use a new single-use namespace:

- Remote: `/workspace/runs/continuation-a157-human-packet`.
- Staged remote inputs: the same path with `-inputs` appended.
- Local operation: `private/human-audit/a157`.
- Local compiled request: `private/human-audit/a157-request`.
- Exact task pod name: `lexical-continuation-a157-human-packet`.

No CPU30 namespace may be overwritten or automatically resumed. The provider
ownership boundary remains the retained study volume plus the exact newly
created pod ID, with the complete pre-existing inventory recorded. Inventory
must show prior task compute closed before creation. Never mutate an unrelated
pod. Do not retry an uncertain create; reconcile its exact identity and close it
under the tested ownership procedure.

The proposed new operation has a **$0.50 total cap**, within the existing **$70
cumulative round ceiling**, starting from the closed compute base **$25.68** plus
the storage reserve accumulated since `2026-09-05T16:29:04.398900+00:00`.
Require CPU compute at no more than **$0.16/hour**, with a **$0.10/hour storage
planning reserve**. No GPU is authorized. At those maximum rates, a full hour is
budgeted at $0.26; record actual rounded-up allocation-to-verified-absence cost,
not the $0.50 cap, and do not describe the estimate as a reconciled invoice.

Retain the 3,600-second allocation ceiling and 3,300-second worker ceiling
measured **from allocation**, including startup and runtime qualification. Keep
the independent exact-owner guardian alive before creation, startup and
no-progress limits, the 180-second final transfer/teardown reserve, bounded
opaque transfer, exact-ID deletion and independent absence verification.
Preserve the A156 source/member/read/export byte caps, including at most 16 MiB
of selected raw payloads and a 32 MiB complete transport bundle. Preserve atomic
per-case checkpoints and failure receipts; do not claim automatic replay.

Any failure must leave an auditable closure and real cost record, including an
uncertain-create path without an ordinary controller execution receipt. A data
or runtime error cannot erase spend or turn a failed packet into a success.
Retain the sustained volume. No additional paid job starts until this operation
has either completed or been independently closed.

## 5. Tests and freezes before the next paid attempt

Before compilation or allocation, main-agent review must confirm:

- New source files preserve consumed A156 files and bind this annex plus the
  unchanged A156 review protocol.
- Synthetic tests exercise distribution/module version disagreement, exact
  module acceptance, wrong-module rejection, missing imports, rejected-receipt
  durability and zero selected-payload reads on every failed qualification.
- Actual generated owner wrappers, runtime entry point, inherited guardian and
  all duration bounds execute under synthetic tests; test uncertain-create
  closure and no unrelated-resource mutation again.
- The compiled request passes exact numeric equivalence against A156 before a
  fresh operational freeze. Freeze source/input/runtime/namespace/cost pins and
  a new expiry before any provider allocation.
- Success and failure receipt schemas, private transfer verification and cost
  accounting pass local tests before the pod exists. Emit only fixed statuses,
  counts, byte/time totals and hashes to the agent.

After successful retrieval and compute closure, independently verify the local
bundle, all selected lineage and prefix bindings, and the accepted pre-content
runtime receipt. Only then may the unchanged A156 blinded-review preparation
continue through its separate tested UI and human-access gates. A157 changes no
classifier labels, detector thresholds, confirmation panels or scientific claim.
