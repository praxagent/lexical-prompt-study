# A158: preserve exact version bytes when a library returns a string subclass

Prospective implementation repair only. A156's scientific/human-review protocol
and all A157 protections remain in force. This annex does not authorize new
inference, outcome selection, fitting, automated ratings, confirmation access,
publication, or a relaxed runtime pin.

## Evidence and narrow correction

CPU35 stopped before reading selected payloads. Its rejected qualification is
`e3677ad9892c2819d95ab2d5961ba370c2545ed6d676e62d819f26087c68457c`.
The recorded hash of the rejected PyTorch version value equals SHA-256 of the
exact required bytes `2.13.0+cu130`:
`571c2e7c52dcd93d2994e7ae649a0bcd716f4f217760076cd0d0f38ccf020d70`.
The observed PyTorch initializer hash matches the locally inspectable source,
which imports its version from a module defining `TorchVersion(str)`.
A157 incorrectly required `type(value) is str` at observation time. The earlier
qualified probe accepted string subclasses. This is a type-handling error, not
evidence for changing the version pin.

At the observation boundary only, accept an existing `str` instance, including
a subclass, and extract its unchanged base-string content with
`str.__str__(value)`. Continue to require a plain built-in string in the saved
receipt, the existing bounded version syntax, and the exact pinned bytes.
Do not apply permissive `str(value)` conversion to arbitrary objects. Do not use
a subclass's overloaded equality to decide runtime acceptance. No suffix
stripping, semantic-version comparison, installation, download or fallback is
allowed. Bytes, non-string objects, wrong build suffixes and misleading subclass
`__str__`/`__eq__` overrides must fail or be handled by their actual base content.

Both accepted and rejected qualification receipts remain durable before
tokenizer construction or selected-payload reads. Test the real source-defined
TorchVersion class without importing a model, plus synthetic adversarial types,
and exercise the generated wrapper/worker integration before freezing.

## Exact inherited authorities and closed operation

- A156 protocol: `b5d73d4290793242b9dae92322b2e99274f6b40120a3db1bb12f2dea3c35ce58`.
- A157 implementation annex: `586821041f55c15fa21612b8728be244ffbf1e4b919520f0e9fbe0c9dd64e1b3`.
- CPU35 failed closure: `e2b53ac5207fbfb865b941719e8bdc3acf24daf69ddb254bac2b6ea8bfc97b15`.
- CPU35 teardown: `54133398090ea937204f208f303ca6e85035cf87f995c86ffd8c39183ffaed6d`.
- CPU35 request: `ae224a8e8394cc189f2fcfa733a71684453fc6361816526aa4152527e81817af`.

CPU35 was deleted and independently verified absent after 114.977160 seconds;
rounded-up compute cost was $0.01. The new closed base is **$25.69**, before the
unchanged cumulative storage reserve. Preserve every consumed CPU30/CPU35
source, freeze, request and receipt. Never replay those namespaces.

This annex replaces only the current value of `runtime_repair_plan_sha256` in a
new request; `plan_sha256` remains the unchanged A156 protocol. Bind this annex
and its A157 parent, the CPU35 qualification, closure, teardown and request into
the fresh source/operational authority. Exact canonical case and contract
equality against CPU35 is mandatory. All other scientific provenance must match;
only new compiler/source identities and this repair-annex identity may differ.
Preserve all 87 cases, 348 horizon slots, 287 distinct prefixes, both placements,
EOS aliases, the fixed timing sample and every token/text/artifact hash.

## Fresh, bounded execution

Use `recover37_a158.py` and a separately tested/frozen lifecycle in the new
single-use namespace:

- Remote: `/workspace/runs/continuation-a158-human-packet` and its `-inputs` sibling.
- Local: `private/human-audit/a158` and `private/human-audit/a158-request`.
- Exact pod name: `lexical-continuation-a158-human-packet`.

After main-agent review, passing synthetic tests and fresh source/request/lease
freezes, one CPU-only retrieval may run under the existing user-approved $70
round ceiling. Keep the **$0.50 operation cap**, **$0.16/hour CPU maximum**,
**$0.10/hour storage reserve**, **3,600-second allocation ceiling**,
**3,300-second worker ceiling including startup**, and **180-second final
reserve**. Retain the fixed twice-worst throughput gate, byte caps, exact owner
checks, independent guardian, opaque transfer and immediate teardown/cost closure.
No unrelated pod may be accessed or changed. Preserve the study volume.

Do not introduce another blind retry. A further failure must first be explained
from its retained receipts and reviewed locally. Once retrieval is verified and
compute is off, prepare the unchanged A156 human-review packet through its own
source/input freezes. Actual human ratings remain a separate human action.
