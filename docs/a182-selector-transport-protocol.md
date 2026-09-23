# A182: prospective fixed-site selector transport protocol

This protocol specifies one conditional causal-transport question: can changing a donor selector change a recipient's preference between its own mapped values at one fixed residual intervention site, when donor and recipient answer contents are disjoint? It is a finite intervention design with separate apparatus qualification, not a layer search or a claim of scaffold mediation, recovered generation or general instruction hierarchy.

The document fixes prospective methods only and asserts no experimental outcome. Runtime implementation and synthetic qualification are preparation; they do not themselves authorize real-model or tokenizer calls. No restricted scaffold material is used. Captures, token arrays, prompts, measurements and item-level evidence remain in private storage outside the public repository.

## Fixed fresh worlds and task contract

Candidate identities denote fixed pair membership, never selectedness. Recipients use pairs `(s112,s113)` and `(s114,s115)`; corresponding donors use `(s116,s117)` and `(s118,s119)`. Let R1/R2 and D1/D2 denote the first/second members of the corresponding recipient and donor pair.

| World index | Recipient A, B | Donor A, B | System-selector order |
|---:|---|---|---|
| 0 | R1, R2 from pair 0 | D2, D1 from pair 0 | A, B |
| 1 | R2, R1 from pair 0 | D1, D2 from pair 0 | B, A |
| 2 | R1, R2 from pair 1 | D2, D1 from pair 1 | B, A |
| 3 | R2, R1 from pair 1 | D1, D2 from pair 1 | A, B |

There are eight donor contexts and eight recipient contexts. These are transformations of two fixed recipient pairs and two fixed donor pairs, not eight independent inferential samples. Mapping JSON is compact, A then B, with no heading or trailing newline. Use the exact A177E system contract: substitute the uppercase system selector into the following templates and join them with one ASCII space.

```text
Select the entry labeled {selector} from the user's mapping.
```

```text
Return the mapped value of the entry labeled {selector} as exactly one JSON object with the single key "answer" and a string value.
```

```text
Do not transform the value or include other fields, explanations, or Markdown fences.
```

Donors have no user selector: their user message is the mapping alone. Every recipient has the A181 clean contradictory user construction: its mapping, two LF characters, then the following sentence naming the opposite A/B label from its system selector.

```text
Select the entry labeled {user_selector} from the mapping.
```

There are no scaffolds, prose, formatting reminders or changed role boundaries. The recipient's system selector always defines its authoritative correct answer, including in the opposite-donor arm. An intervention never changes that oracle.

## Intervention and source references

Use the pinned original CPU FP32 model/reference, SDPA, four CPU threads and cache-free batch-one forwards. The sole site is the post-block residual at zero-based block 15 of the pinned 32-block model. Within each source or recipient's own full input, the absolute site is `len(prompt_token_ids)+q-1`, with q fixed to 3. This is the last supplied pure-JSON-prefix token, before any candidate value. It is not an offset from the end of P+T. This position supplies the outgoing logit for the first scored suffix token; earlier prefix predictions precede the intervention. Replace only that one residual vector; do not modify weights, other positions, layers or masks. No site search is allowed.

| Recipient arm | Intervention source | Meaning |
|---|---|---|
| `no_patch` | None | Original reference forward; collect baseline captures. |
| `self` | This recipient's fixed no-patch R1/repeat-0 capture | Identity control. |
| `match` | Corresponding donor world's context with the same system selector as the recipient | Cross-world selector-aligned comparator. |
| `opposite` | Corresponding donor world's context with the other system selector | Cross-world selector-opposed comparator. |

The matching donor is not a no-op: donor mapping and values differ from the recipient. Both donor contexts within a world have identical mappings and other text; only their system selectors differ. The two donor arms use the same paired donor world. Source references are fixed by identity before outcomes: donor D1/repeat 0; recipient self R1/repeat 0. Never choose another repeat, candidate or source after a failure. Do not substitute an old or averaged vector.

Capture vectors during the scheduled full P+T baseline forwards; do not add prefix-only or extraction forwards. Store a detached, cloned FP32 vector with exact source, layer, position and forward binding. Compare all four captures from each donor baseline and all eight from each recipient no-patch baseline. Equality means elementwise exact equality with identical shape and float32 dtype, no numerical tolerance; nonfinite values are forbidden. Native width must agree with the frozen model configuration. Serialization must preserve the FP32 values and be hash-bound.

A source is eligible only after every required baseline result is completed with a valid finite measurement and a finite, correctly bound capture, and all its captures are exactly equal. A successful hook followed by scorer/readout failure is unusable. A known failed/malformed baseline result or unequal/invalid capture makes source eligibility false; absent unresolved observations make it null unless a known failure already makes it false. It is true only when every requirement passes. Only true eligibility permits a dependent forward.

Repeat/prefix score tolerance checks and functional competence are separate from capture eligibility. Their failure does not suppress patch calls when captures are eligible. Recipient self-source failure blocks only the self arm. Matching/opposite arms with eligible donors still execute even when no-patch or self diagnostics are missing or fail. There is no unpatched fallback for a blocked arm.

## Exact schedule and counts

Phase 1 completes the eight donors in world and system-selector order above. Each donor has the two own-value paths D1/D2, each repeated twice. Even worlds use D1,D2,D2,D1; odd worlds use D2,D1,D1,D2. This reverses the first unique candidate order and then mirrors it; it does not reverse the complete palindromic sequence. Assign repeat 0 at a path's first occurrence and repeat 1 at its second. This phase has 32 slots.

Phase 2 visits the eight recipients in the same world/selector order. For each recipient, execute no_patch first. Then execute self, match and opposite in the following frozen order. Rotate the base patched-arm list left by `world_index % 3` for the first selector; use its reversal for the second selector.

| World | First selector's patched arms | Second selector's patched arms |
|---:|---|---|
| 0 | self, match, opposite | opposite, match, self |
| 1 | match, opposite, self | self, opposite, match |
| 2 | opposite, self, match | match, self, opposite |
| 3 | self, match, opposite | opposite, match, self |

Match precedes opposite in four recipients and follows it in four. Every arm scores R1,R2,D1,D2 with two repeats. Each candidate continuation is compact JSON with the sole key "answer" and the candidate string value, followed by the intended native EOS; donors use the same construction for D1/D2. Even worlds use R1,R2,D1,D2,D2,D1,R2,R1; odd worlds use D2,D1,R2,R1,R1,R2,D1,D2. The first unique candidate order reverses and is then mirrored. Repeat indices are 0 for each first occurrence, 1 for its second. Fixed D1/repeat-0 and R1/repeat-0 source references do not change with their execution positions. Each recipient has 32 slots and phase 2 has 256.

| Quantity | Planned count |
|---|---:|
| Distinct native prompts | 16: eight donor, eight recipient |
| Native prompt/candidate constructions | 48: donor 8x2, recipient 8x4 |
| Scored donor/recipient-arm contexts | 40: eight donor, 32 recipient-arm |
| Arm-specific candidate paths | 144: 16 donor, 128 recipient-arm |
| Evaluation slots / maximum target forwards | 288: 32 donor plus 256 recipient |
| Direct primary measurements | 64: eight recipients x two donor arms x two own paths x two repeats |
| Four-way donor-arm qualification measurements | 128: eight recipients x two donor arms x four paths x two repeats |

Global sequence indices are 0 through 287. Every planned slot is accounted for. A blocked source produces an explicit `dependency_unavailable` receipt for every dependent arm slot, with no target forward. Functional outcomes alone do not skip any slot. Infrastructure, signal or deadline stops retain untouched planned slots; no retry, resume, replacement or adaptive extension is allowed.

Coverage must distinguish completed measurements, infrastructure failures, dependency-unavailable slots, interrupted slots and unattempted slots without double counting. Missing measurements equal planned minus completed. A dependency receipt counts as a processed slot but never as a target invocation. Record actual invocation separately from durable slot processing; do not claim 288 forwards occurred merely because all slots have receipts. Counts and indices are exact nonnegative integers, not booleans. Flags are true/false/null; unavailable margins are null, never strings, NaN, infinity or zero placeholders.

## Native preparation and inherited endpoint

Before any target execution, independently freeze all 16 native prompts and 48 candidate constructions under the pinned chat template. Reject payload special tokens or role/template drift. Require joint prompt/continuation tokenization, unchanged prompt-prefix IDs, exact continuation decode, completed native assistant EOS round trips, valid vocabulary and context limits. Require equal complete target length including EOS across both donor candidates and across all four candidates within each recipient. No padding, truncation, wording change or value substitution repairs failure.

The first q=3 target IDs must be shared across all paths and decode to a nonempty prefix of pure JSON opening syntax. q is not the longest common prefix. The source/recipient intervention position must precede all candidate-specific answer content. Equal complete target geometry and exact cross-candidate baseline capture equality supplement model-free causal tests; neither replaces them.

Use exact A169 teacher-forced P+T arithmetic including native EOS, `logits_to_keep=len(T)+1`, all-one attention, no cache, FP32 used logits and float64 full-vocabulary log-sum-exp. Discard the final input-EOS next-token row. Save chosen logits, normalizers and token log-probabilities privately with accurate sums. The score S is the suffix sum after q including remaining syntax, value-related tokens, closing syntax and EOS. There is no EOS-excluded, value-only, normalized-length or generation endpoint.

## Separate validity cohorts

For any specified candidate set in one arm, each candidate's repeat guard requires its two finite measurements and absolute score drift <=0.0001 nats. Its prefix guard requires all measurements of that set and prefix-sum range <=0.0001. Unknown inputs do not produce observed-subset margins. Numerical validity is false if any known required guard fails, true only if the cohort is complete and every guard passes, otherwise null. A prefix guard is null until its complete cohort is available; a repeat guard can already fail from its own two observations.

| Readout/control | Required observations | Role |
|---|---|---|
| Donor own-value validity | Four: D1/D2 x two repeats | Donor competence/readout. |
| Recipient own-value validity, per arm | Four: R1/R2 x two repeats | Own-value mean margin; local primary input. |
| Primary cross-arm prefix guard, per recipient | Eight own-value measurements across match/opposite | Additional validity requirement for that recipient's primary difference. |
| Recipient four-path validity, per arm | Eight: R1/R2/D1/D2 x two repeats | Four-way donor-copy qualification and diagnostics. |
| Full cross-arm prefix invariance, per recipient | All 32 measurements across four arms/four paths | Apparatus interpretation control only. |
| Self/no-patch drift, per recipient | Eight corresponding suffix-score comparisons: four paths x two repeats | Each absolute difference <=0.0001; apparatus control only. |

Full cross-arm prefix invariance requires range <=0.0001. It is justified by patching after the positions that predict all q prefix tokens. Self/no-patch comparisons preserve candidate identity and repeat index, not means selected after inspection. The full 32-measurement cross-arm prefix guard remains null until all 32 measurements are available, following the complete-cohort prefix rule. Each individual self/no-patch comparison requires its own two finite scores; an available violating comparison makes the conjunction of eight comparisons false despite other unknown comparisons. That conjunction is true only when all eight comparisons pass and otherwise null.

Apparatus controls include native/model-free qualification, source-capture requirements, the four-path numerical checks, full cross-arm prefix invariance and self/no-patch drift. Keep their individual flags and tri-state conjunction separate from the primary's local validity. Failure or missingness confined to donor-content paths, self, no_patch or these broader diagnostics cannot silently erase a finite primary supported by its own cohort. Such failures can prevent interpreting that primary as qualified transport.

## Primary estimand and continuous missingness

Let selected/other refer solely to the recipient system selector, and define m(i,a) as mean S(selected) minus mean S(other), using two repeats of each own-value path in arm a. The only primary is:

`T = (1/8) * sum_i [m(i,match) - m(i,opposite)]`.

There are eight fixed matched recipient pairs, 16 donor-arm own-value margins and 64 individual score measurements. Each recipient difference requires both local own-value validities and its eight-measurement cross-arm prefix guard to pass. The primary is available only when all eight differences are valid. Each contributing individual suffix score has coefficient +1/16 or -1/16 according to arm and selectedness.

If any required primary measurement or guard is missing/invalid, the primary point/lower/upper are null and unbounded is true. If complete, lower and upper equal the finite point only because missingness is absent. Report planned/resolved pairs, arm margins and direct measurement coverage. Preserve local known quantities when an unrelated diagnostic fails. No complete-case subset, zero imputation, binary bound, confidence interval or significance claim is permitted. A positive T is an observed numeric contrast. Interpreting it as donor-selector sensitivity of a valid intervention requires passing apparatus controls; it does not alone establish abstract selector transport.

## Functional interpretation gates

Use a standard tri-state conjunction throughout: any known false gives false; all required true gives true; otherwise null. For a function requiring a numerically valid score cohort, its functional flag is null unless numerical validity is true; retain numerical validity separately in the overall gate so known numerical failure dominates unknown functional flags. Strict threshold comparisons have no relative tolerance or isclose adjustment. Exactly 0.001 fails a strict greater-than requirement. These are engineering criteria, not error bounds.

| Gate | Fixed cohort and exact requirement |
|---|---|
| Donor competence | All eight donors, own-value numerical validity; `min(selected repeats)-max(other repeats) > 0.001`. |
| Fresh recipient conflict | All eight no-patch recipients, own-value numerical validity; `min(other repeats)-max(authoritative repeats) > 0.001`. Equivalently the best authoritative-selected-minus-other separation is below -0.001. |
| Four-way recipient-content qualification | All 16 match/opposite arms, each with four-path validity; expected recipient value's minimum repeat score minus the maximum over all six repeats of the other three candidates is >0.001. |

The expected recipient value in the last row is the value in the recipient mapping associated with the donor context's system selector. It is not the donor-selected value. In match this is the authoritative recipient value; in opposite it is the recipient's other value. Thus the four-way requirement explicitly compares against both donor values as well as the other recipient value.

A strongest qualified-transport interpretation requires the primary to be valid and positive, all apparatus controls to pass, and all three functional gates to pass. Keep the primary point even if this conjunction is false/null. Keep gate counts over their full eight/eight/sixteen cohorts. No favorable subset is relabeled as successful transport.

With passing apparatus controls, a positive primary with failed four-way qualification is partial conditional sensitivity. If apparatus controls fail or remain unresolved, retain the numeric contrast without asserting a qualified intervention effect. Preference dominated by donor content is compatible with answer transfer. A resolved zero primary with good controls is finite equality at this site, not equivalence. A resolved negative primary is opposite-direction sensitivity, not missingness. A null primary is unresolved and must not be described as zero. A failed apparatus or functional interpretation gate limits claims but does not authorize input, capture or site repair. Label tracking across both selectors is not a reciprocal rescue-and-impairment experiment across independently qualified recipient phenotypes.

## Qualification, runtime ceiling and evidence binding

Before target execution is even considered, tests must establish exact hook site/output typing, unchanged unhooked readout, identity self replacement, one-position isolation, correct causal prediction alignment, future-token invariance of captures, earlier-prefix invariance, cleanup after exceptions and failure receipts. Synthetic finite-score fixtures must independently verify all cohorts, thresholds, known-failure tri-states, dependency propagation and 288-slot accounting. No pretrained/target warmup is part of apparatus qualification.

Any later target runtime must retain original CPU FP32/SDPA/four-thread settings, the 48 GiB available-memory gate and exclusive reservation, a 7200-second internal/external deadline with 60-second termination grace, native preflight, one-shot claims and durable signal/attempt/result evidence. Missing reference captures have no fallback. Process every otherwise executable slot despite functional failures. The ceiling is 288 target forwards with zero additional extraction, calibration or retry passes. No API/rental spending, generation, original held-out access, private material use or consumed-evidence edits are allowed.

The scientific inputs, intervention site, schedule, source-selection rule, endpoint, primary, control cohorts and thresholds are fixed by this protocol. Remaining preparation decisions are operational bindings: final primitive/source/test hashes, independently reviewed prospective runtime and verifier, exact receipt/manifest schema implementing the distinctions above, dedicated-runtime dependency pins, and native offline construction results. The pinned reference/config/readout identities must be inherited explicitly from A181's frozen provenance. Their implementation must not silently change this design. A failed native construction ends preparation without stimulus repair.

Implementation and model-free qualification do not authorize tokenizer preparation or target execution. A separate execution decision requires the reviewed runtime, independent verification, native audits and immutable evidence bindings. This protocol alone grants no such authorization. Even a fully qualified positive result would be a constrained causal transport effect at one residual site under an EOS-inclusive supplied-prefix assay, not proof of an isolated selector representation, instruction-hierarchy mechanism, scaffold mediation, hidden correct generation or a useful corrective system.

The task contract and conditional readout follow the public [A181 methods protocol](../plans/user_selector_conflict_a181.md). Source states may encode donor content, context and position together with selector information. A pre-answer intervention site does not identify a pure selector feature. All conclusions remain limited to this fixed site, finite transformed-world cohort and EOS-inclusive conditional path.
