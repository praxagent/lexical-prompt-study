# A168: fixed displayed-label renaming in all A167 prose trials

Prospective, separate 32-call descriptive diagnostic. A167 is complete: all 48
cells resolved, baseline 16/16 correct, inert 0/16 with 16 caps, prose 10/16, all
prose JSON plus EOS. The six prose errors motivated this diagnostic: five copied
the authoritative displayed label and one copied the other given value. These
historical errors never determine which A168 pairs are run or its transfer subset.
A164's failed gate and unused held-out worlds remain unchanged.

Run all 16 original A167 prose trials: four fixed development worlds in their
original order, both original selectors, both placements. Each pair contains the
unchanged original prompt and one renamed prompt. Rename displayed labels A to C
and B to D only in the system's authoritative-selector clause, its label list
(`labeled A and B` to `labeled C and D`), and the task JSON table keys. Preserve
all other wording, values, query order, prose bytes and placement. Reconstruct
the exact task renderer; never blindly replace A/B inside prose. The internal
A/B world and selector metadata and answer oracle remain unchanged; each trial
explicitly records its displayed-label mapping. No new or edited prose is allowed.

Even parity of zero-based world + selector + placement indices runs original
then renamed; odd parity runs renamed then original. There are eight pairs in
each order, balanced within every factor. No error-based selection or adaptation.
Both full native prompts must have exactly equal token counts for every pair,
and original messages and native tokens must equal their bound A167 sources.
Any mismatch blocks the entire schedule before model loading, with no padding,
truncation, dropped pairs or revised labels. Freeze source, tests, this protocol,
private plan, 32 native prompts and matching receipts before calls.

The compiler accepts caller-supplied bound A167 plan/header objects and this
protocol's hash; it reads no historical responses. A167 source is pinned at
`12de497f8977bf51edf09c284c8a18d882d187fca7dd050578540518fff6cb43`.
Keep the consumed A167/A166/A165/A164 source closure unchanged. Reuse their
native-token preparation and strict scorer through explicit schema adapters.
Preparation also reconstructs the complete A167 native freeze and verifies its
header bindings. Every A168 plan, identity, run, attempt, result, cell, native
freeze, completion receipt and aggregate has its own A168 schema.

Use the unchanged original A165 config and CPU FP32 reference script SHA-256
`f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`.
The inherited config still describes the original NF4 instrument; effective
settings remain original-weight, unquantized CPU FP32, CPU SDPA, four threads,
unchanged seed, greedy single-beam generation with a 64-token maximum. Require
48 GiB available RAM before model loading. Use one fresh process, one model load,
and no state resets between calls. Run all 32 scheduled calls even when model
responses are errors or generation raises a recoverable exception. Unresolved
infrastructure outcomes remain missing; no retries or replacement calls.

Use an absolute private directory outside the source checkout, exclusive flock,
immutable one-shot claim before loading, a process consumption guard, hash-bound
source/config/model/plan/native receipts, and sequential durable attempts. Launch
under an external 2,400-second deadline and 60-second kill grace; an internal
2,400-second alarm covers preflight, loading and execution. SIGTERM, SIGINT,
SIGHUP and the alarm raise BaseException to bypass per-trial Exception recovery.
Preserve partial receipts and all 32 planned outcomes. No resume, retries,
held-out calls or automatic extension beyond this fixed diagnostic.

Export only after shared-lock replay reconstructs native prompts, validates
response-token decoding, EOS/cap termination and strict boolean scores, and
rejects drift, duplicate/unknown/out-of-order attempts, orphan results or changed
stored exports. Raw prompts, responses, token arrays, trial/world identities and
per-item token hashes stay private. The aggregate contains counts, bounds and
whole-artifact provenance hashes only; no interim effects go to stdout.

Primary estimand: strict accuracy renamed minus original across all 16 pairs.
Average the four selector/placement differences within each fixed world, then
average all four fixed-world means equally. Report all four world means and the
overall difference. Unresolved binary outcomes contribute [0,1], so missingness
yields worst/best bounds, never deletion or imputation; point estimates require
identification. Retain all 16 pairs regardless of the transfer subset. Secondary
placement and execution-order contrasts each retain eight planned pairs over the
same four worlds, plus parser-format and cap counts/contrasts.

Secondary response classes are mutually exclusive, computed only after receipt
validation from the private JSON string with EOS: correct value, other given
value, selected displayed label, other displayed label, legacy selected A/B
label (renamed only), legacy other A/B label (renamed only), other allowed symbol,
other string, or format/cap. Capped or invalid-format responses enter format/cap;
infrastructure failures and unattempted/interrupted cells remain unresolved.

The descriptive transfer subset is contemporaneous ORIGINAL selected-displayed-
label errors. Let K be its observed membership and U the number of original
outcomes unresolved. Report K, U, membership bounds [K,K+U], known nonmembers,
renamed-counterpart category counts and missing counts for both known members
and unknown-membership pairs. For each category, known-member count bounds are
[c,c+m] and rate bounds [c/K,(c+m)/K], where m is unresolved known-member
counterparts. Preserve K=0 as undefined rates: report uninformative only if U=0;
otherwise report inconclusive membership. For the possible full subset, report
conservative rate bounds [c/(K+U),(c+m+U)/(K+U)] when K>0. With K=0,U>0 report
[0,1] conditional on a nonempty subset and explicitly note it may be empty.
Category bounds are marginal and need not sum to one. Unknown memberships are
never silently dropped or counted as known failures. No generalized threshold
or mechanism decision is declared.

This is a descriptive role-versus-letter renaming comparison in these fixed
worlds with this fixed prose passage. A renamed label response is compatible
with role-following label copying; retained A/B responses are compatible with
letter persistence. Neither pattern proves a causal mechanism, nor does an
accuracy change isolate semantics, letter frequency, token identity or carryover.
No CIs, population inference or human-harm claims. Later separately reasoned
studies retain the user's existing research authorization.

API: `compile_plan(a167_plan, a167_run_header, protocol_sha256)`, `prepare_inputs`,
`run_plan`, `export_run`. Preparation requires bound `plan_path`, `plan_sha256`,
`config_path`, `config_sha256`, `protocol_path`, `reference_script`; run/export
also require `output_root`. CLI: `python -m
lexical_prompt_study.instruction_selection_label_rename --mode run` (or `export`)
with `--plan`, `--plan-sha256`, `--config`, `--config-sha256`, `--protocol`,
`--reference-script`, `--output-root`. Preserve the original virtualenv
interpreter path when launching the frozen source closure.
