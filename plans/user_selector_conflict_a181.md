# A181: prospective package-by-user-selector alignment study

This is a new exploratory input-intervention question motivated by inspected
A180 results. A180 did not manipulate an explicit competing lookup selector in
the user instruction. A181 asks whether changing that selector from agreement
with the system selector to disagreement changes the conditional canonical-
answer margin more under the exact full package than under sham. It is not
confirmation, an A180 extension, a generation repair or activation patching.
The strict-generation branch and incomplete A169 study remain closed.

The system selector and mapping always define the correct answer. The user
selector is a treatment, never a second scoring oracle. The primary measures
package-specific alignment sensitivity on a fixed answer path. It does not
identify an instruction-hierarchy mechanism or isolate semantics, role
privilege, redundancy, salience, recency or termination.

## Fixed fresh inputs and private materials

Use exactly two fixed public pairs `(s104,s105)` and `(s106,s107)`. The first
pair appears in normal and swapped A/B assignments, followed by the second
pair in normal and swapped assignments, giving four worlds. Their system-
selector execution orders are AB, BA, BA, AB respectively. Each user mapping
is compact JSON with `separators=(',', ':')`, A followed by B, no heading and
no trailing newline. These are assignment and selector transformations of two
fixed pairs, not independent inferential samples. No vocabulary substitution,
selection or outcome-guided edit is permitted.

The system message consists of exactly these three single-line ASCII templates,
substituted with the authoritative uppercase selector and joined by one ASCII
space, without trailing whitespace:

```text
Select the entry labeled {selector} from the user's mapping.
```

```text
Return the mapped value of the entry labeled {selector} as exactly one JSON object with the single key "answer" and a string value.
```

```text
Do not transform the value or include other fields, explanations, or Markdown fences.
```

Both system messages equal A177/A179/A180's `of_explicit_label` instruction
byte for byte. Keep the genuine system and user roles under the pinned native
chat template. Do not add a warning, priority reminder, formatting instruction,
heading or study prefix.

There are exactly three user-selector states. For `none`, the user base is the
mapping alone and there is no user selector. For `concordant`, append exactly
two LF characters followed by this single-line ASCII template, with the user
selector equal to the system selector. For `contradictory`, use the identical
joiner and sentence with the other uppercase A/B label:

```text
Select the entry labeled {user_selector} from the mapping.
```

For a fixed base context, concordant and contradictory user bases differ only
at the intended label substitution in this directive. The system message,
mapping and scoring oracle remain identical. Both conflict directions and both
assignments are retained.

Cross each user state with exactly three packages: `clean`, `full`, `sham`.
Clean uses the user base unchanged. Full and sham append exactly two LF
characters and the corresponding original private material. Material is last,
after the mapping and optional user directive. This after-only placement and
ordering are part of the intervention and are exploratory choices motivated
by A180. No before placement, inert condition, padding or additional task is
introduced.

The full/sham materials are restricted frozen bytes. Do not read or print them
during public development or put them in Git. Root supplies the exact two-
string dictionary to plan compilation. Validate exact keys, string types,
UTF-8 byte lengths and SHA-256 digests:

| Material | Bytes | SHA-256 |
|---|---:|---|
| full | 901 | 150cfb4e3b6fffab221543a8434a3b093859f28f44ee11510c169cafeb6822f6 |
| sham | 1019 | 3f3819e8b468a35cc62e60e19ba2a4fbd6d30432479d2e6f645ce7673e355cc9 |

Bind the original source-artifact SHA-256
`ee899a47ceca1b53e90787c5279174412e8d01e5872db9bba44a486921eba345`.
Preserve its exact 25055 bytes privately at
`frozen/materials/materials-source.private.json`. Independent preparation
reconstructs only the original four-block full and sham strings with the
original joiner and checks them against the plan and receipt pins. The private
plan retains these strings and their canonical object hash. Runtime reconstructs
all contexts from the pinned strings. Material reuse does not reuse old complete
task contexts, held-outs or outputs. Record full/sham native geometry without
padding, matching their lengths or assuming equality.

## Fixed 288-forward schedule

The base condition list is `clean_none`, `clean_concordant`,
`clean_contradictory`, `full_none`, `full_concordant`, `full_contradictory`,
`sham_none`, `sham_concordant`, `sham_contradictory`. Base-context indices 0
through 7 follow world order and the selector orders above. For zero-based
world index w, rotate this nine-condition list left by 2w positions, giving
offsets 0,2,4,6. The first selector uses that order and the second uses its
reversal. Every one of the 36 condition-pair orders occurs four times in each
direction; absolute positions are only approximately balanced. Finish all four
candidate evaluations for a context before moving to the next condition.

Candidate identities remain tied to the first and second members of their fixed
pair, irrespective of selectedness or assignment. Each candidate is the compact
JSON object with exactly the key `answer` and that value, followed by intended
native EOS. Even worlds use candidate sequence 1,2,2,1 and odd worlds use
2,1,1,2; repeat indices are 0,0,1,1 in execution order. Every evaluation is a
fresh cache-free, batch-one teacher-forced forward. Global sequence indices
run from 0 through 287.

There are eight base contexts, 72 scored contexts, 144 candidate paths and
exactly 288 planned forwards. Execute the entire schedule regardless of interim
control results, numerical guards, signs or margins. The clean interpretation
gate never stops execution. No generation, target warmup, new judge, historical
input reuse, saved-score reuse, original held-outs, batching, retry, resume,
replacement or adaptive extension is allowed. Hard infrastructure, signal or
deadline stops retain the remaining planned slots as missing.

## Native construction and inherited readout

Prepare native constructions offline and freeze them before target calls. Use
the pinned A179 native/record helpers and exact A169 `teacher_force` arithmetic.
Require all 72 complete prompts to be distinct. Audit genuine role boundaries
and forbid payload special tokens. Jointly
tokenize prompt plus each candidate; require unchanged prompt-prefix IDs,
exact continuation decode and a completed native assistant round-trip with
precisely the intended EOS. Reject an earlier EOS, template/EOS drift, out-of-
vocabulary tokens, excessive prompt length or excessive actual prompt-plus-
complete-candidate length.

Fix q=3. The three prefix IDs must match across both candidates and all 72
contexts, decode to a nonempty prefix of pure JSON opening syntax
`{"answer":"`, and round-trip jointly with the native prompt. q is not the
longest common prefix: any shared value-stem tokens beyond q remain in the
scored suffix. Both complete candidate paths including EOS must have equal
lengths within each context and must extend beyond q.

For every base context and package, independently audit that concordant and
contradictory messages differ only at their intended directive label, with
identical system text, mapping, joiners and material bytes. Require equal native
prompt length for each of these 24 concordant/contradictory pairs. Record
none-versus-explicit length differences and full/sham counts without pretending
they are controlled. Preserve failed preparation and stop if any native audit
fails; do not alter wording, pair values, q or material, or pad or truncate an
input to repair it.

Feed complete prompt P plus target T including EOS to each forward, with an
all-one attention mask, `use_cache=False` and `logits_to_keep=len(T)+1`.
Score every target from its preceding input position; discard the final row
predicting beyond the input EOS. Convert used logits to FP32 and compute the
full-vocabulary log-sum-exp and chosen-logit subtraction in float64. Require
finite used logits and saved arithmetic; use accurate summation.

The candidate endpoint is the suffix sum after q, including remaining syntax,
value-related material, closing syntax and EOS. Retain exact prefix, complete-
path and EOS diagnostics privately. Do not replace the endpoint with a value-
only, EOS-excluded, token-normalized, whole-path, sign, worst-repeat or generation
measure. No new response detector, raw-response audit or alternative rescoring
is introduced.

For each context require all four measurements, prefix-sum spread <=1e-4 nats
across them, and absolute suffix-score repeat drift <=1e-4 nats for each
candidate, with no relative tolerance. The prefix guard needs all four
measurements; each repeat guard needs its own two.
Numerical validity is false if any known required guard fails, true only if
complete and all pass, and null otherwise. A known repeat failure may therefore
disqualify an incomplete context. Neither tolerance is relaxed or adapted from
results, and neither is a population confidence interval or independently
validated numerical error bound.
Small signed differences remain observed finite values; their signs are not
precision guarantees or inferential conclusions.

For a numerically valid context, define mean margin m as the mean authoritative
selected-candidate suffix score minus the mean other-candidate suffix score.
Retain worst-repeat separation `min(selected repeats)-max(other repeats)`
diagnostically. Missing or numerically invalid margins are null, with null
lower/upper bounds and unbounded missingness. Never use zero imputation or
average the observed subset of an incomplete context.

## Clean-none interpretation gate

Only the eight `clean_none` contexts receive functional qualification: each
requires worst-repeat separation strictly greater than 0.001 nats. Preserve
ties, reversals, positive low separation and exact equality at this threshold.
Their conjunction is false if any known numerical or functional check fails,
true only if all eight fully pass, and null otherwise. This checks lookup and
readout competence without supportive user agreement. It is an interpretation
gate, not an execution, score-availability or inclusion gate; its failure does
not erase otherwise valid margins or permit a clean-pass subset.

All other 64 contexts, including clean-concordant and clean-contradictory,
have `functional_applicable=false`, `functional_passed=null` and
`functional_status=not_applicable`. Their positive-low-separation and threshold-
boundary qualification flags are null. Retain descriptive mean/worst ties,
reversals and signs when numerically valid. Negative contradictory or contextual
margins are outcomes, never functional inclusion failures.

## Fixed estimands and continuous missingness

Let m(i,p,u) denote the valid authoritative selected-minus-other mean suffix
margin for base context i, package p and user state u. Each of the eight base
contexts has weight 1/8. There is exactly one primary:

`I = sum_i[(m(i,full,contradictory)-m(i,full,concordant))-(m(i,sham,contradictory)-m(i,sham,concordant))]/8`.

Its cohort is exactly 32 distinct context margins in eight matched quartets.
Per quartet the coefficients are +1/8 full-contradictory, -1/8 full-concordant,
-1/8 sham-contradictory and +1/8 sham-concordant. A negative I means the signed
concordant-to-contradictory change is more negative under full than sham on this
endpoint and cohort; positive means the reverse. Zero is finite equality, not
equivalence or evidence of no effect in general. A negative I does not require
either within-package change to be negative or any contradictory margin to
reverse sign.

Retain two direct primary decomposition diagnostics: mean contradictory-minus-
concordant change within full, and the same change within sham. Each uses its
own sixteen context margins in eight matched pairs. Calculate these directly
from their concordant and contradictory observations. Never derive them by
subtracting two none-referenced secondary contrasts: cancel algebraically shared
none coefficients before arithmetic and missingness checks. Missing none
observations must not erase a resolved direct component or the primary. These
components and the four primary constituent condition means describe the
primary; they are not additional hypotheses.

There are exactly six secondary contrasts: for each package p in clean, full,
sham, report `sum_i[m(i,p,contradictory)-m(i,p,none)]/8` and
`sum_i[m(i,p,concordant)-m(i,p,none)]/8`. Each uses sixteen distinct context
margins in eight matched pairs, with coefficients +1/8 and -1/8. Each package's
same eight none observations are shared by its two secondary contrasts; do not
count them as independent repeated samples. Retain all nine condition means,
each requiring its own fixed eight valid context margins.

Combine coefficients of any shared observation before arithmetic and before
testing required missingness. Every contrast and direct component reports its
point, lower, upper, unbounded status, planned/resolved contexts and planned/
resolved matched groups. If all required nonzero-weight margins are known,
lower and upper equal the point solely because missingness is absent. If any
required margin is missing or numerically invalid, point/lower/upper are null
and unbounded is true. Unrelated missing contexts do not erase resolved
contrasts, components or means. No complete-case substitution, observed-subset
mean, binary [-1,1] bound, pooled endpoint, selected-world estimate, confidence
interval, significance test or population effect is permitted.

The primary path is
`contrasts.mean_margin_nats.package_by_user_selector_alignment`, with
`planned_contexts=32`, `resolved_contexts`, `planned_quartets=8`,
`resolved_quartets`, `point`, `lower`, `upper`, `unbounded`. Direct decomposition
paths are `primary_components.mean_margin_nats.full_contradictory_minus_concordant`
and `primary_components.mean_margin_nats.sham_contradictory_minus_concordant`,
each with `planned_contexts=16`, `resolved_contexts`, `planned_pairs=8`,
`resolved_pairs`, `point`, `lower`, `upper`, `unbounded`. The six secondary paths
are `secondary_contrasts.mean_margin_nats.{package}_{concordant|contradictory}_minus_none`,
with the same sixteen-context/eight-pair accounting. `by_condition` retains nine
fixed-cohort means and guard/sign summaries. Overall coverage counts the 72
contexts once; `clean_qualification` counts only the eight `clean_none` contexts.
All status and sign counts preserve explicit unknowns.

Use record schema `a181-context-margin-v1`. Context metadata retains `package`,
`user_selector_state`, `user_selector`, `base_context_index`, `condition`,
`scaffold_kind` and `placement`. Clean has no scaffold or placement; full/sham
use the corresponding scaffold kind and after placement. Record native geometry,
candidate identities, repetitions, coverage and inherited readout diagnostics
privately without changing the system-defined oracle.

## Runtime, evidence and stop rules

Use original local weights, corrected CPU FP32 reference, SDPA and four CPU
threads with the unchanged seed. Require 48 GiB available immediately before
loading and the established exclusive reservation/lock. The internal deadline
is 7200 seconds including preflight/loading, with external enforcement and a
60-second kill grace. SIGTERM/SIGINT/SIGHUP/SIGALRM are terminal; preserve
actual signal name/number, PID and durable attempts/results. KeyboardInterrupt
has null signal fields. Recoverable per-forward failures retain missing slots
without skipping the remaining authorized schedule.

Use an absolute private run root outside the source checkout, durable sequential
attempts/results and an immutable one-shot claim before native startup/loading.
Native startup failure consumes that claim. Changing directory cannot resume a
consumed process/run. No retry, rerun, replacement or score reuse is allowed.
A resource queue may wait for the required headroom without disturbing others.

Bind original config SHA
`1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745`,
reference SHA `f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`,
A179 helper SHA `af0b811da83166db5a410956c1b7160e7fde297458f07d4b531a30586f8790e8`,
and A169 readout SHA `1b63ddf15eae5866b444cdb94293b80dcf7bd6e580978a23873c0167cb40e69e`.
Preserve the dedicated interpreter path
`/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python`
without resolving it. No target warmup, generation, GPU, API/rental spending,
old held-out access or changes to consumed evidence.

API: `compile_plan(config_sha256, protocol_sha256, tests_sha256, materials)`,
`prepare_inputs`, `run_plan` and `export_run`, retaining A180's keyword-only
bound-path preparation/run/export interface. Plan compilation accepts only the
full/sham material dictionary. Root reviews and pushes public source, tests and
protocol before native preparation, independently qualifies preparation and
freezes source, input and native manifests before any target forward. No
execution is authorized by an unqualified construction or an unfrozen plan.

Private material, prompts, numerical/token arrays and per-item identifiers stay
outside Git and console/tool output. Independent terminal replay reconstructs
native paths, saved chosen-logit-minus-normalizer arithmetic, sums, guards,
clean-none interpretation gate, exact cohorts, components, contrasts, condition
means and coverage without new forwards. Reject drift, orphan or out-of-order
receipts, changed arithmetic, summaries or completion claims. Saved scalar
normalizers do not independently regenerate full-vocabulary log-sum-exp;
full-vocabulary vectors are not retained. Analytical and tiny untrained Torch
controls qualify the readout implementation separately.

## Interpretation and public synthetic qualification

A resolved negative primary is an exact package-by-alignment interaction.
Interpret it alongside both signed direct components, six none-reference
secondaries, nine condition means and guard/sign diagnostics. Greater negative
contradictory-minus-none change with little change in agreement benefit is
consistent with an added contradiction penalty on this endpoint; greater
concordant-minus-none benefit with similar contradiction behavior is consistent
with losing stronger agreement support. Both may contribute, and mixed results
remain mixed. These are readings of the frozen components, not selected tests
or a replacement primary. The none-reference comparisons also change sentence
presence and length, so they do not isolate conflict semantics or repetition.

Concordant/contradictory native length matching controls that geometric
difference; it does not isolate privilege from token identity, salience,
redundancy, recency or termination. Full/sham compare exact packages differing
in content and byte length; native prompt counts are recorded and are not assumed
to differ. The material-last order and after-only placement limit
the claim to that ordering. The readout conditions on supplied syntax and
includes EOS; it is not format-independent, pure value selection, generation
accuracy, hidden correctness, loss of system-role binding or patch eligibility.

Nonnegative or zero primary values are equally valid outcomes. Clean-none
failure limits interpretation; numerical failure and missingness preserve the
specified nulls. Do not seek a preferred sign through another sentence, pair,
placement, tolerance or input set. Complete, independently verify, interpret
and archive this finite study before making a distinct next decision. No
outcome automatically authorizes mechanism claims, intervention eligibility,
activation patching, repaired generation or expanded scope.

Qualification uses public synthetic fixtures and monkeypatched material pins
only. Independently enumerate all 288 slots, fixed identities, 36 pairwise
condition-order balances, intended directive substitutions and fixed cohorts.
Test exact system/user bytes, full/sham-only pins, material tampering, native
roles/special-token guards, q=3 and EOS geometry, equal candidate lengths and
all 24 concordant/contradictory prompt-length audits. Exercise numerical
boundaries, repeat missingness, retained negative margins and clean-none
interpretation failure with continued execution. Qualify causal alignment
analytically and with a tiny untrained Torch model without pretrained weights.
Test the four-term signed primary, both direct components and all six secondary
cohorts independently, including missing none margins that leave the primary
and direct components resolved. Check nine means, shared-control accounting,
three-state validity, one-shot/signal/deadline/lock semantics, durable evidence,
tamper rejection, safe exports and privacy. No restricted material, historical
raw context or target tokenizer/model is read in these public tests. This
prospective protocol asserts no test pass or experimental outcome.
