# A180: prospective scaffold conditional-margin study

A179 completed 32 fresh clean forwards and qualified all eight clean contexts,
with zero saved prefix/repeat drift and selected-minus-other margins from
11.2595128009 to 12.0259486908 nats. That supports a separate contextual study;
it does not qualify new contexts in advance. A178's strict-generation clean
7/8 versus contextual 0/48 remains unchanged. The strict-generation branch and
incomplete A169 study remain closed. This study measures conditional canonical
answer paths, never generated responses, hidden correctness or patch eligibility.

## Fixed fresh inputs and private materials

Use exactly two fixed public pairs `(s100,s101)` and `(s102,s103)`. Worlds w00
and w01 use the first pair in normal and swapped A/B assignment; w02 and w03
use the second pair in normal and swapped assignment. The selector execution
orders are AB, BA, BA, AB respectively. Each user mapping is compact JSON with
`separators=(',', ':')`, A followed by B. These are four assignment transformations
of two fixed pairs, not four independent pairs or eight independent samples.
The three-digit value encoding is part of this fresh package and is checked by
concurrent clean controls. No vocabulary search or replacement is allowed.

Use exactly these three single-line ASCII templates, substituted with the active
uppercase selector and joined by one ASCII space, without trailing whitespace:

```text
Select the entry labeled {selector} from the user's mapping.
```

```text
Return the mapped value of the entry labeled {selector} as exactly one JSON object with the single key "answer" and a string value.
```

```text
Do not transform the value or include other fields, explanations, or Markdown fences.
```

Both complete system messages equal A179/A177's explicit-label instruction
byte for byte. No instruction, heading, reminder, study prefix or prose is added.

The seven conditions, in base order, are `clean`, `full_before`, `full_after`,
`sham_before`, `sham_after`, `inert_before`, `inert_after`. Clean is mapping only.
Before is exact material + two LF characters + mapping. After is mapping + two
LF characters + exact material. Metadata records `scaffold_kind` as
none/full/sham/inert and `placement` as none/before/after.

The full/sham/inert materials are restricted frozen bytes. Do not read or print
them during public development or put them in Git. Root supplies the exact
three-string dictionary to plan compilation. Validate its exact keys, string
types, UTF-8 byte lengths and SHA-256 digests:

| Material | Bytes | SHA-256 |
|---|---:|---|
| full | 901 | 150cfb4e3b6fffab221543a8434a3b093859f28f44ee11510c169cafeb6822f6 |
| sham | 1019 | 3f3819e8b468a35cc62e60e19ba2a4fbd6d30432479d2e6f645ce7673e355cc9 |
| inert | 1275 | 402a805911f0ad6e6708ca7447bc47c2149183433cbde15df62310cbacb021ed |

Bind source-artifact SHA-256
`ee899a47ceca1b53e90787c5279174412e8d01e5872db9bba44a486921eba345`.
Preserve its exact 25055 bytes privately at
`frozen/materials/materials-source.private.json`. Independent preparation audits
this self-contained artifact, reconstructs the three four-block materials with
the original joiner, and compares them with the plan and receipt pins. The
private plan retains material strings and their canonical object hash; runtime
reconstructs all contexts from those pinned strings. Reusing materials does not
reuse old complete task contexts, held-outs or outputs. Never match or pad
material lengths; record native geometry without claiming length control.

## Fixed 224-forward schedule

Base-context indices 0 through 7 follow world order and the displayed selector
orders. For world index w, rotate the seven-condition base list left by 2w
positions: offsets 0,2,4,6. Its first selector uses that list; its second uses the
reverse. Thus every one of the 21 condition-pair orders occurs four times in
each direction. Absolute position frequencies are only approximately balanced.
Finish all four candidate evaluations of a condition before the next condition.

`value_1` and `value_2` remain tied to the first and second members of the fixed
pair, regardless of selector or assignment. Candidate strings are compact JSON
objects with exactly `answer` and the corresponding value, plus native EOS.
Even worlds execute candidate order 1,2,2,1; odd worlds execute 2,1,1,2. Repeat
indices are 0,0,1,1 in execution order. All evaluations are fresh cache-free,
batch-one forwards with unique global sequence indices 0 through 223.

There are eight base contexts, 56 scored prompt contexts, 112 candidate paths
and 224 forwards. Execute every scheduled forward regardless of interim clean,
scaffold, numerical or functional findings. No baseline gate, warmup, batching,
retry, resume, saved-score reuse, adaptive extension or replacement is allowed.

## Native construction and inherited readout

Freeze native constructions before target calls. Use the pinned A179 preparer
and the exact A169 `teacher_force` implementation. Jointly tokenize prompt plus
candidate; require unchanged prompt-prefix IDs, exact candidate decode and
completed native assistant round-trip with precisely the intended EOS. No
candidate may contain an earlier EOS. Enforce bound template/EOS, vocabulary,
maximum prompt and actual prompt-plus-complete-candidate context limits.

Fix q=3. Its IDs must match across both candidates and all 56 contexts and
decode to a nonempty prefix of pure JSON opening syntax `{"answer":"`. Require
joint prefix round-trip. q is not the longest common prefix; shared value-stem
tokens beyond q stay in the scored suffix. Both candidate paths including EOS
must have equal lengths within each context and extend beyond q. Failed native
qualification ends preparation without padding, truncation or input repair.

Feed complete P+T including EOS to each fresh forward, all-one attention mask,
`use_cache=False`, `logits_to_keep=len(T)+1`. Score each target from its preceding
input position and discard the final next-token row produced by the input EOS.
Convert used logits to FP32 and compute full-vocabulary log-sum-exp and chosen
logit subtraction in float64. Require finite used logits and saved arithmetic.
The primary score is the sum after q, including remaining syntax, value-related
material, closing syntax and EOS. Prefix sum, complete-path sum and EOS
contribution remain separate diagnostics. Do not normalize by token count or
switch to a value-only, EOS-excluded, sign or worst-repeat primary outcome.

For each context require prefix-sum range across all four evaluations <=1e-4
nats, and each candidate's two suffix scores to differ by <=1e-4 nats. The
prefix guard needs all four; each repeat guard needs its own two. Numerical
validity is false if any known required guard fails, true only when all four
are complete and all guards pass, and null otherwise. A known repeat failure
can disqualify an incomplete context without imputing its missing measurement.
No guard is waived because A179 passed or because a contrast appears useful.

For a numerically valid context, define mean margin m as the mean selected
suffix score minus mean other suffix score. Retain worst-repeat separation
`min(selected repeats)-max(other repeats)` diagnostically. Missing or numerically
invalid margins are null, with lower and upper bounds null and uncertainty
unbounded. Never average an observed subset of the four evaluations.

## Clean interpretation gate and scaffold outcomes

Only clean contexts receive functional qualification: worst-repeat separation
must be strictly greater than 0.001 nats. Preserve exact mean/worst ties,
reversals, positive low separation and equality at the threshold. The eight-clean
conjunction is false if any known numerical/functional check fails, true only
if all eight fully pass, and null otherwise. This is an interpretation gate,
not an execution or score-availability gate. Its failure limits claims of a
qualified clean instrument and must accompany otherwise available contrasts.
It does not delete valid scores or authorize a clean-pass subset.

Scaffold contexts have `functional_applicable=false`, `functional_passed=null`,
`functional_status=not_applicable`, and null positive-low-separation/boundary
qualification flags. Descriptive mean/worst ties and reversals remain available
when numerically valid. Negative, zero and small positive scaffold margins are
outcomes, not failed inclusion criteria. Neither engineering threshold is a
confidence limit or a validated numerical error bound.

## Fixed estimands and continuous missingness

There are exactly two placement-specific primary contrasts, indexed by all
eight base contexts i:

- `full_minus_sham_before = sum_i(m(i,full_before)-m(i,sham_before))/8`.
- `full_minus_sham_after = sum_i(m(i,full_after)-m(i,sham_after))/8`.

There are exactly two secondary contrasts:

- `inert_minus_clean_before = sum_i(m(i,inert_before)-m(i,clean))/8`.
- `inert_minus_clean_after = sum_i(m(i,inert_after)-m(i,clean))/8`.

Every contrast has 16 unique scored contexts, eight matched pairs and weights
+1/8 and -1/8. The two secondary contrasts share the same eight clean margins;
these are eight observations, not sixteen. Do not add a pooled contrast,
interaction, alternate readout or data-selected estimand. Preserve separate
placements even when signs differ.

Each contrast reports point/lower/upper/unbounded, planned/resolved contexts
and planned/resolved pairs. If every required margin is known, lower and upper
equal the point (not confidence limits). If any required nonzero-weight margin
is unknown, point/lower/upper are null and unbounded is true. Missing conditions
outside that contrast's fixed cohort must not invalidate it. No complete-case
substitution, zero imputation or binary [-1,1] bounds apply to these continuous
log-probability margins. Combine any shared-observation coefficients before
arithmetic; there is no additional pooled clean estimate in this study.

The primary path is `contrasts.mean_margin_nats`; the secondary path is
`secondary_contrasts.mean_margin_nats`, with only the four keys above. Retain
seven `by_condition` summaries, each requiring all eight numerically valid
margins for its fixed-cohort mean. Report planned/resolved denominators,
coverage, numerical statuses and descriptive sign/guard counts with explicit
unknowns. Overall counts cover 56 contexts once; clean qualification covers
eight once. No population estimates, confidence intervals or equivalence tests.

## Runtime, evidence and stop rules

Use original local weights, corrected CPU FP32 reference, SDPA and four CPU
threads with the unchanged seed. Require 48 GiB available immediately before
loading and the established exclusive reservation/lock. Freeze 7200 seconds
including preflight/loading with external enforcement and 60-second kill grace.
SIGTERM/SIGINT/SIGHUP/SIGALRM are terminal; preserve actual signal name/number,
PID and durable attempts/results. KeyboardInterrupt has null signal fields.
Native startup failure consumes the one-shot claim. Changing directory cannot
resume a consumed process/run. Recoverable per-forward failures retain fixed
missing slots and do not skip remaining scheduled evaluations.

Bind original config SHA
`1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745`,
reference SHA `f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e`,
A179 helper SHA `af0b811da83166db5a410956c1b7160e7fde297458f07d4b531a30586f8790e8`,
and A169 readout SHA `1b63ddf15eae5866b444cdb94293b80dcf7bd6e580978a23873c0167cb40e69e`.
Preserve interpreter path
`/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python`
without resolving it. Root reviews, pushes code-only changes, freezes a verified
private prepared archive and launches once. No target warmup, generation, GPU,
API/rental spending, old held-out access or changes to consumed evidence.

Private material, prompts, numerical/token arrays and item identifiers stay
outside Git and console output. Independent terminal replay reconstructs native
paths, saved chosen-logit-minus-normalizer arithmetic, sums, guard flags, clean
gate, fixed cohorts, contrasts and coverage without new forwards. Saved scalar
normalizers cannot independently regenerate full-vocabulary log-sum-exp;
full-vocabulary vectors are not retained. Analytical and tiny untrained Torch
checks qualify the readout implementation separately.

A completed study ends here. Numerical failure or missingness yields the frozen
null contrasts; clean failure yields its interpretation limitation; small or
zero package differences do not establish equivalence. Full versus sham
compares exact packages differing in content and length, not isolated semantics
or token count. Inert versus clean concerns the exact added package. No outcome
authorizes numerical repair, vocabulary/wording search, repeat studies, rescued
generation, mechanism attribution or causal patching. Close and synthesize the
bounded study before considering another question.

## Public synthetic qualification

Use public fixture text and monkeypatched material receipts only. Independently
enumerate all 224 slots, candidate identities and 21 pairwise order balances;
exercise material tampering, native geometry/round-trip guards, exact numerical
boundaries, all repeat missingness patterns, clean interpretation failure with
continued execution, and retained negative scaffold margins. Test analytical
causal alignment and a tiny untrained Torch model without pretrained weights.
Qualify fixed continuous cohort arithmetic, missingness local to each estimand,
shared clean accounting, one-shot/signal/deadline/lock semantics, durable
publication, tamper rejection, safe exports and privacy. No actual restricted
material, historical raw context or target tokenizer/model is read in tests.
