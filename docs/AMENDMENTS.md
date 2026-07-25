# Plan amendments

The locked charter may change only through an entry added here before the
affected outcome is inspected.

| ID | Date | Status | Outcomes inspected? | Change | Rationale | Approval |
|---|---|---|---|---|---|---|
| A000 | 2026-07-25 | Baseline | OpenRouter exploratory outcomes only; no raw-checkpoint target outcomes | Lock Study Charter v1.0 | Establish an auditable starting point before implementation or paid compute | Human requested first plan commit |
| A001 | 2026-07-25 | Pre-outcome design change | No raw-checkpoint target outcomes | Validate HarmBench on JBB's 300 published human-majority comparisons using balanced accuracy and recall as gates; report precision without gating it; use the runtime affirmative-token probability for the paired primary score | Published HarmBench binary labels pass balanced accuracy (0.802) and recall (0.873) gates but have precision 0.653, so a predeclared 0.80 precision gate would reject the evaluator while discarding its useful paired continuous signal | Human authorized execution under the locked change-control process |
| A002 | 2026-07-25 | Pre-outcome implementation clarification and infrastructure correction | No raw-checkpoint target outcomes | Resolve the HarmBench continuous score as the exact one-token `Yes` versus `No` softmax; retain both turns for E02; persist figure byte-verification status; verify Git blob IDs for non-LFS artifacts; bind restricted generation files to trial receipts; reserve H200 KV-cache headroom with the previously validated 125 GiB-per-GPU sharding ceiling | Remove implementation ambiguity and make the frozen evaluator, visual contract, artifact inventory, crash-resume evidence, and paid topology fail closed without changing any estimand, split, threshold, arm, or prompt | Human authorized execution under the locked change-control and provenance policies |
| A003 | 2026-07-25 | Post-generation environment correction | Target generations complete but raw outcomes and evaluator scores not inspected; failed scorer and tokenizer smoke wrote 0 score receipts | Add the `sentencepiece` and `protobuf` runtime dependencies required to load and convert the frozen HarmBench Llama-2 tokenizer, then retry the unchanged scorer against the same generation receipts and output root | The first scorer invocation and first corrected tokenizer smoke failed during tokenizer construction before loading the evaluator or writing a score; this repairs the locked environment without changing prompts, outputs, evaluator weights, scoring code, estimands, or thresholds | Authorized operational recovery within the approved study protocol |
| A004 | 2026-07-25 | Post-score analysis implementation correction | All evaluator receipts exist and pass linkage checks, but raw outcomes were not inspected and no gate statistic was produced | Retain the generation receipt's `turn` field in the gate analyzer's joined provenance row and cover the complete 20-behavior/160-receipt discovery topology in a regression test | The analyzer already selected turns 1 and 2 but omitted `turn` from the constructed row before filtering for the frozen turn-2 estimand; adding the missing field repairs execution without changing inputs, scores, bootstrap seed, thresholds, or estimand | Authorized fail-closed recovery within the approved study protocol |

An amendment entry must distinguish:

- clarification with no change to the estimand;
- pre-outcome design change;
- infrastructure correction;
- post-outcome exploratory analysis;
- new prospective protocol version.

The Git diff, not this table alone, is the authoritative record of what changed.
