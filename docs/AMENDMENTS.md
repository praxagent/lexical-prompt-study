# Plan amendments

The locked charter may change only through an entry added here before the
affected outcome is inspected.

| ID | Date | Status | Outcomes inspected? | Change | Rationale | Approval |
|---|---|---|---|---|---|---|
| A000 | 2026-07-25 | Baseline | OpenRouter exploratory outcomes only; no raw-checkpoint target outcomes | Lock Study Charter v1.0 | Establish an auditable starting point before implementation or paid compute | Human requested first plan commit |
| A001 | 2026-07-25 | Pre-outcome design change | No raw-checkpoint target outcomes | Validate HarmBench on JBB's 300 published human-majority comparisons using balanced accuracy and recall as gates; report precision without gating it; use the runtime affirmative-token probability for the paired primary score | Published HarmBench binary labels pass balanced accuracy (0.802) and recall (0.873) gates but have precision 0.653, so a predeclared 0.80 precision gate would reject the evaluator while discarding its useful paired continuous signal | Human authorized execution under the locked change-control process |
| A002 | 2026-07-25 | Pre-outcome implementation clarification and infrastructure correction | No raw-checkpoint target outcomes | Resolve the HarmBench continuous score as the exact one-token `Yes` versus `No` softmax; retain both turns for E02; persist figure byte-verification status; verify Git blob IDs for non-LFS artifacts; bind restricted generation files to trial receipts | Remove implementation ambiguity and make the frozen evaluator, visual contract, artifact inventory, and crash-resume evidence fail closed without changing any estimand, split, threshold, arm, or prompt | Human authorized execution under the locked change-control and provenance policies |

An amendment entry must distinguish:

- clarification with no change to the estimand;
- pre-outcome design change;
- infrastructure correction;
- post-outcome exploratory analysis;
- new prospective protocol version.

The Git diff, not this table alone, is the authoritative record of what changed.
