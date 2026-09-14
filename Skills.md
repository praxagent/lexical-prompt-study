# Project research and compute skills

Updated 2026-09-14 after migration to Zeta.

## Use the local server first

The user authorizes use of this server's CPU, RAM, and GPU for otherwise
authorized project work. Prefer local execution when the workload fits and
resources are available. The earlier weak-computer assumption is obsolete.

Verified installed hardware:

| Resource | Capacity |
|---|---|
| CPU | Intel Core i9-14900K, 24 cores / 32 logical CPUs |
| RAM | Approximately 91 GiB reported by Linux |
| GPU | NVIDIA RTX 2000 Ada Generation, 16,380 MiB VRAM (about 16 GiB) |

Capacity is shared with other agents. Before substantial work, check CPU load,
available RAM, scratch space, GPU utilization, and GPU processes. Use `uptime`,
`free -h`, `df -h` for the intended paths, and:

```sh
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Size worker counts, BLAS/OpenMP threads, batches, and memory against live
headroom. Recheck before launch and use an existing reservation mechanism when
available. Do not interrupt other agents' jobs or reset the shared GPU. A
restricted GPU query is not evidence that the GPU is idle or broken; use an
authorized host-level check. Release only this task's processes and resources
when finished.

## RunPod remains available

Use RunPod when the local GPU is busy, lacks enough VRAM, cannot support the
required runtime or multiple GPUs, or cannot deliver suitable throughput.
CPU/RAM-heavy work may also use a rental when it cannot fit safely alongside
other local work. Record the reason and honor existing scoped cost approvals
and cumulative budgets; new spending outside them needs additional approval.

Local use requires no rental-cost approval. Keep scientific protocols, model
and precision requirements, privacy boundaries, and immutable evidence intact.
Qualify the actual runtime and numerical behavior before fresh experimental
work; hardware migration alone does not authorize rerunning completed studies.

The shared [compute playbook](/data2/agent-skill-documents/GPU_COMPUTE_SKILLS.md)
provides the full local-capacity and rental lifecycle guidance. Its provider
billing and pod-deletion rules apply to rentals. The historical migration
skills snapshot remains preserved as evidence of the earlier environment.

## Keep authorized research moving

Continue to the next useful authorized step after each milestone. A pending
human rating, review, or external dependency does not stop independent code,
tests, analysis preparation, or documentation work. Report the dependency and
continue what can be done honestly; never fabricate human ratings or new
experimental outcomes. Keep the user informed of progress and findings.

This is a one-person research company. Use local or appropriately authorized
API LLM judges for feasible label audits; recruiting human reviewers is not a
prerequisite to continuing research. Freeze a separate LLM-audit protocol,
qualify judges on synthetic controls, preserve disagreement and uncertainty,
and label the outputs as automated judgments rather than human ground truth.
Preserve the existing human-audit protocol as historical work, not an active
staffing requirement.

## Commit and push regularly

The user authorizes regular Git backup of project code. Commit and push after
each coherent, verified milestone, before a handoff or ending a work session,
and at least every 30 minutes of active code changes when meaningful changes
are ready. Use a clearly named `codex/` work branch for unfinished work, record
which checks have passed, and keep unverified changes off `main`. Do not wait
until an entire research program is complete to back up its implementation.

Use the existing `github-work` origin; never switch to a personal identity or
force-push shared history. Fetch before reconciling remote changes. Stage files
by explicit path, review the staged diff, and check all outgoing commits for
secrets, restricted data, and oversized files before pushing. A `.gitignore`
entry does not remove a file already committed in unpublished history. Verify
the pushed branch's remote commit after each push. A failed push must be
reported and retried when the underlying authentication/network issue clears.

Run `python scripts/check_git_payload.py --staged` before committing and
`python scripts/check_git_payload.py --base origin/main` before publishing a
work branch. The checker reads Git objects across outgoing history, rejects
known private/data paths, potential credentials and files over 5 MiB, and
prints paths/reasons only. It supplements explicit diff and release review;
it does not prove that arbitrary prose or source code contains no private data.

Keep reusable code, tests, protocols, and small reviewed release artifacts in
Git. New general-purpose code should not live only under ignored `private/`.
Do not broadly unignore legacy private tools: audit/promote reusable code into
`src/` or `scripts/` while preserving the original frozen copies and bindings.
An explicit embargo or separate result-release gate is not lifted by the
code-backup cadence; push code separately when necessary.

## Keep raw and large data outside Git

New raw data, row-level outputs, model weights, caches, and large run artifacts
belong under `../lexical-prompt-study-data/`, created beside this repository:

- `raw/`: restricted inputs and responses;
- `runs/`: private per-run outputs and receipts;
- `models/` and `cache/`: weights and disposable caches;
- `backups/`: independently verified private backups.

Use `/data2/PRAX/lexical-prompt-study-data` on this server. The directory is
private to the project owner and outside the Git worktree. Do not add its
contents or symlinks to public commits. Existing ignored `private/` evidence
remains in place because frozen runtimes bind its paths and hashes; move it
only through a separately verified migration. Keep `.gitignore` protections
for any temporary data written inside the worktree. Git is the code backup,
not a backup of ignored data: retain verified private copies separately.
