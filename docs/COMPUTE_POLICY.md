# Compute authorization and persistence policy

Status: active planning policy

Date: 2026-07-25

## Authorization envelope

- Soft escalation threshold: **$50 cumulative**
- Hard cumulative ceiling: **$100**
- Python environment and commands: **uv**
- GPU provider: **RunPod**
- Durable working set: **task-owned RunPod network volume**
- Expected volume retention: **a few days, reviewed daily**

The soft threshold and hard ceiling have different meanings.

At the soft threshold:

1. checkpoint the current atomic unit;
2. retrieve or sync critical receipts;
3. stop launching new phases;
4. report actual spend, current progress, remaining work, and revised estimate;
5. request approval to raise the soft threshold.

A healthy, progressing atomic unit may continue while a response is pending if
stopping at that instant would waste the authorized work and projected spend
remains below the hard ceiling. This does not authorize another pod, retry,
parameter sweep, or phase.

At the hard ceiling, paid compute stops at the next safe checkpoint and the pod
is terminated after evidence retrieval. A duplicate, runaway, idle,
misconfigured, or no-progress job is terminated earlier regardless of budget.

## Per-run approval

The overall envelope does not replace a per-run cost statement. Before creating
each pod, record:

- exact GPU type and count;
- on-demand or interruptible status;
- observed `$ / hour`;
- measured unit throughput and expected units;
- expected and worst-case duration;
- expected and worst-case pod cost;
- network-volume size and standing rate;
- no-progress timeout;
- pod wall-time limit;
- task-owned volume and pod IDs once created.

## Persistent network volume

The network volume is a separately billed, separately owned resource. It stores:

- the pinned Llama checkpoint and tokenizer;
- the pinned Jacobian-lens artifact;
- the pinned Goodfire SAE;
- download and file-hash manifests;
- uv caches or wheels worth preserving;
- immutable completed trial shards;
- resumable checkpoints and append-only attempt logs.

It does not store the only copy of irreplaceable results. Completed receipts are
hashed remotely, retrieved locally, verified, and recorded in the run ledger.

The volume must have:

- a provider-issued ID and datacenter recorded before pod attachment;
- a capacity budget derived from actual artifact sizes plus working headroom;
- explicit cache paths such as `HF_HOME` resolved onto the mount;
- a small write/read and free-space check before large downloads;
- an owner, purpose, creation time, daily cost, review time, and deletion
  deadline;
- verified deletion evidence when the active work window closes.

Pod termination must never be conflated with volume deletion.

## Checkpoint and resume contract

Checkpoint granularity is the smallest stable trial unit that is economical to
repeat. Each completed unit:

1. writes to a temporary path;
2. closes and hashes all files;
3. atomically renames into the completed-shard namespace;
4. updates an append-only attempt ledger;
5. records stable trial IDs and the source/plan hashes.

Resume verifies hashes, skips completed units, rejects duplicates, and never
mutates an immutable completed shard. A completion marker is written only after
all expected units and hashes pass.

Before interruptible capacity is allowed, the exact resume path must pass a
forced `kill -9` restart test at minimum scale.

## Local continuity file

`checkpoint.md` is deliberately ignored by Git. It records live operational
state that would be unsafe or noisy in the public history, including:

- current phase and last verified Git commit;
- exact task-owned pod and volume IDs;
- remote paths and process IDs;
- completed and expected trial counts;
- last checkpoint and receipt hashes;
- cumulative and projected spend;
- next safe action;
- teardown or retention deadline.

Secrets never enter `checkpoint.md`.
