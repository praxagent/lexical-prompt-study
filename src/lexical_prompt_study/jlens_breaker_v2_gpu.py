from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .jlens_breaker_v2_runner import run_jlens_breaker_v2_prefill_loop
from .weaponization_gpu import WeaponizationPrefillRuntime


def run_jlens_breaker_v2_prefill_gpu(
    *,
    public_plan_path: Path,
    private_topology_path: Path,
    authorization_path: Path,
    probe_plan_path: Path,
    model_path: str,
    lens_path: Path,
    sae_path: Path,
    inherited_material_path: Path,
    output_root: Path,
    run_id: str,
    batch_size: int,
) -> dict[str, Any]:
    public_plan = json.loads(public_plan_path.read_text())
    runtime: WeaponizationPrefillRuntime | None = None

    def execute_batch(
        observations: Sequence[dict[str, Any]], attempt: int
    ) -> Sequence[Mapping[str, Any]]:
        nonlocal runtime
        if runtime is None:
            runtime = WeaponizationPrefillRuntime(
                public_plan=public_plan,
                probe_plan_path=probe_plan_path,
                model_path=model_path,
                lens_path=lens_path,
                sae_path=sae_path,
                factorial_material_path=inherited_material_path,
            )
        return runtime.execute_batch(observations, attempt)

    return run_jlens_breaker_v2_prefill_loop(
        public_plan_path=public_plan_path,
        private_topology_path=private_topology_path,
        authorization_path=authorization_path,
        output_root=output_root,
        run_id=run_id,
        batch_size=batch_size,
        execute_batch=execute_batch,
    )
