from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

from .behavior import _as_token_ids
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text

BOUNDARY_MARKER = "<<<FACTORIAL_BLOCK_BOUNDARY>>>"
MATERIAL_FILES = {
    "full_scaffold": "full_scaffold.private.txt",
    "structural_sham": "structural_sham.private.txt",
    "inert_length": "inert_length.private.txt",
}


def _private_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.chmod(0o600)
    temporary.replace(path)
    return sha256_bytes(payload)


def _private_json(path: Path, payload: Any) -> str:
    return _private_bytes(path, canonical_json_bytes(payload))


def prepare_factorial_block_review(
    *,
    tokenizer: Any,
    tokenizer_revision: str,
    output_root: Path,
) -> dict[str, Any]:
    from .followup_generation import _split_scaffold
    from .scaffolds import render_arms

    safe_request = "Describe checksum verification in one sentence."
    rendered = render_arms(safe_request, tokenizer)
    arm_to_material = {
        "full": "full_scaffold",
        "structural_sham": "structural_sham",
        "inert_length": "inert_length",
    }
    materials = {
        material: _split_scaffold(rendered[arm].text, safe_request)
        for arm, material in arm_to_material.items()
    }
    if BOUNDARY_MARKER in "".join(materials.values()):
        raise ValueError("factorial block marker collides with restricted material")
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.chmod(0o700)
    manifest_materials = {}
    for material, filename in MATERIAL_FILES.items():
        text = materials[material]
        path = output_root / filename
        _private_bytes(path, text.encode("utf-8"))
        manifest_materials[material] = {
            "path": str(path),
            "original_text_sha256": sha256_text(text),
            "canonical_token_count": len(
                _as_token_ids(tokenizer.encode(text, add_special_tokens=False))
            ),
            "review_status": "insert_boundaries_then_compile",
        }
    manifest = {
        "schema_version": "1.0",
        "status": "human_semantic_block_review_required",
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_chat_template_sha256": sha256_text(
            getattr(tokenizer, "chat_template", None) or ""
        ),
        "boundary_marker": BOUNDARY_MARKER,
        "instructions": (
            "A human reviewer must insert the exact boundary marker at every "
            "semantic block boundary in each private text file. Do not edit any "
            "other byte. Use corresponding block order in full, sham, and inert. "
            "The compiler removes markers, verifies the original hashes, and "
            "fails unless all cumulative pinned-token counts match exactly."
        ),
        "agent_plaintext_inspection_forbidden": True,
        "materials": manifest_materials,
    }
    manifest_path = output_root / "manifest.private.json"
    manifest_sha256 = _private_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "material_text_sha256": {
            material: row["original_text_sha256"]
            for material, row in manifest_materials.items()
        },
        "canonical_token_counts": {
            material: row["canonical_token_count"]
            for material, row in manifest_materials.items()
        },
        "raw_text_returned": False,
    }


def compile_factorial_block_review(
    *,
    manifest_path: Path,
    tokenizer: Any,
    output_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema_version"] != "1.0"
        or manifest["status"] != "human_semantic_block_review_required"
        or manifest["boundary_marker"] != BOUNDARY_MARKER
        or manifest["agent_plaintext_inspection_forbidden"] is not True
    ):
        raise ValueError("factorial block-review manifest drift")
    compiled: dict[str, list[dict[str, Any]]] = {}
    cumulative_counts: dict[str, list[int]] = {}
    for material in MATERIAL_FILES:
        row = manifest["materials"][material]
        path = Path(row["path"])
        marked = path.read_text()
        if BOUNDARY_MARKER not in marked:
            raise ValueError(f"{material}: no human semantic boundary marker")
        original = marked.replace(BOUNDARY_MARKER, "")
        if sha256_text(original) != row["original_text_sha256"]:
            raise ValueError(f"{material}: non-boundary bytes changed during review")
        blocks = marked.split(BOUNDARY_MARKER)
        if any(not block for block in blocks):
            raise ValueError(f"{material}: empty semantic block")
        compiled[material] = [
            {
                "block_id": f"block-{index:03d}",
                "text": block,
                "text_sha256": sha256_text(block),
            }
            for index, block in enumerate(blocks, start=1)
        ]
        cumulative_counts[material] = [
            len(
                _as_token_ids(
                    tokenizer.encode(
                        "".join(blocks[:index]),
                        add_special_tokens=False,
                    )
                )
            )
            for index in range(1, len(blocks) + 1)
        ]
    block_counts = {material: len(blocks) for material, blocks in compiled.items()}
    if len(set(block_counts.values())) != 1:
        raise ValueError(f"semantic block-count mismatch: {block_counts}")
    reference = cumulative_counts["full_scaffold"]
    mismatches = {
        material: counts
        for material, counts in cumulative_counts.items()
        if counts != reference
    }
    if mismatches:
        raise ValueError(
            "semantic block cumulative-token mismatch; "
            f"counts={cumulative_counts}"
        )
    payload = {
        "schema_version": "1.0",
        "status": "human_semantic_blocks_compiled_exact_token_match",
        "source_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_revision": manifest["tokenizer_revision"],
        "block_count": next(iter(block_counts.values())),
        "cumulative_token_counts": reference,
        "scaffold_materials": {
            material: {"blocks": blocks} for material, blocks in compiled.items()
        },
    }
    output_sha256 = _private_json(output_path, payload)
    return {
        "status": payload["status"],
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "block_count": payload["block_count"],
        "cumulative_token_counts": reference,
        "raw_text_returned": False,
    }


def write_factorial_block_review_aid(
    *,
    manifest_path: Path,
    tokenizer: Any,
    output_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema_version"] != "1.0"
        or manifest["status"] != "human_semantic_block_review_required"
        or manifest["boundary_marker"] != BOUNDARY_MARKER
        or manifest["agent_plaintext_inspection_forbidden"] is not True
    ):
        raise ValueError("factorial block-review manifest drift")

    panels: list[str] = []
    summary: dict[str, Any] = {}
    for material in MATERIAL_FILES:
        row = manifest["materials"][material]
        marked = Path(row["path"]).read_text()
        segments = marked.split(BOUNDARY_MARKER)
        if len(segments) < 2 or any(not segment for segment in segments):
            raise ValueError(f"{material}: invalid human boundary markers")
        original = "".join(segments)
        if sha256_text(original) != row["original_text_sha256"]:
            raise ValueError(f"{material}: non-boundary bytes changed during review")

        encoded = tokenizer(
            original,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        token_ids = _as_token_ids(encoded["input_ids"])
        if len(offsets) != len(token_ids):
            raise ValueError(f"{material}: tokenizer offset count mismatch")
        if not offsets or offsets[-1][1] != len(original):
            raise ValueError(f"{material}: tokenizer offsets do not cover material")
        if any(
            start < 0 or end < start or end > len(original)
            for start, end in offsets
        ):
            raise ValueError(f"{material}: invalid tokenizer offsets")

        boundary_positions: list[int] = []
        consumed = 0
        for segment in segments[:-1]:
            consumed += len(segment)
            boundary_positions.append(consumed)
        current_counts = [
            len(
                _as_token_ids(
                    tokenizer.encode(
                        original[:position],
                        add_special_tokens=False,
                    )
                )
            )
            for position in boundary_positions
        ]
        current_counts.append(len(token_ids))

        events: dict[int, list[str]] = {}
        for index, (_start, end) in enumerate(offsets, start=1):
            events.setdefault(end, []).append(
                f'<span class="token-index">T{index:03d}</span>'
            )
        for index, position in enumerate(boundary_positions, start=1):
            events.setdefault(position, []).append(
                f'<span class="current-boundary">CURRENT B{index}</span>'
            )

        rendered: list[str] = []
        cursor = 0
        for position in sorted(events):
            rendered.append(escape(original[cursor:position]))
            rendered.extend(events[position])
            cursor = position
        rendered.append(escape(original[cursor:]))

        panels.append(
            "\n".join(
                (
                    '<section class="material">',
                    f"<h2>{escape(material.replace('_', ' ').title())}</h2>",
                    "<p><strong>Current cumulative token counts:</strong> "
                    f"{escape(str(current_counts))}</p>",
                    '<div class="annotated">',
                    "".join(rendered),
                    "</div>",
                    "</section>",
                )
            )
        )
        summary[material] = {
            "marker_count": len(boundary_positions),
            "current_cumulative_token_counts": current_counts,
            "canonical_token_count": len(token_ids),
        }

    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Private factorial semantic-block review</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f7f4f0;color:#2c2924}
.warning{max-width:1100px;padding:14px 16px;background:#f3e8e0;border:1px solid #a67c52;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;align-items:start}
.material{min-width:0}
.annotated{white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,monospace;font-size:14px;line-height:2.05;background:#fff;border:1px solid #c4b8a8;border-radius:8px;padding:14px}
.token-index{font-size:8px;vertical-align:super;color:#4b6787;margin:0 1px;user-select:none}
.current-boundary{display:inline-block;font:700 10px system-ui,sans-serif;color:#fff;background:#a67c52;padding:1px 4px;margin:0 2px;border-radius:3px;user-select:none}
@media(max-width:1000px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<h1>Human-only semantic block review</h1>
<div class="warning">
This page contains restricted plaintext. Open it only in the local human
review surface. Choose corresponding semantic boundaries that end after the
same token index in all three panels. In the source files, insert the exact
bare marker without spaces or backticks:
<code>&lt;&lt;&lt;FACTORIAL_BLOCK_BOUNDARY&gt;&gt;&gt;</code>.
Do not edit any other byte.
</div>
<div class="grid">
""" + "\n".join(panels) + """
</div>
</body>
</html>
"""
    output_sha256 = _private_bytes(output_path, document.encode("utf-8"))
    return {
        "status": "human_only_factorial_block_review_aid_written",
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "materials": summary,
        "raw_text_returned": False,
    }
