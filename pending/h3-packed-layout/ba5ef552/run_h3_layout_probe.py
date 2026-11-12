# SPDX-License-Identifier: Apache-2.0

"""Run the #666 layout through the repository's MiniMax-H3 block 0 replay."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from rl_engine.models.minimax_h3.packed_layout import H3PackedLayout
from rl_engine.validation.models import h3_block, h3_provider
from rl_engine.validation.models.h3_cases import H3_BLOCK_LAYOUTS, h3_fl2va_layout, h3_timesteps
from rl_engine.validation.models.h3_weights import (
    WEIGHTS_ENV,
    load_h3_block_weights,
    load_h3_manifest,
    sha256_file,
    sha256_tensor,
)

ROOT = Path(__file__).resolve().parent
REVISION = "ba5ef5520d3f032b055ec9073673d58e6adf4c4b"
FIXTURE = ROOT / "tests/reference/minimax_h3/packed_layout/upstream_cases.json"
CASE = "fl2va_block_tiny"
SEED = 0
STREAMS = ("text", "video", "audio")


def same_bytes(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


def make_layout(manifest):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = next(case for case in fixture["cases"] if case["case_id"] == CASE)
    fields = case["expected"]["fields"]
    source_indices = {
        stream: torch.tensor(fields[f"{stream}_indices"]["values"], dtype=torch.int64)
        for stream in STREAMS
    }
    packed = h3_fl2va_layout(**H3_BLOCK_LAYOUTS["tiny"], device="cpu")
    pos_spec = fields["position_ids"]
    sealed_positions = torch.from_numpy(
        np.frombuffer(bytes.fromhex(pos_spec["data_le_hex"]), dtype="<f8").copy()
    ).reshape(pos_spec["shape"])
    sealed_tags = torch.tensor(fields["token_tags"]["values"], dtype=torch.int64)
    if not same_bytes(packed["position_ids"], sealed_positions) or not same_bytes(
        packed["token_tags"], sealed_tags
    ):
        raise ValueError("The block replay positions or tags differ from the sealed fixture")
    if fixture["checkpoint_revision"] != manifest["model_identity"]["revision"]:
        raise ValueError("The layout fixture and block weights pin different checkpoints")

    # Use the timestep inputs from the block replay together with the sealed selectors.
    timestep = h3_timesteps(h3_block.NUM_TIMESTEPS, seed=SEED, device="cpu")
    layout = H3PackedLayout(
        **packed,
        **{f"{stream}_indices": indices for stream, indices in source_indices.items()},
        timestep=timestep,
        num_condition_video_rows=case["expected"]["num_condition_video_rows"],
        num_condition_audio_rows=case["expected"]["num_condition_audio_rows"],
        provenance={
            "checkpoint_revision": fixture["checkpoint_revision"],
            "runtime_revision": REVISION,
            "task_family": case["task_family"],
            "preparation": {"fixture": CASE, "diffusers_revision": fixture["upstream"]["commit"]},
        },
    )
    # The existing block permutation probe uses seed + 7.
    perm = torch.randperm(sealed_tags.numel(), generator=torch.Generator().manual_seed(SEED + 7))
    moved = layout.permute_rows(perm)
    old_to_new = perm.argsort()
    expected = {name: rows[perm] for name, rows in packed.items()}
    expected.update({f"{s}_indices": old_to_new[rows] for s, rows in source_indices.items()})
    expected["timestep"] = timestep
    expected["adaln_indices"] = expected["timestep_indices"] * 3 + expected["token_tags"]
    mismatches = [
        name for name, rows in expected.items() if not same_bytes(getattr(moved, name), rows)
    ]
    if mismatches:
        raise ValueError(
            f"Schema metadata differs from manual permutation: {', '.join(mismatches)}"
        )
    return layout, moved, perm, source_indices


def consumer_inputs(context, layout, moved, perm, source_indices):
    """Compare direct row indexing with scatter, AdaLN and RoPE from the schema."""
    hidden = context["hidden"]
    device = hidden.device
    perm = perm.to(device)
    if not same_bytes(context["index"], layout.adaln_indices.to(device)) or not same_bytes(
        context["position_ids"], layout.position_ids.to(device)
    ):
        raise ValueError("Canonical layout differs from the block replay metadata")

    manual = {
        "hidden": hidden.index_select(1, perm),
        "index": context["index"][perm],
        "position_ids": context["position_ids"][perm],
        "rope": tuple(table[perm] for table in context["rope"]),
    }
    scattered = torch.empty_like(hidden)
    for stream, indices in source_indices.items():
        source = hidden.index_select(1, indices.to(device))
        scattered[:, getattr(moved, f"{stream}_indices").to(device)] = source
    positions = moved.position_ids.to(device)
    schema = {
        "hidden": scattered,
        "index": moved.adaln_indices.to(device),
        "position_ids": positions,
        "rope": h3_provider.provider_rope(positions),
    }
    matches = {
        name: same_bytes(manual[name], schema[name]) for name in ("hidden", "index", "position_ids")
    }
    matches["rope"] = all(
        same_bytes(a, b) for a, b in zip(manual["rope"], schema["rope"], strict=True)
    )
    return {**context, **manual}, {**context, **schema}, matches


def load_weights(args, manifest):
    if args.weights_dir is not None:
        os.environ[WEIGHTS_ENV] = str(args.weights_dir.expanduser().resolve())
        return load_h3_block_weights("cpu")

    from safetensors import safe_open

    # Verify every consumed tensor against the manifest, even when opening a raw shard.
    specs = {**manifest["tensors"], **manifest["block_tensors"]}
    weights = {}
    with safe_open(str(args.shard.expanduser()), framework="pt", device="cpu") as shard:
        shard_names = set(shard.keys())
        for name, spec in specs.items():
            if name not in shard_names:
                raise ValueError(f"{name}: missing from {args.shard}")
            weight = shard.get_tensor(name)
            if list(weight.shape) != spec["shape"] or str(weight.dtype) != f"torch.{spec['dtype']}":
                raise ValueError(f"{name}: shape or dtype differs from the manifest")
            if sha256_tensor(weight) != spec["sha256"]:
                raise ValueError(f"{name}: SHA-256 differs from the manifest")
            weights[name] = weight
    return weights


def compare_outputs(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Block outputs differ in shape or dtype")
    unequal = left.contiguous().view(torch.uint8) != right.contiguous().view(torch.uint8)
    rows = unequal.reshape(*left.shape[:2], -1).any(dim=-1).any(dim=0).nonzero().flatten()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    max_abs = (left.float() - right.float()).abs().max().item() if finite else None
    return {
        "byte_equal": rows.numel() == 0,
        "finite": finite,
        "different_rows": rows.numel(),
        "first_different_row": rows[0].item() if rows.numel() else None,
        "max_abs": max_abs if max_abs is not None and math.isfinite(max_abs) else None,
    }


@torch.no_grad()
def run_forward(weights, device, layout, moved, perm, source_indices):
    context = h3_block.make_block_context(weights, layout="tiny", seed=SEED)
    # The provider graph only reads the provider RoPE tables.
    del context["rope_fp32"]
    manual, schema, matches = consumer_inputs(context, layout, moved, perm, source_indices)
    print(f"Consumer inputs: {matches}", flush=True)
    with sdpa_kernel(SDPBackend.MATH):
        print(f"Running three block 0 forwards on {device}", flush=True)
        baseline_out = h3_block.run_graph("provider", context)[h3_block.OUTPUT]
        manual_out = h3_block.run_graph("provider", manual)[h3_block.OUTPUT]
        schema_out = h3_block.run_graph("provider", schema)[h3_block.OUTPUT]
    restored = schema_out.index_select(1, perm.argsort().to(device))
    return {
        "consumer_inputs_byte_equal": matches,
        "manual_vs_schema_permuted": compare_outputs(manual_out, schema_out),
        "canonical_vs_restored": compare_outputs(baseline_out, restored),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("check-layout", help="exercise CPU metadata and consumer inputs")
    for command in ("check-weights", "run"):
        subcommand = subcommands.add_parser(command)
        source = subcommand.add_mutually_exclusive_group(required=True)
        source.add_argument("--weights-dir", type=Path, help="directory of extracted pinned files")
        source.add_argument(
            "--shard", type=Path, help="path to the released first safetensors shard"
        )
        if command == "run":
            subcommand.add_argument("--out", type=Path, required=True)
            subcommand.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    manifest = load_h3_manifest()
    layout, moved, perm, source_indices = make_layout(manifest)
    print(f"CPU schema matches manual permutation for {perm.numel()} rows", flush=True)
    if args.command == "check-layout":
        # Unique packed-row markers make a selector error visible in the consumer inputs.
        context = {
            "hidden": torch.arange(perm.numel()).view(1, -1, 1),
            "index": layout.adaln_indices,
            "position_ids": layout.position_ids,
            "rope": h3_provider.provider_rope(layout.position_ids),
        }
        _, _, matches = consumer_inputs(context, layout, moved, perm, source_indices)
        print(f"CPU consumer inputs: {matches}")
        return 0 if all(matches.values()) else 1

    if args.command == "run" and args.device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            parser.error("run requires a CUDA-enabled PyTorch and a BF16-capable GPU")
    print("Checking block 0 and conditioning tensor SHA-256 against the manifest", flush=True)
    weights = load_weights(args, manifest)
    print(f"Verified {len(weights)} pinned tensors", flush=True)
    if args.command == "check-weights":
        return 0

    device = torch.device(args.device)
    weights = {name: weight.to(device) for name, weight in weights.items()}
    result = run_forward(weights, device, layout, moved, perm, source_indices)
    evidence = {
        "rl_kernel_revision": REVISION,
        "packed_layout_source_sha256": sha256_file(
            ROOT / "rl_engine/models/minimax_h3/packed_layout.py"
        ),
        "probe_sha256": sha256_file(Path(__file__)),
        "model_revision": manifest["model_identity"]["revision"],
        "weights_verified": len(weights),
        "diffusers_layout_revision": layout.provenance["preparation"]["diffusers_revision"],
        "case": CASE,
        "batch_size": 1,
        "num_rows": perm.numel(),
        "source_lengths": layout.source_lengths,
        "num_condition_video_rows": layout.num_condition_video_rows,
        "num_condition_audio_rows": layout.num_condition_audio_rows,
        "seed": SEED,
        "permutation_seed": SEED + 7,
        "permutation": perm.tolist(),
        "input": "seeded synthetic BF16 hidden rows, real pinned block 0 weights",
        "mode": "provider",
        "attention": "PyTorch SDPA math",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if device.type == "cuda" else str(device),
            "cuda_capability": (
                torch.cuda.get_device_capability() if device.type == "cuda" else None
            ),
        },
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Manual vs schema: {result['manual_vs_schema_permuted']}")
    print(f"Canonical vs restored: {result['canonical_vs_restored']}")
    print(f"Evidence: {args.out}")
    direct = result["manual_vs_schema_permuted"]
    return (
        0
        if all(result["consumer_inputs_byte_equal"].values())
        and direct["byte_equal"]
        and direct["finite"]
        and result["canonical_vs_restored"]["finite"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
