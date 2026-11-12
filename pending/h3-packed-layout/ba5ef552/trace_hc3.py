# SPDX-License-Identifier: Apache-2.0

"""Trace row-order drift in the pinned MiniMax-H3 block 0 probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def compare_nodes(probe, canonical, manual, schema, inverse):
    nodes = {}
    for node in probe.h3_block.NODES:
        name = node.name
        base, ref, moved = (run[name] for run in (canonical, manual, schema))
        if name == "adaln_projection":
            # AdaLN projections are shared timestep/modality tables, without a packed row axis.
            restored = all(probe.same_bytes(a, b) for a, b in zip(base, ref, strict=True))
            schema_matches = all(probe.same_bytes(a, b) for a, b in zip(ref, moved, strict=True))
            comparison = {"byte_equal": restored}
        else:
            comparison = probe.compare_outputs(base, ref.index_select(1, inverse))
            schema_matches = probe.same_bytes(ref, moved)
        nodes[name] = {
            "canonical_vs_manual_restored": comparison,
            "manual_vs_schema_byte_equal": schema_matches,
        }
    return nodes


@torch.no_grad()
def trace(probe, weights_dir, check_controls=False):
    manifest = probe.load_h3_manifest()
    layout, moved, perm, selectors = probe.make_layout(manifest)
    weights = probe.load_weights(SimpleNamespace(weights_dir=weights_dir), manifest)
    weights = {name: value.to("cuda") for name, value in weights.items()}
    context = probe.h3_block.make_block_context(weights, layout="tiny", seed=probe.SEED)
    manual_context, schema_context, matches = probe.consumer_inputs(
        context, layout, moved, perm, selectors
    )
    if not all(matches.values()):
        raise RuntimeError(f"The two permutation paths disagree on inputs: {matches}")
    with sdpa_kernel(SDPBackend.MATH):
        canonical = probe.h3_block.run_graph("provider", context)
        manual = probe.h3_block.run_graph("provider", manual_context)
        schema = probe.h3_block.run_graph("provider", schema_context)
    inverse = perm.argsort().to("cuda")
    nodes = compare_nodes(probe, canonical, manual, schema, inverse)
    first = next(
        (
            name
            for name, comparison in nodes.items()
            if not comparison["canonical_vs_manual_restored"]["byte_equal"]
        ),
        None,
    )
    report = {
        "rl_kernel_revision": probe.REVISION,
        "model_revision": manifest["model_identity"]["revision"],
        "torch": torch.__version__,
        "torch_revision": torch.version.git_version,
        "device": torch.cuda.get_device_name(),
        "attention": "PyTorch SDPA math",
        "num_rows": perm.numel(),
        "consumer_inputs_byte_equal": matches,
        "first_drifting_node": first,
        "nodes": nodes,
    }
    if check_controls:
        report["full_block_controls"] = check_full_block(
            probe, context, manual_context, canonical[probe.h3_block.OUTPUT], inverse
        )
    return report, canonical, manual, perm


def check_full_block(probe, context, manual_context, baseline, inverse):
    """Change only the attention calculation and re-run the full block."""
    attention = probe.h3_provider.provider_attention
    output = probe.h3_block.OUTPUT

    def canonical_key_order(q, k, v):
        return attention(q, k.index_select(1, inverse), v.index_select(1, inverse))

    def fp64_attention(q, k, v):
        return attention(q.double(), k.double(), v.double()).to(q.dtype)

    with sdpa_kernel(SDPBackend.MATH):
        with patch.object(probe.h3_provider, "provider_attention", canonical_key_order):
            canonical_kv = probe.h3_block.run_graph("provider", manual_context)[output]
        with patch.object(probe.h3_provider, "provider_attention", fp64_attention):
            fp64_canonical = probe.h3_block.run_graph("provider", context)[output]
            fp64_moved = probe.h3_block.run_graph("provider", manual_context)[output]

    return {
        "restoring_only_key_value_order": probe.compare_outputs(
            baseline, canonical_kv.index_select(1, inverse)
        ),
        "both_orders_using_fp64_attention_then_bf16": probe.compare_outputs(
            fp64_canonical, fp64_moved.index_select(1, inverse)
        ),
        "canonical_fp64_attention_vs_original_provider": probe.compare_outputs(
            baseline, fp64_canonical
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--capture-attention", type=Path, help="save canonical Q/K/V and permutation"
    )
    parser.add_argument("--check-controls", action="store_true")
    parser.add_argument("--require-hc3", action="store_true", help="exit 1 when H-C3 differs")
    args = parser.parse_args()

    sys.path.insert(0, str(args.probe_root.resolve()))
    import run_h3_layout_probe as probe

    report, canonical, _, perm = trace(probe, args.weights_dir, args.check_controls)
    if args.capture_attention is not None:
        from safetensors.torch import save_file

        capture = {
            "q": canonical["rope_q"],
            "k": canonical["rope_k"],
            "v": probe.h3_block._heads(canonical["v_proj"]),
            "perm": perm,
        }
        save_file(
            {name: value.cpu().contiguous() for name, value in capture.items()},
            args.capture_attention,
        )
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, node in report["nodes"].items():
        comparison = node["canonical_vs_manual_restored"]
        print(
            f"{name:18} restored_equal={comparison['byte_equal']} "
            f"different_rows={comparison.get('different_rows', '-')} "
            f"max_abs={comparison.get('max_abs', '-')} "
            f"manual_schema_equal={node['manual_vs_schema_byte_equal']}"
        )
    print(f"First drift: {report['first_drifting_node']}")
    if args.check_controls:
        print(json.dumps(report["full_block_controls"], indent=2, allow_nan=False))
    print(f"Evidence: {args.out}")
    if args.require_hc3 and report["first_drifting_node"] is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
