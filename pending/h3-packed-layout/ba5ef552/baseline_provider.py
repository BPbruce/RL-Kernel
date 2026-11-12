# SPDX-License-Identifier: Apache-2.0

"""Run the existing provider graph directly from test-h3 without the schema."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def compare(canonical, shuffled, inverse):
    restored = shuffled.index_select(1, inverse)
    unequal = canonical.contiguous().view(torch.uint8) != restored.contiguous().view(torch.uint8)
    rows = unequal.reshape(1, inverse.numel(), -1).any(dim=-1)
    return {
        "different_rows": int(rows.sum()),
        "max_abs": float((canonical.float() - restored.float()).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))

    from rl_engine.validation.models import h3_block
    from rl_engine.validation.models.h3_weights import WEIGHTS_ENV, load_h3_block_weights

    os.environ[WEIGHTS_ENV] = str(args.weights_dir.resolve())
    weights = load_h3_block_weights("cuda")
    context = h3_block.make_block_context(weights, layout="tiny", seed=0)
    perm = torch.randperm(context["seq_len"], generator=torch.Generator().manual_seed(7)).to("cuda")
    moved = {
        **context,
        "hidden": context["hidden"].index_select(1, perm),
        "index": context["index"].index_select(0, perm),
        "position_ids": context["position_ids"].index_select(0, perm),
        "rope": tuple(x.index_select(0, perm) for x in context["rope"]),
    }
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        canonical = h3_block.run_graph("provider", context)
        shuffled = h3_block.run_graph("provider", moved)

    inverse = perm.argsort()
    report = {
        "rl_kernel_revision": (args.source / "SOURCE_REVISION").read_text().strip(),
        "attention": compare(canonical["attention"], shuffled["attention"], inverse),
        "block_output": compare(canonical[h3_block.OUTPUT], shuffled[h3_block.OUTPUT], inverse),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
