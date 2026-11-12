# MiniMax-H3 packed layout replay

Tests for [PR #666](https://github.com/RL-Align/RL-Kernel/pull/666) at [`ba5ef552`](https://github.com/BPbruce/RL-Kernel/commit/ba5ef5520d3f032b055ec9073673d58e6adf4c4b).

## Results

| Check | Result |
| --- | --- |
| H3 CPU selection and layout compatibility | 233 passed, 10 skipped |
| Windows named-dimension cases | 9 passed |
| Block 0 manual vs schema-based permutation | Block inputs and outputs were byte-identical. [Replay JSON](block-probe-rtx5060.json) |
| Block 0 H-C3 with SDPA math | 84 of 264 output rows differed after restoring row order. Max absolute difference was 2.0. The first difference was in attention. [Per-node JSON](node-trace.json) |
| Manual permutation on `test-h3@07533aee` | Same 84 rows and max absolute difference of 2.0. [Baseline JSON](baseline-07533aee.json) |
| Full-model H-C3 | Not run |

The WSL skips were nine named-dimension cases and one optional Diffusers comparison.

## Environment

- **CPU.** Ubuntu 24.04 on WSL 2 (x86-64), Python 3.10.20, PyTorch `2.14.1+cpu`. The separate named-dimension run used Windows 11, Python 3.14.3 and PyTorch `2.11.0+cpu`.
- **Block 0.** Ubuntu 24.04 on WSL 2 (x86-64), one RTX 5060 (SM 12.0), driver 610.88, Python 3.12.13, PyTorch `2.13.0+cu130` and CUDA 13.0. The repository's provider path used PyTorch SDPA math. Hidden states were BF16, and SDPA math used FP32 accumulation with reduced-precision math reductions disabled.
- **Inputs.** Block 0 and conditioning weights from `MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`, plus seeded synthetic hidden states. The FL2VA replay used batch size 1 and 264 rows. The input and permutation seeds were 0 and 7. Both condition row counts came from the fixture and were 0 for this case.

## CPU commands

Run from a checkout of `ba5ef5520d3f032b055ec9073673d58e6adf4c4b` with the project's test dependencies installed.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m pytest -q -ra -p no:cacheprovider \
  tests/reference/minimax_h3 \
  tests/runtime/test_h3_sigma_schedule_dispatch.py \
  tests/models/minimax_h3/test_h3_packed_layout.py \
  tests/runtime/test_layout_compatibility.py
```

On Windows with the named-dimension-capable PyTorch build:

```powershell
python -B -m pytest -q -ra -p no:cacheprovider tests/models/minimax_h3/test_h3_packed_layout.py -k named
```

## Block 0 commands

Use a CUDA-enabled PyTorch build, NumPy and safetensors. The weight preparation command also needs `huggingface_hub`. Set the paths to this evidence directory, the PR checkout and the extracted weight directory.

```bash
probe_assets=/path/to/this/evidence-directory
h3_source=/path/to/pr-checkout
h3_weights=/path/to/h3-block0-weights

cd "$h3_source"
cp "$probe_assets/run_h3_layout_probe.py" .
python tools/weights/prepare_h3_weights.py --out "$h3_weights" --block

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -B run_h3_layout_probe.py run \
  --weights-dir "$h3_weights" --out block-probe.json

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -B "$probe_assets/trace_hc3.py" \
  --probe-root "$h3_source" --weights-dir "$h3_weights" --out node-trace.json
```

Use a separate checkout at `07533aee8c981473c00d69a847d86f9a2539b0ed` to repeat the manual permutation on `test-h3`.

```bash
h3_baseline=/path/to/test-h3-checkout
git -C "$h3_baseline" rev-parse HEAD > "$h3_baseline/SOURCE_REVISION"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -B "$probe_assets/baseline_provider.py" \
  --source "$h3_baseline" --weights-dir "$h3_weights" --out baseline.json
```
