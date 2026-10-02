# LGWM: A Latent World Model that Verifies GUI Agent Transitions

Official code for **Right Screen, Wrong Transition: World Models as Verifiers for GUI Agents**.

Jiaming Zhang, Xuan Wang, Fuyao Zhang, Yang Cao, Lingjuan Lyu, Wei Yang Bryan Lim

[Project page](https://jiamingzhang94.github.io/lgwm/) · [Model weights](https://huggingface.co/jiamingzz/lgwm) · [Data](https://huggingface.co/datasets/jiamingzz/lgwm-data)

LGWM is a decoder-free, action-conditioned world model for GUI agents. Given the
current screenshot and an action, it predicts the representation of the next
screen, and compares it with the screen that actually appears. One cosine
distance tells whether the transition is the one the action should have produced.

## Overview

A login screen that appears after a tap on *Sign in* is expected; the same screen
after a tap on *View order* is an attack. For GUI agents, safety is therefore a
property of the transition rather than of the screen, and a monitor that inspects
only screens can be defeated by reusing a legitimate one. Judging a transition
requires an expectation of what should have followed the action. Existing GUI
world models provide one, but they output it as text, code or images, so checking
it against the observed screen requires a second model to judge the two.

LGWM instead predicts in the space in which observations are encoded. It is
trained without semantic annotation on 1.85M real GUI transitions. Verification
reduces to a vector comparison, and the same signal reveals whether a mismatch is
harmful. We evaluate on RSWT-Bench, a diagnostic where each credential screen
appears under both a legitimate and a hijacked transition, so detectors that see
only the screen are at chance by construction.

## Key results

- **0.987 AUC** on RSWT-Bench with a training-free integrity score, on par with the
  strongest closed-source VLMs.
- **17 ms** per decision on one A100, over three orders of magnitude faster than
  generative GUI world models, and about ten AUC points more accurate.
- **0.954 AUC** at separating harmful from benign violations with a linear head on
  the prediction residual, where prompted VLMs are near chance.
- **173M** parameters at inference (ViT-B encoder, action encoder and predictor).

## Install

Python 3.11+; CUDA for training. Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -c requirements-tested.txt -e '.[text]'
```

## Weights and inference

```bash
hf download jiamingzz/lgwm --local-dir weights
```

| Directory | Model |
| --- | --- |
| `weights/lgwm-online/` | LGWM world model |
| `weights/lgwm-harm-head/` | Supervised harm head |

Save the action as `action.json`. For a tap at the center of the screen:

```json
{"action_type": "tap", "x": 0.5, "y": 0.5}
```

Coordinates range from 0 to 1. Supported `action_type` values are `tap`,
`long_press`, `swipe`, `scroll`, `type_text`, `key`, `open_app`, `wait`, `click`,
`hover` and `drag`. `swipe` and `drag` also take an end point `x2`, `y2`; `scroll`
takes `direction` (`up`, `down`, `left` or `right`); `type_text` takes `text`.
Then score the transition:

```bash
python tools/score_transition.py --before before.png --after after.png \
  --action action.json --harm-head weights/lgwm-harm-head
```

The command prints `integrity` and `harm_score`. Higher values indicate a less
expected transition and a higher predicted hijack probability, respectively.
Omit `--harm-head` to compute only integrity. Use `--device cpu` to run on CPU.

## Data

Training and evaluation use processed transitions from five sources:
[AITW](https://github.com/google-research/google-research/tree/master/android_in_the_wild),
[AndroidControl](https://github.com/google-research/google-research/tree/master/android_control),
[GUIOdyssey](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey),
[AMEX](https://huggingface.co/datasets/Yuxiang007/AMEX) and
[MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus).

```bash
pip install -c requirements-tested.txt -e '.[training,evaluation]'
hf download jiamingzz/lgwm-data --repo-type dataset --local-dir data
```

| Path | Content |
| --- | --- |
| `data/processed/` | Screenshot shards (256×512 WebP) with actions |
| `data/index/` | Transition indexes for each source and split |
| `data/text_table.npz` | Action-text embeddings |

The corpus has 1,824,824 training and 20,835 validation transitions.

## Evaluation

RSWT-Bench is defined by `benchmark/gui_hijack_test.jsonl` (966 examples)
and `benchmark/gui_hijack_train.jsonl` (6,786 examples for the harm head). Each
record references transitions in the indexes:

| Field | Meaning |
| --- | --- |
| `base` | Transition supplying the current screen and action |
| `obs` | Transition supplying the observed next screen |
| `pair_id` | Hijacked/paired-safe example pair |
| `donor_id` | Replacement-screen identity used for the paired bootstrap |
| `cls` | 1 hijacked, 2 paired safe, 3 legitimate credential, 4 surprising benign, 5 benign mismatch |

```bash
python tools/extract_features.py --manifest benchmark/gui_hijack_test.jsonl \
  --data-root data --index-dir data/index --output outputs/features/test
python tools/evaluate_benchmark.py --features outputs/features/test \
  --output outputs/eval/test.json
```

Results and per-example scores are saved under `outputs/eval/`:

| AUC | Integrity | Harm head |
| --- | ---: | ---: |
| RSWT | 0.9865 | 0.9296 |
| Harm | 0.5139 | 0.9542 |

## Training

Training starts from DINOv2, adapts the visual encoder to GUI screenshots in
Stage A, then trains the world model. Stage A takes two epochs; the main recipe
uses 120,000 updates with a global batch of 512 on four GPUs. Settings are in
[configs/lgwm_main.yaml](configs/lgwm_main.yaml).

```bash
python tools/train_stage_a.py --data-root data --index-dir data/index --run-name stage_a
python tools/export_stage_a_init.py --checkpoint outputs/runs/stage_a/stage_a_encoder.pt
python -m torch.distributed.run --nproc_per_node=4 tools/train.py \
  --data-root data --run-name lgwm_main
python tools/export_weights.py --checkpoint outputs/runs/lgwm_main/ckpt/final.pt \
  --output weights/retrained-online
```

Fit the harm head on the benchmark training split:

```bash
python tools/extract_features.py --manifest benchmark/gui_hijack_train.jsonl \
  --weights weights/retrained-online --data-root data --index-dir data/index \
  --output outputs/features/train_retrained
python tools/train_harm_head.py --features outputs/features/train_retrained \
  --output weights/retrained-harm-head
```

Add `--resume` to resume a training run. To evaluate your trained model, use
`--weights weights/retrained-online` in feature extraction and
`--harm-head weights/retrained-harm-head` in evaluation, with new output paths.

## Citation

```bibtex
@misc{zhang2026lgwm,
  title        = {Right Screen, Wrong Transition: World Models as Verifiers for GUI Agents},
  author       = {Zhang, Jiaming and Wang, Xuan and Zhang, Fuyao and Cao, Yang and Lyu, Lingjuan and Lim, Wei Yang Bryan},
  year         = {2026},
  howpublished = {\url{https://github.com/jiamingzhang94/lgwm}}
}
```

## License

[Apache License 2.0](LICENSE)
