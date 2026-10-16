<div align="center">

# StepSafe: Learning Timely Intervention with Context-Aware Safety Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2610.20960-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.20960)
[![Models](https://img.shields.io/badge/Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Caaaarr1e/stepsafe)

</div>

This repository trains **StepSafe**, a step-level reasoning monitor for LLM agents. Before each
proposed agent action executes, the monitor reads the task, writes a short reasoning trace, and returns a ten-field verdict:

```
\boxed{act:<CONTINUE|ASK|STOP>, risk:<0-100>, obj:<entity|NONE>, fail:<label>, fault:<label>,
       harm:<label>, src:<label>, timing:<label>, q:<question|NONE>, cite:<sequence|NONE>}
```

StepSafe is trained with **ASPO** (Adaptive Stage-Aligned Policy Optimization), which combines two
training signals:

- **SAPO** (Stage-Aligned Policy Optimization) credits each decision at its own step. The
  annotated signal *s*, boundary *b* and last preventable step *h* place every step in a stage,
  and the stage scores the sampled action. Each response is compared only with the other
  responses sampled at the same visible prefix. Safe and intervention steps receive balanced
  weight.
- **AFSD** (Adaptive Feedback Self-Distillation) credits the reasoning behind each decision.
  The monitor acts as its own teacher. On intervention steps the teacher sees the annotated
  verdict and violation; on safe steps it sees rules mined from the monitor's own false alarms
  (the *false-alarm rubric*).

This repository reproduces the main experiment of the paper, **StepSafe-4B** (Qwen3-4B). It builds
the training and evaluation data, trains ASPO from the released supervised warm start, and
evaluates the result.

The training framework is a trimmed fork of [verl](https://github.com/volcengine/verl) and
[verl-agent](https://github.com/langfengQ/verl-agent), as packaged by
[SDAR](https://github.com/ZJU-REAL/SDAR); see [Notice.txt](Notice.txt).

## Installation

Tested with Python 3.12, CUDA 12.8, PyTorch 2.8.0, vLLM 0.11.0, flash-attn 2.8.3, Ray 2.50 and
transformers 4.57.3, on NVIDIA B200 and RTX PRO 6000 GPUs.

```bash
conda create -n stepsafe python=3.12 -y
conda activate stepsafe
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.11.0
pip install flash-attn==2.8.3.post1 --no-build-isolation
pip install -r requirements.txt
```

The scripts set `PYTHONPATH` themselves; no package installation is needed. To run code by hand,
use `export PYTHONPATH=$PWD:$PWD/third_party/harness_audit`.

## Data

The trajectories come from other papers' datasets, so this repository ships only our labels and
builds the training and evaluation files from the official releases of the sources (CPU, about
a minute; every source is pinned and SHA-256 checked):

```bash
python scripts/build_data.py                # all data; --only eval skips the AgentDoG download
```

| Source | Official release | Used for |
|---|---|---|
| AgentDoG | [AI45Research/AgentDoG1.0-Training-Data](https://huggingface.co/datasets/AI45Research/AgentDoG1.0-Training-Data) | 1,189 RL replay tasks |
| ASSEBench | [Astarojth/AgentAuditor-ASSEBench](https://github.com/Astarojth/AgentAuditor-ASSEBench) | ASSE-Safety, ASSE-Security, R-Judge, AgentHarm |
| ATBench | [AI45Research/ATBench](https://huggingface.co/datasets/AI45Research/ATBench) | ATBench |

```
data/
├── rl/labels/                          # ours: annotations of the 3,184 replay tasks, AgentDoG text
│                                       #   replaced by placeholders
├── eval/labels/                        # ours: benchmark record ids and the irreversible step h
├── rl/train.parquet                    # built: 3,184 replay tasks (AgentDoG, HarnessAudit, red-teamed)
├── rl/trajectory_annotations.jsonl     # built: per-trajectory label, landmarks (s, b, h), step labels
├── rl/step_annotations.jsonl           # built: annotated ten-field verdict of each of the 22,000 steps
├── eval/validation/assebench.parquet   # built: validation during training: ASSE-Safety, ASSE-Security,
├── eval/validation/atbench.parquet     #   R-Judge, AgentHarm and ATBench, 100 episodes each
├── eval/table1.parquet                 # built: 4,033 evaluation trajectories (ASSE-Safety,
│                                       #   ASSE-Security, R-Judge, AgentHarm, ATBench)
└── eval/agentharm_step_labels.jsonl    # built: harm steps for the AgentHarm ASR
```

The script checks every output against the checksums in `data/manifest.json`; the RL files are
byte-identical to the paper's. [data/README.md](data/README.md) describes the labels, sources
and licenses, and [docs/DATA.md](docs/DATA.md) the field formats.

## Checkpoints

| Model | Description |
|---|---|
| [StepSafe-4B-SFT](https://huggingface.co/Caaaarr1e/StepSafe-4B-SFT) | the supervised warm start of Qwen3-4B; ASPO training starts from it |
| [StepSafe-4B](https://huggingface.co/Caaaarr1e/StepSafe-4B) | the ASPO-trained monitor of the paper (Qwen3-4B) |
| [StepSafe-8B](https://huggingface.co/Caaaarr1e/StepSafe-8B) | the same recipe on Qwen3-8B |

## Reproducing StepSafe-4B

**1. Data** (CPU, about a minute; see "Data"):

```bash
python scripts/build_data.py
python scripts/check_data.py
```

**2. Warm start:**

```bash
hf download Caaaarr1e/StepSafe-4B-SFT --local-dir checkpoints/StepSafe-4B-SFT
```

**3. ASPO training** (two GPUs; validation every 10 steps, the 3 best steps by validation J are
kept):

```bash
SFT_CKPT=checkpoints/StepSafe-4B-SFT bash scripts/train_stepsafe_4b.sh
# -> outputs/stepsafe_4b_s42/global_step_*
```

On Slurm, edit the `#SBATCH` lines and the environment block of the templates in
`scripts/slurm/`. A run longer than the partition's time limit can be split into chained jobs:

```bash
SFT_CKPT=checkpoints/StepSafe-4B-SFT bash scripts/slurm/submit_chain.sh scripts/slurm/train_stepsafe_4b.sbatch 8
```

**4. Evaluation** on the 4,033 Table 1 trajectories (two GPUs, about an hour), of a kept step of
the run or of a released model:

```bash
MODEL=checkpoints/StepSafe-4B-SFT CKPT=outputs/stepsafe_4b_s42/global_step_320 bash scripts/eval_stepsafe.sh
hf download Caaaarr1e/StepSafe-4B --local-dir checkpoints/StepSafe-4B
MODEL=checkpoints/StepSafe-4B bash scripts/eval_stepsafe.sh
```

`outputs/<run>/ckpt_val_metrics.json` lists the kept steps and their validation J. Training,
validation and evaluation all use the full visible history (12,000 characters); append
`env.validation.action_history_mode=evidence_retrieval` to evaluate with evidence retrieval over
earlier records (memory 4,096 / read 2,048 tokens) instead.

## Main settings

All values are in [`verl/trainer/config/stepsafe_trainer.yaml`](verl/trainer/config/stepsafe_trainer.yaml).

| Setting | Value |
|---|---|
| Initialization | supervised warm start of Qwen3-4B, thinking enabled |
| Replays | 16 tasks × G = 8 replays per step, temperature 1.0; full visible history (12,000 characters) |
| SAPO | stage score of Section 2.3; leave-one-out advantage at the same prefix, no std normalization; equal weight per (task, run, class) cell with Balance-GRPO class factors |
| AFSD | λ = 0.01, gate β = 5; rule mining every 2,048 teacher calls, ≥ 3 cases from ≥ 2 trajectories, at most 64 rules, UCB constant 0.5 |
| Optimization | AdamW, lr 1e-6 (linear warmup over 10% of a 2,985-step schedule), KL loss 0.01 (low-variance), entropy 0.001, clip 0.2, dual clip 3, token-mean |
| Lengths | prompt 16,384 tokens, response 2,048 tokens |
| Validation | every 10 steps on the 500 benchmark episodes of the 750-episode validation set; greedy decoding, the same full history as training; W&B logs `val/{j,asr,utility}` and the `paper/*` table cells; the 3 best steps by `val/j` are kept |
| Hardware | 2 GPUs (vLLM tensor parallel 2, FSDP) |

## Acknowledgements

We thank the authors of **AgentDoG** ([Liu et al., 2026](https://arxiv.org/abs/2601.18491)) and
**HarnessAudit** ([Liu et al., 2026](https://arxiv.org/abs/2605.14271)). StepSafe's training data
builds in part on their releases, and we generate our own taxonomy annotations and reasoning
(CoT) on them:

- 1,189 of the RL replay tasks replay AgentDoG trajectories. This repository never includes
  AgentDoG data; `scripts/build_data.py` downloads it from the official release.
- 707 replay tasks are trajectories we generated on HarnessAudit's multi-agent tasks, whose task
  definitions are included in [`third_party/harness_audit`](third_party/harness_audit).

We also thank the authors of ASSEBench, R-Judge, AgentHarm and ATBench, on which we evaluate,
and of verl, verl-agent and SDAR, on which the training code is built.

```bibtex
@article{agentdog,
  title   = {{AgentDoG}: A Diagnostic Guardrail Framework for {AI} Agent Safety and Security},
  author  = {Liu, Dongrui and Ren, Qihan and Qian, Chen and Shao, Shuai and Xie, Yuejin and Li, Yu and Yang, Zhonghao and Luo, Haoyu and Wang, Peng and Liu, Qingyu and Hu, Binxin and Tang, Ling and Mei, Jilin and Guo, Dadi and Yuan, Leitao and Yang, Junyao and Chen, Guanxu and Lin, Qihao and Yu, Yi and Zhang, Bo and Guo, Jiaxuan and Zhang, Jie and Shao, Wenqi and Deng, Huiqi and Xi, Zhiheng and Wang, Wenjie and Wang, Wenxuan and Shen, Wen and Chen, Zhikai and Xie, Haoyu and Tao, Jialing and Dai, Juntao and Ji, Jiaming and Ba, Zhongjie and Zhang, Linfeng and Liu, Yong and Zhang, Quanshi and Zhu, Lei and Wei, Zhihua and Xue, Hui and Lu, Chaochao and Shao, Jing and Hu, Xia},
  journal = {arXiv preprint arXiv:2601.18491},
  year    = {2026}
}
@article{harnessaudit,
  title   = {Auditing Agent Harness Safety},
  author  = {Liu, Chengzhi and Guo, Yichen and Liu, Yepeng and Yang, Yuzhe and Yan, Qianqi and Zhao, Xuandong and Hua, Wenyue and Liu, Sheng and Li, Sharon and Bu, Yuheng and Wang, Xin Eric},
  journal = {arXiv preprint arXiv:2605.14271},
  year    = {2026}
}
```

## Citation

```bibtex
@article{gu2026stepsafe,
  title   = {StepSafe: Learning Timely Intervention with Context-Aware Safety Reasoning},
  author  = {Gu, Siyi and Gao, Kunhan and Yang, Chao Peter and Ganguly, Sukanta and Ying, Rex and Cohan, Arman},
  journal = {arXiv preprint arXiv:2610.20960},
  year    = {2026}
}
```

## License

The code is released under the Apache License 2.0 ([LICENSE](LICENSE)).
`third_party/harness_audit` is a subset of HarnessAudit under its MIT license. Our labels in
`data/` are released under Apache 2.0; the source datasets keep their own licenses
([data/README.md](data/README.md)).
