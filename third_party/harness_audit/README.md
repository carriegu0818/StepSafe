# HarnessAudit task snapshot

A minimal subset of [HarnessAudit](https://github.com/UCSB-AI/HarnessAudit)
(base commit `6317162590aeeb1c8dde32b880ac199933343e4a`, MIT license, see [LICENSE](LICENSE)).

StepSafe replays trajectories recorded on HarnessAudit tasks and renders each task's access
policy into the monitor's observation. This directory holds only what that rendering needs: the task loader,
the schemas, the 210 task definitions referenced by the training data, and the per-domain tool
catalogs. Agent adapters, generated traces, workspaces, fixtures and multimodal prompt assets
are not included.

We thank the HarnessAudit authors for releasing their tasks and harness. If you use these tasks,
please cite their paper:

```bibtex
@article{harnessaudit,
  title={Auditing Agent Harness Safety},
  author={Liu, Chengzhi and Guo, Yichen and Liu, Yepeng and Yang, Yuzhe and Yan, Qianqi and Zhao, Xuandong and Hua, Wenyue and Liu, Sheng and Li, Sharon and Bu, Yuheng and Wang, Xin Eric},
  journal={arXiv preprint arXiv:2605.14271},
  year={2026}
}
```

Local changes to `multi_agent/loader.py`: task paths recorded in the training rows are mapped
to this snapshot, and missing multimodal prompt assets are tolerated (the task definitions are
used for static policy rendering only).
