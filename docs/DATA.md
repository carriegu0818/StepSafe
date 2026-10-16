# Data format

This page lists the fields the code reads, for users who want to train on their own
trajectories. The released files carry additional provenance fields that the code ignores.
Paths are relative to `data/`; the files are built by `scripts/build_data.py`
([data/README.md](../data/README.md)).

## Replay tasks: `rl/train.parquet`, `eval/validation/*.parquet`, `eval/table1.parquet`

One row per recorded trajectory, in the verl RL-dataset format:

| Column | Use |
|---|---|
| `data_source` | benchmark or corpus name; selects the source-specific intervention guidance in the observation (`agentdog*`, `assebench_*`, `atbench_*`, HarnessAudit otherwise) |
| `prompt` | `[system, user]` messages of the first step; the system message is the monitor protocol (`render_protocol()`). Used for dataset loading and length filtering; the step prompts are rebuilt by the environment |
| `env_kwargs` | everything the replay environment needs (below) |

`env_kwargs` fields:

| Field | Meaning |
|---|---|
| `audit_id` | trajectory id; links the annotations |
| `trajectory` | JSON list of records `{sequence_num, run_id, surface ("tool_call" or "communication"), agent_id, agent_role, tool_name, tool_args, tool_result, target_agent, target_role, content}` |
| `timepoints` | JSON `{unsafe, t_signal, t_boundary, t_harm, channel, risk_object, cf_safe, context_risk, t_context_risk, context_risk_object, require_channel}`; `t_signal`, `t_boundary`, `t_harm` are the landmarks s, b, h (1-based sequence numbers) of an unsafe trajectory |
| `step_labels` | optional JSON list of per-step labels; `prefix_binary_label` (SAFE/UNSAFE) overrides the annotated class d*_t of that step |
| `agent_action_only` | judge agent actions only; turns by the roles in `env.context_roles` are shown as history |
| `domain`, `goal`, `roles`, `role_prompts` | task context of the observation |
| `profile` | benchmark rows: the record's original agent profile, shown instead of domain/goal/roles; records are then rendered verbatim (`Turn N:`) |
| `task_path` | HarnessAudit rows: task definition under `third_party/harness_audit/multi_agent/tasks/`, whose access rules are rendered as the policy |
| `audit_mode` | `trajectory_binary_streaming` (or an `atbench_*` source) scores a trajectory-level label without landmarks |
| `v2_foresight_step_boxes` | the annotated verdict of each step embedded in the task; `scripts/check_data.py` checks it against `rl/step_annotations.jsonl` (training only) |

## AFSD annotations (training only)

`rl/step_annotations.jsonl`, one line per annotated step:

| Field | Meaning |
|---|---|
| `audit_id`, `sequence_num` | the step |
| `box` | the ten-field verdict `{act, risk, obj, fail, fault, harm, src, timing, q, cite}` |
| `response` | the reference response, ending with the boxed verdict |
| `status` | provenance of the annotation (one of the verified states in `afsd_context.VERIFIED_STATUSES`) |

`rl/trajectory_annotations.jsonl`, one line per trajectory: `audit_id`, `unsafe`, `t_signal`,
`trajectory`, `step_labels` (`episode_act_target`, `audit_reason` (the annotated reason to
continue), `violations[].description`), `context_risk`, `t_context_risk`,
`context_risk_object`, and `contrastive_pair_id` (the benign twin of a red-teamed pair).

## Evaluation labels

`eval/validation/assebench.parquet` (100 episodes each of ASSE-Safety, ASSE-Security, R-Judge
and AgentHarm) and `eval/validation/atbench.parquet` (100 ATBench episodes) form the validation
set used during training. `eval/table1.parquet` holds the 4,033 evaluation trajectories. Their
`timepoints` carry the benchmark's `unsafe` label and, for an unsafe ASSEBench episode, the
irreversible step h as `t_signal` = `t_boundary` = `t_harm`.

`eval/agentharm_step_labels.jsonl`: `audit_id`, `unsafe`, `t_boundary`, `t_harm` of each
AgentHarm trajectory, used for its ASR and timing cells.
