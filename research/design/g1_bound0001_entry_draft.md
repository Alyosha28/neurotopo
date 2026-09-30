# G1-BOUND-0001 entry 合同**草稿**（`research/entries/EXP-0015.json` 的拟稿）

2026-09-24。**这不是注册**：`research/entries/` 下的正式 entry 只在**独立审查通过后**创建。本文是把注册时所需的全部字段**事前**写好，使审查一通过即可落盘。**不改变 Gate 1 `NOT_PASSED`，不授权任何训练。**

## 1. 拟稿 JSON

```json
{
  "schema_version": 1,
  "experiment_id": "EXP-0015",
  "direction": "routing_alignment_bound_realisation",
  "track": "project_gate1",
  "gate1_claim": true,
  "matching_criterion_version": "G1-MATCH-v3",
  "question": "在表示容量几乎无差别的条件下，训练能否兑现对齐路由臂的表示界？",
  "factor": {
    "name": "G1-BOUND-0001 aligned vs cross-group input routing under near-zero capacity channel",
    "kind": "h3_candidate",
    "manipulation": "两臂共用同一块对角循环矩阵（块大小序列相同、块索引随臂置换），只有输入到分块的维度路由不同：S 为组内切分 (01)(2)(34)(5)，U 为跨组划分 (05)(23)(4)(1)。",
    "declared_consumed_axis": "none",
    "axis_declaration_note": "两臂在参数、状态、稀疏模式、度、跨块边数、样本、更新、内部步数与执行计算上恒等；注册须附 repeats>=101 的全轴比值表与收敛序列、以及元素级前向/反向计数。",
    "actual_primary_control": "U（跨组路由）臂 —— 与 S 臂共用矩阵、同初值分布、同数据、同优化器，唯一差异是路由。",
    "prediction": "主条件 c=4 的臂差 beta = KL_U - KL_S 的 95% CI 下界 > 冻结容量上界 C_ub = 0.0064；由 beta = delta + (e_U - e_S) 且 delta <= 0.0064，该结果蕴含跨组路由的训练超出量大于对齐路由。",
    "prior_art_boundary": "机制无新颖性主张（RIMs/PF-RNN/CTM 已覆盖分组递归与未知参数联合推断）；可审的只是条件化的路由对齐效应与其容量通道的关系。",
    "resource_matching": "见 axis_declaration_note；条件侧 c=4 与 c=5 的图与度序列不同，属任务侧差异，按 G1-MATCH-v3 披露。"
  },
  "conditions": {
    "primary": {
      "id": "c4",
      "edges": [[0,1],[1,2],[0,3],[0,4],[0,5],[1,3]],
      "degree_sequence": [1,1,1,2,3,4],
      "delta": 0.0020,
      "delta_ci95": [-0.0006, 0.0054],
      "gap16": {"mean": 0.4132, "ci95": [0.3143, 0.5267]}
    },
    "replication": {
      "id": "c5",
      "edges": [[0,1],[0,3],[0,4],[0,5],[1,3],[1,4]],
      "degree_sequence": [0,1,2,2,3,4],
      "delta": 0.0003,
      "delta_ci95": [-0.0013, 0.0021],
      "gap16": {"mean": 0.4428, "ci95": [0.3628, 0.5327]}
    },
    "frozen_capacity_upper_bound": 0.0064,
    "frozen_capacity_upper_bound_derivation": "|delta(c4) - delta(c5)| + 两条件 delta 区间半宽之和"
  },
  "estimand": {
    "statistic": "逐 seed 配对 beta_i = (KL_U - KL_S)，取均值；双侧 95% t 区间，df = n-1",
    "n": 8,
    "alpha": 0.05,
    "delta_min": 0.05,
    "delta_min_anchor": "项目既有 eps_abs，非本轮数据",
    "sd_source": "GPI-0001 pilot 逐 seed sd 0.0211（3 seeds），预先作 sqrt(2) 放大为 0.0298，按臂差使用",
    "mdd_convention": "MDD = (t_{0.975,n-1} + t_{0.80,n-1}) * sd / sqrt(n)；n=8 时为 0.0344",
    "escalation": "资格流实测逐 seed sd > 0.0441 时允许一次预注册升到 n=16；仍不满足则 INCONCLUSIVE；不得事后下调 delta_min",
    "no_covariate": "本设计不设协变量；难度差按 conditions 表披露而非调整"
  },
  "branches": {
    "success": "c=4 的 beta 95% CI 下界 > 0.0064，且 c=5 同向（点估计 > 0），且限定检查通过，且全部资格门与操作 guard 通过，且独立复核 seed 同向。",
    "partial": "beta CI 下界 > 0 但未超过 C_ub，或仅 c=4 成立而 c=5 不成立。",
    "unattributable_positive": "beta 点估计 > 0 但 CI 跨零 —— 不作正面判词，不得被引用为支持。",
    "failure": "beta 的 CI 上界 < 0。",
    "null_and_uninformative": "全部门通过且 beta 的 CI 含 0 且区间完全落在 [-MDD, +MDD] 内 —— 记录为该 n 下不可分辨，不判 FALSIFIED。",
    "borderline": "任一资格门失败、限定检查失败、tau_term 违反、MDD > delta_min 且 n=16 仍不足、或对称坍缩停止项触发，均为 INCONCLUSIVE。"
  },
  "qualification_gates": {
    "budget_doubling": "B 与 2B 下末端 KL 都须 > eps_abs",
    "nonlocality": "两条件的 gap(k_max) 的 95% 区间下界均 >= 0.20，k_max = 16（<= T/2 = 32）；gap(T) 恒为 0，不得作为门",
    "axis_ratio": "全轴比值落在 1.000 +/- 0.02；时延须 repeats >= 101",
    "block_exchangeability": "两臂块初始化逐块 i.i.d.、不依赖块索引",
    "confinement": "任一臂任一条件出现 KL_trained < 该臂该条件投影代价即 INCONCLUSIVE（S4 的兑现）",
    "capacity_ratchet": "C_ub 在任何资格样本前冻结；资格流复算若升高则门槛随之升高，不得降低；该规则与其在 delta_min 处的功效一并预注册"
  },
  "resource_cap": {
    "max_updates": 0,
    "max_wall_seconds": 0,
    "note": "注册时按资格流实测的 tau_step 与 tau_data 填写；本稿不预估，避免出现未经实测的预算数字。"
  },
  "task_search": {
    "max_configurations": 1,
    "retain_all_attempts": true,
    "heldout_rule": "条件对 (c=4, c=5) 事前固定，不得在看到训练结果后更换；正式 seeds 与复核 seeds 须在首样本前冻结并与资格 seeds 不相交。"
  },
  "forbidden_inferences": [
    "本设计检验了递归组织或拓扑",
    "模块化有效",
    "机制或方法新颖",
    "H3 语义分工、生物图效应、自然任务泛化",
    "把 c=4 与 c=5 的差异读作机制收益",
    "把本设计当作 G1-ALIGN-0001 或 GPI-0001 的追认或重开",
    "使用「超出容量」措辞"
  ]
}
```

## 2. 注册前必须补齐的三项（**本稿刻意留空**）

1. **`resource_cap`**：`max_updates` 与 `max_wall_seconds` 须由资格流实测的 `τ_step`（训练一步 p95）与 `τ_data`（batch 生成 p95）算出，**不得**预估。本稿填 `0` 以免出现未经实测的预算数字。
2. **正式与复核 seeds**：须在首样本前冻结，且与资格 seeds 不相交。
3. **`c=4` 与 `c=5` 的边集**：**已核对**（2026-09-24）与 `scripts/align0001_channel_sweep.py` 的 `graph_for_c(4)` / `graph_for_c(5)` 及 `align0001_channel_sweep.json` 的 `cells.*.edges` **三者逐字节一致**：

   | 条件 | 边集 |
   |---|---|
   | `c=4` | `[[0,1],[1,2],[0,3],[0,4],[0,5],[1,3]]` |
   | `c=5` | `[[0,1],[0,3],[0,4],[0,5],[1,3],[1,4]]` |

   **本稿初版把 `c=4` 写成 `[[0,1],[1,2],[3,4],[0,3],[0,4],[1,5]]`，为错误**，已按脚本更正——这正是本稿自设的核对步骤所要抓的错误。注册时须再记两条件边集的 SHA。

## 3. 注册的前置条件（**未满足**）

- **G1-BOUND-0001 的独立审查尚未完成**（前两次投递未返回；第三次收窄审查运行中）。按仓库规则，设计须先通过独立批评，**故本稿不得落盘为 `research/entries/EXP-0015.json`**。
- `gap(k)` 曲线覆盖到 `k_max = 16` 已实测（两条件区间下界 `0.3424` / `0.3609`）；`gap(T)` 恒为 0 的更正见 `g1_bound0001_gap_gate_correction.md`。
- 训练循环与评估路径的**第二实现核对尚待补**。
