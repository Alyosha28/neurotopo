# 决定0080：入口规则放宽——项目 Gate 1 轨允许申报的 `architecture_organization` 因子（经用户授权）

2026-09-24。**状态：`EFFECTIVE`。生效依据：用户明确直接授权**——"授权该入口规则、generic 暂时先不管"。

**本决定是经用户授权的规则变更，不是经独立审查确认的规则变更**（与决定0069 同类）。它不注册实验、不授权训练、不改变 Gate 1 `NOT_PASSED`、不追认任何既有判词、不重开非回归清单任何条目。

## 1. 改了什么

- 策略版本：`gate1_candidate_first_v1` → **`gate1_candidate_first_v2`**（`src/neurotopo/entry_policy.py` 的 `POLICY`）。
- `factor.kind` 允许集合新增 **`architecture_organization`**（与 `h3_candidate`、`reference_k`、`independent` 并列）。
- **`project_gate1` 现在接受** `factor.kind ∈ {h3_candidate, architecture_organization}`；错误信息同步更新。
- **未改**：`reference_k` 在项目轨仍返回 `PROJECT_ELIGIBILITY_UNRESOLVED`（原样保留）；`field_reference` 仍只接受 `reference_k`；`gate1_claim` 仍强制等价于 `track == "project_gate1"`；结构校验（预算、任务搜寻上限、全尝试保留、分支三段、因子九字段非空文本）全部不变。
- 测试：新增 `test_project_gate1_admits_a_declared_architecture_organization_factor`（同时断言 `reference_k` 在项目轨仍被拒）；`tests/test_entry_policy.py` 9 项通过，**全套 219 项通过**。

## 2. 为什么需要它

DIR-0004（混沌潜轨迹跟踪）的机制是"显式扩张/收缩状态组织"，**不是 H3 形状**（H3 = 持续上下文 + K 个候选状态 + 证据驱动权重）。在 v1 下它**结构上不可能**进入项目轨，只能走 `independent_benchmark`——即使它按设计通过全部前置门也不可能成为 Gate 1 证据。本次放宽只解决这一处结构性阻塞。

## 3. `Generic` 的处理（按用户指示：暂时不管，但必须留痕）

决定0069 之后 Gate 1 文本里的 `Generic` **仍无可核验操作化**。本决定**不定义它**，代价必须写明：

- 任何在 v2 下进入项目轨的注册，**必须在其自身合同内自缚一个冻结的通用性谓词**（例如"≥2 个预注册任务族，或 1 个留出族"），且判词必须声明该谓词是**注册方自缚**而非判据方定义；
- 该缺口**留待前瞻裁定**；在裁定之前，项目轨判词的通用性成分**不可被第三方按现行文本核验**。

## 4. 这一放宽不产生候选，也不构成任何科学结论

- **不批准 DIR-0004**：它仍是设计稿，仍须先完成 ① §2/§4 解析量的独立脚本数值化、② packet-only 独立前瞻审查、③ 通过资格阶梯（oracle 拟合探针 → 余量门 → 学习门）。
- **不使任何旧候选复活**：非回归清单（候选级 9 条 + 任务/组合级 3 条）全部保持被否。
- **不降低证据与判词标准**：`G1-MATCH-v3` 的单轴声明与全轴披露、预注册、控制不得弱化、负结果保留、四类判词含义，全部不变。
- 本决定属"入口结构放行"，与决定0027 的性质相同：**结构放行不等于科学资格**。

## 5. 非回归与边界

Gate 1 仍 `NOT_PASSED`；无 EXP 注册；无训练授权；非回归条目无一重开；不主张新颖性、能力或生物学结论。若日后补做独立审查，须对**生效稿**重新冻结 packet 并另立记录，**不得**把事后审查追认为事前审查。
