# NeuroTopo 本地 Skills 参考审计

审计日期：2026-09-23；本轮 GitHub REST API 元数据查询于当日完成。仓库元数据、提交及嵌套 `SKILL.md` 均做了核查；提交链接固定到检索到的 commit。Stars 是仓库级 `stargazers_count` 快照，只表示仓库关注度，不能当成某个嵌套 skill 的使用量、质量或科学有效性指标。下表是已发现候选中的审计记录，不是 GitHub 全站排名。

## 决策摘要

本地技能建议按项目记录形成五个边界清楚的工作流：`prior-art-audit`、`benchmark-design`、`experiment-runner`、`statistical-review`、`paper-writing`。外部材料只作流程参考；本地技能以 NeuroTopo 的 `AGENTS.md`、`research/stage_gates.md`、`docs/experiment_protocol.md`、实验注册和证据记录为权威。

论文写作技能必须包含用户指定的 `anti-defensive-writing` 审查，但它只在证据状态、实验判定和主张层级稳定后启用。只去除没有信息增量的预防性辩解、重复免责和过度犹豫；必须保留会改变结果解释、适用范围或可复现性的限制。不得隐藏负结果、偏离、未达门槛或替代解释，不得为了故事线换指标、重定义胜负条件或把局部合成任务结果扩写成机制、生物学或普遍能力主张。

## 本项目边界

参考审计发生于 Gate 0（构建与验证）阶段，当时活动实验为 `experiments/EXP-0001/v2.md`，研究问题限定为区块独立 GRU 与度匹配随机掩码 GRU 在延迟证据任务中的比较。注册记录已明确最强反驳：模型可能只是缓存候选值并在选择器出现后取值；两个监督输出只能证明可观察答案修订，不能证明内部信念修订；单一训练日程也不能比较最佳可达性能。

审计启动时先行工作文件尚未落盘；本次集成已补齐 `research/prior_art/` 的证据矩阵、新颖性边界和来源记录。简短提及 RIMs 不等于先例审查完成。Gate 0 未通过之前，不应让任何 skill 扩展到适应、通信、DARC、生物图或 LM 实验。

## 已核查候选

| GitHub 仓库 | Stars¹ | 最近 push（API） | SPDX | 固定 commit；实际检查的 skill 路径 |
|---|---:|---|---|---|
| [bytedance/deer-flow](https://github.com/bytedance/deer-flow) | 82,891 | 2026-09-23 | MIT | [`academic-paper-review/SKILL.md`](https://github.com/bytedance/deer-flow/blob/5d8a9492eea97e2f06f28d46df69c29355acf4da/skills/public/academic-paper-review/SKILL.md)、[`systematic-literature-review/SKILL.md`](https://github.com/bytedance/deer-flow/blob/5d8a9492eea97e2f06f28d46df69c29355acf4da/skills/public/systematic-literature-review/SKILL.md)；`5d8a9492eea97e2f06f28d46df69c29355acf4da` |
| [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) | 46,251 | 2026-09-21 | MIT | [`experimental-design/SKILL.md`](https://github.com/K-Dense-AI/scientific-agent-skills/blob/49c6e97775eaa18ba791bebe23162a70ae601c18/skills/experimental-design/SKILL.md)、[`statistical-analysis/SKILL.md`](https://github.com/K-Dense-AI/scientific-agent-skills/blob/49c6e97775eaa18ba791bebe23162a70ae601c18/skills/statistical-analysis/SKILL.md)、[`statistical-power/SKILL.md`](https://github.com/K-Dense-AI/scientific-agent-skills/blob/49c6e97775eaa18ba791bebe23162a70ae601c18/skills/statistical-power/SKILL.md)、[`scientific-writing/SKILL.md`](https://github.com/K-Dense-AI/scientific-agent-skills/blob/49c6e97775eaa18ba791bebe23162a70ae601c18/skills/scientific-writing/SKILL.md)、[`literature-review/SKILL.md`](https://github.com/K-Dense-AI/scientific-agent-skills/blob/49c6e97775eaa18ba791bebe23162a70ae601c18/skills/literature-review/SKILL.md)、[`peer-review/SKILL.md`](https://github.com/K-Dense-AI/scientific-agent-skills/blob/49c6e97775eaa18ba791bebe23162a70ae601c18/skills/peer-review/SKILL.md)；`49c6e97775eaa18ba791bebe23162a70ae601c18` |
| [msimchowitz/writing-skills](https://github.com/msimchowitz/writing-skills) | 39 | 2026-07-26 | MIT | [`literature-review/SKILL.md`](https://github.com/msimchowitz/writing-skills/blob/214981fe02326f27b0fc8790d00eb4b731607073/for-agents/literature-review/SKILL.md)、[`paper-writing/SKILL.md`](https://github.com/msimchowitz/writing-skills/blob/214981fe02326f27b0fc8790d00eb4b731607073/for-agents/paper-writing/SKILL.md)；`214981fe02326f27b0fc8790d00eb4b731607073` |
| [jurgendn/agent-skills](https://github.com/jurgendn/agent-skills) | 10 | 2026-09-17 | 未声明 | [`experiment-design/SKILL.md`](https://github.com/jurgendn/agent-skills/blob/f290312fb6c55a044402e8649afff5c112e660ab/skills/research-experimentation/experiment-design/SKILL.md)、[`benchmark-and-baseline-selector/SKILL.md`](https://github.com/jurgendn/agent-skills/blob/f290312fb6c55a044402e8649afff5c112e660ab/skills/research-experimentation/benchmark-and-baseline-selector/SKILL.md)、[`statistical-testing-guide/SKILL.md`](https://github.com/jurgendn/agent-skills/blob/f290312fb6c55a044402e8649afff5c112e660ab/skills/research-experimentation/statistical-testing-guide/SKILL.md)；`f290312fb6c55a044402e8649afff5c112e660ab` |
| [Kiterlin/anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing) | 636 | 2026-09-23 | MIT | 精确命名的 [`anti-defensive-writing/SKILL.md`](https://github.com/Kiterlin/anti-defensive-writing/blob/4b1938cdff9f7796e26065357f6ccbee5fd552ce/skill/anti-defensive-writing/SKILL.md)；`4b1938cdff9f7796e26065357f6ccbee5fd552ce` |
| [Adkid-Zephyr/anti-defensive-writing-Skill](https://github.com/Adkid-Zephyr/anti-defensive-writing-Skill) | 1,625 | 2026-09-12 | MIT | [`anti-defensive-writing-en/SKILL.md`](https://github.com/Adkid-Zephyr/anti-defensive-writing-Skill/blob/102c8b21acf5eda3a0aef3d9779a65db646c8980/skills/anti-defensive-writing-en/SKILL.md)、另有中文 skill；`102c8b21acf5eda3a0aef3d9779a65db646c8980` |
| [lensback940701/Evidence-Bound-Press-Conference-Revision-Skill](https://github.com/lensback940701/Evidence-Bound-Press-Conference-Revision-Skill) | 254 | 2026-08-23 | MIT | [`SKILL.md`](https://github.com/lensback940701/Evidence-Bound-Press-Conference-Revision-Skill/blob/33e6c3a2f69c8c27dfd328e7b8df97609c81687f/SKILL.md)；`33e6c3a2f69c8c27dfd328e7b8df97609c81687f` |
| [Worigin0314/academic-defensive-writing-auditor](https://github.com/Worigin0314/academic-defensive-writing-auditor) | 100 | 2026-08-30 | MIT | [`SKILL.md`](https://github.com/Worigin0314/academic-defensive-writing-auditor/blob/89f26b8a7f8149e6458f657010d354587d42cfe/SKILL.md)；`89f26b8a7f8149e6458f657010d354587d42cfe` |
| [Toadoum/ai-research-skill](https://github.com/Toadoum/ai-research-skill) | 12 | 2026-07-02 | MIT | 根目录 [`SKILL.md`](https://github.com/Toadoum/ai-research-skill/blob/bc98310bc2dc5c600f89f60099eed6cb950f6de1/SKILL.md)；`bc98310bc2dc5c600f89f60099eed6cb950f6de1` |

¹ Stars 为本轮 GitHub API 查询时的仓库计数。嵌入多项能力的大仓库不能据总 stars 推断某个具体 skill 的受欢迎程度。`jurgendn/agent-skills` 未发现声明许可证；仅用作功能发现，不复制其文字或资产。

## 适配建议

### `prior-art-audit`

借鉴 K-Dense 与 DeerFlow 文献综述中的先定问题、检索范围、截止日期、纳入/排除规则、两轮筛选、来源账本与主题综合流程；借鉴 `msimchowitz/writing-skills` 的来源优先级、claim-evidence matrix、版本/方法明细和“未披露”标注。NeuroTopo 版本还需强制检索同一机制的别名、早期工作、代码实现、实验对照与强零假设；区分论文报告、官方代码、二手综述和本项目推断。每个先例标注其对机制、表示、计算或能力链条的重合位置，并写出未解决反驳。无论搜索未发现还是材料不可得，都记录查询路径与覆盖边界；“没有搜到”不能写成“没有先例”。输出填充 `research/prior_art/evidence_matrix.md`、`novelty_boundary.md` 和 open questions，不自动判定创新成立。

### `benchmark-design`

借鉴 K-Dense 的随机化、阻断、正确重复单位和伪重复警戒。`jurgendn` 的 skill 发现了“最小可判别实验”、按主张类型选择指标、区分强基线/简单基线/最接近先例/自消融及预设阈值等需求，但该仓库没有明确许可证；不把它的文本、资源或结构作为底稿，本地规则由 NeuroTopo 协议和可用许可的来源独立编写。NeuroTopo 的基线匹配另须读取项目章程，逐项对齐或披露有效与分配参数、状态量、训练样本与更新、内部步数、实际执行计算、时延和内存。要区分工程可运行、合成任务效应、机制证据、生物效应和泛化。EXP-0001 已有冻结指标和决策阈值；skill 只检查缺口，不在结果出来后增加/替换指标或阈值。

### `experiment-runner`

把运行阶段限制为执行已注册方案：读取冻结配置和 gate；核对 source/config hash、依赖、种子、设备、数据版本与运行身份；在正式运行前完成协议要求的实现等价、mask/connectivity、数据泄漏、形状梯度、过拟合小样本和资源核算检查；只从批准的配置入口调用项目 runner；保留所有失败种子、日志、异常、输出清单和偏差。脚本/模型生成代码先人工审阅，skill 不可自行更改任务、调参到测试集、补造统计结果、覆盖注册阈值或推进未获门槛支持的机制。

### `statistical-review`

借鉴 K-Dense 的先定假设与分析、检查统计假设、报告效应量与区间、纳入全部计划结果的要求。NeuroTopo 必须先识别真正重复单位：EXP-0001 是训练/初始化配对 seed，不是每个测试对；复核成对差值、预注册的区间计算、全部 seeds 与失败记录、样本/seed 不确定性各自的含义。报告效应量、CI、离散度和预注册决策类别；不以单独 p 值代替实际阈值，不把未显著写成等效，不把结果不确定改写成证伪。固定集和 seed 数导致的边界应准确说明，不能套用面向独立样本或临床组别的默认检验。

### `paper-writing`（含 `anti-defensive-writing`）

借鉴 `msimchowitz` 的 claim-evidence map、由最强证据向外起草、直陈证据与解释的分界；借鉴 K-Dense scientific-writing 的方法/结果一致、保留负面与不确定结果；使用 Kiterlin `anti-defensive-writing` 的分类与“只删除不影响准确性的冗余辩解”；参考 Worigin 的逐句诊断字段和 Lensback 的 claim ceiling、evidence status、scope conditions、rival explanations。

触发顺序应是：先确认结果和协议偏差已经进入仓库，实验状态已判为 `SUPPORTED` / `PARTIALLY_SUPPORTED` / `INCONCLUSIVE` / `FALSIFIED`，逐项主张已有证据映射；再运行防御性表达审查。初始输出是“位置—原句—类型—是否必要限制—理由—保留/改写建议”，只有用户授权修改或主写作流程明确包含编辑时才改文。任何必要限制都留在准确的位置；仅压缩重复出现的免责。Adkid 版本的“只围绕优势”“不要主动示弱”“实验是论证工具、可删除不强化主线的结果”“换到更能反映优势的评价维度”等原则与本项目保留失败结果、禁止事后更改决策规则的章程冲突，不能原样吸收。

## 取舍

- **首选流程参考**：K-Dense 的 `experimental-design`、`statistical-analysis`、`scientific-writing`，以及 DeerFlow 的 `systematic-literature-review`、`academic-paper-review`。前两仓库 stars 高，但仅借用适用的流程原则；DeerFlow 的综述 skill 明确是 arXiv-only、最多 50 篇并依赖 subagent，不能把这些运行时限制变成本地规则。
- **高相关但不直接复用**：`jurgendn/agent-skills` 的 baseline selector、experiment-design、statistical-testing-guide 与 NeuroTopo 实验形态很近，但仓库没有 SPDX/license 声明。查阅供功能比对，不复制其表达、脚本或资源。
- **论文组织补充**：`msimchowitz/writing-skills` stars 较低但实际技能强调可追溯 source ledger、claim-evidence map、稳定后再写摘要，以及将直接证据和机制解释分开；项目可采用这些一般方法并按本地证据记录重写。
- **精确 anti-defensive-writing 候选**：本地 `~/.codex/skills` 与 `~/.agents/skills` 对文件名及正文做了窄搜，未找到该 skill；GitHub 上发现 Kiterlin 与 Adkid 两个同名能力。首选 Kiterlin 中“保留必要精度与方法限制”的写作诊断边界，并补入 Worigin 的必要 caveat 留存与 Lensback 的证据状态/主张上限约束。Adkid star 较高，但其若干规则会压低失败与不利结果的可见性，因此不采纳这些规则。
- **不把被评审的技能测试器当成科学 benchmark**：本轮还检视了 `Prism-Shadow/penguin-harness` 的 `benchmark-design/SKILL.md`。它是为校准 agent 技能测试集、隐藏评分 rubric 和固定 Pilot 规则而设计，且绑定其私有运行时；与 NeuroTopo 的模型实验、配对 seeds 和注册门槛不同，不作为本地 `benchmark-design` 的底稿。

没有安装全局技能，没有复制任何外部 `SKILL.md`、脚本或示例；本文仅记录候选来源、适用边界和重新编写方向。

## 已落地
完成上述检索与审读后，已在 `.codex/skills/` 创建五个原创项目适配技能，并以 skill-creator 的 `quick_validate.py` 全部验证通过。`paper-writing` 强制读取 `references/anti-defensive-writing.md`，保留实验事实和实质限制。
