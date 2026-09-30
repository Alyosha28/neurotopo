# 完成任务后的独立反思

每轮研究在 `DECIDED` 后、开启下一轮前执行一次：冻结任务设计与实际 action 输出 → 独立上下文找问题 → 主编排者逐条核验并采取行动。反思补充原有运行前 `CRITIQUE` 和结果科学审查；不能自行改写预注册规则、历史结果或阶段门。

独立审查的目标决定 action 范围：若审查执行和量化结果，默认只给登记设计与冻结的分析产物；若审查最终判定是否合理，判定文本本身也是需要检查的 action 输出，可以明确附入。不要为了减轻模型负担而事后挑选有利的结果字段，也不要把先前批评或主模型给反思者列的候选问题混入 packet。加入额外 action 产物时，在反思记录中说明为什么这些产物是回答该审查问题所必需的。
若任务本身是解释用户章程、论文或规则文本，事前 design 应包含足以核对关键推论的短原文及准确位置；只给文件路径会让只读 packet 的反思者无法检查遗漏的限定。补引文须标为任务后补证，不能声称原 packet 已提供。

## 输入边界

`python -m neurotopo.reflection packet --design <已登记协议> --action <已分析结果 JSON> --output <新 packet.json>` 生成只含 `design` 与 `action` 两类内容及其 SHA-256 的 JSON。`action` 的主内容必须是 `ANALYZED` 证据；若只给分析摘要不足以判断执行与任务完成情况，可用 `--action-output manifest=<manifest.json> --action-output verification=<verification.json> --action-output decision=<decision.md>` 加入同一类中的已完成 action 产物。账本将 design、主 action、verification、decision 分别与本轮 `REGISTERED`、`ANALYZED`、`REVIEWED`、`DECIDED` 证据哈希核对；manifest 的内容哈希单独留档。packet 不含对话、先前审查正文、主模型给出的候选问题或文件路径。额外产物仍应按具体审查问题保持最小，不要把整个仓库塞入上下文。

把 **packet 的内容** 发给 `fork_turns="none"` 的独立模型，要求它不浏览仓库、不读取其他上下文，只依据两段文本指出设计和执行中可定位的问题。反思者给出 `review.json`，每项包括 `id`、`source`（design/action/both）、`severity`、`evidence`、`problem`、`why_it_matters`、`falsifiable_check`。`evidence` 可以是一个逐字原文字符串，或由 1–4 个逐字原文字符串组成的数组；`source=both` 时必须分别锚定 design 与 action，不能把两段原文拼成一条伪引文。可以报告零问题；不能将其当作无问题的证明。代码校验 ID 唯一、证据短摘确实出现在相应输入、review 指向 packet 哈希。独立性来自实际新上下文和最小投递，不是 JSON 校验能够证明的性质；执行记录必须注明所用模型和上下文设置。

主编排者读取 review 后逐项 `accept`、`reject` 或 `defer`，核对仓库证据，并写 `reaction.json`。每项必须给出理由和具体行动；接受项还需指向已存在的行动证据文件。`decision_impact` 仅可为 `UNCHANGED` 或 `REASSESSMENT_REQUIRED`；后一种不覆盖旧判定，而是创建新协议或独立复核。反思者的判断不是科学证据，不能以“多数模型同意”推进阶段门。

`python -m neurotopo.loop reflect --packet ... --review ... --reaction ...` 验证三份文件、记录哈希与事件。新账本默认要求反思完成才能 `begin` 下一轮；旧账本完成首次反思后启用此门。相同反思可幂等重放，不同内容不能静默替换。若反思发现障碍来源于既有 design/action，先修正策略或登记可区分的新问题；不得用额外机制或改终点来救旧结果。

这一步不自动调用外部模型、重跑实验或联系他人。由当前编排者实际创建独立上下文并保存 review/reaction；失败、缺输入或校验不通过时保留记录并停止开启下一轮。

## 非实验轮次的研究任务

独立完成的设计审计、图可行性检查等实质性研究任务也使用同一 `packet → review → reaction` 流程。`python -m neurotopo.reflection check --packet ... --review ... --reaction ...` 可校验这些记录；它们不属于当前 `ResearchLoop` 迭代时，不调用 `loop reflect`，也不借此改变实验账本或阶段门。归档时写明该任务的设计、action 输出、独立上下文实际收到的输入、主模型如何处理每项问题，以及为何仍保留原研究判定。

此类任务也应**先保存简短设计，再开始审计**：至少包括问题、输入范围、最强反驳、接受/拒绝条件和禁止的推论。若漏做，只能用明示为“事后重建”的文件进行反思；不能把它称为预注册、用它追认已做取舍，或在反思校验通过后掩盖这个偏离。下一项任务须实际从事前设计开始。
对会受已存结果影响的只读再分析，最好在读取指标前单独提交设计；若只是同会话先写文件、在看结果后才提交，必须保存设计与分析的 SHA/执行顺序，并承认仓库快照本身不能独立证明先后。策略标签也应同时保留使其成立的各个事实轴，例如主对照余量和效应方向，避免一个标签遮住可用信息。
