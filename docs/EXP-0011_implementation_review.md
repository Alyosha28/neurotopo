# EXP-0011 实现独立审查及复核

2026-09-24，独立 `gpt-6-sol` / high 只读初版`1a58511`的任务/五臂/资源/runner/analysis/worker及测试，隔离工程测试28项通过；未读取8050001、资格121xx、正式122xx或复核123xx研究根。初判REVISE：超时会卡RUNNING、分析只信摘要KL、预检来源/时间不可复核、profile只覆盖两臂且漏裁剪、解释诊断未产出、final流未在训练前封存。模型生成律、独立oracle、逐token方程和静态参数账未发现数学硬错。

主代理按审查修正：worker故障生成`worker_failure.json`和`analysis.json(INCONCLUSIVE)`；分析从保存NPZ独立重算KL/后验熵并重生登记测试流；预检receipt/原始计时可重分类；五臂训练/推理operator profile包含梯度裁剪；任务新增诊断用mode_means，评估报告总体与逐头Gaussian校准、熵/MAP/轨迹相似；各阶段test stream在模型训练前封存哈希。随后复核又指出`preflight`数值PASS但receipt超时时replay误拒、零更新重放未核训练输入/初始权重、校准未逐头；提交`f28498e`修正并增加专测。

独立同一审查者对`f28498e`的最终裁定 **`READY_FOR_ENGINEERING_PREFLIGHT`**：38项隔离工程测试通过，未见仍会错误放行任务/资格失败或绕过原始预测KL核验而产生包效果判词的硬错。该裁定**仅**允许在干净提交上运行工程验证/未训练实例资源profile；实际三段时限、全算子、控制资格与任何效果尚未验证。全量本地工程测试当前177项通过；没有研究根或正式模型运行。
