# CAL-0003 三种子干净 checkout 复现

2026-09-24。状态：**PASS，范围为工程校准的三个模型种子与 d=1/4/8/16 全部条件**，并非正式 Gate 1 架构实验的独立科学复现。原校准产物 `artifacts/calibration/gate1_distractor_transfer_v1.json` 在脚本未跟踪时运行，但记录了脚本 SHA-256 `3191e66d484c58bbec8caa4178a6203d1318e26ebe01b46ccbc16cef57921aea`；冻结脚本随后提交于 `ab4a0da`，源码字节哈希相同。

第一次从 `ab4a0da` 创建干净 checkout 后，`uv run --frozen pytest -q` 在收集 `test_distractor_calibration.py` 时失败：测试以 `import scripts...` 依赖原工作区的导入路径。校准脚本未开始复现。这是测试包装问题，不是数值 mismatch。只修正测试以绝对脚本路径加载的方式，并在 `5422a672eaf4216604e59841ec3b91acda2d0934` 提交；校准脚本及生成器未改动。失败记录保留在本说明中。

第二次从 `5422a67` 创建新的独立克隆工作目录 `/private/tmp/neurotopo-cal3-repro-2-20260924`，`git status --short` 为空，独立 venv 为 `/private/tmp/neurotopo-cal3-repro-venv2`，由同一 `uv.lock` 执行 `uv sync --frozen --extra dev`。`uv run --frozen pytest -q` 得到 **99 passed**。原配置完整重跑：

```sh
uv run --frozen python scripts/calibrate_distractor_transfer.py \
  --output /private/tmp/neurotopo-cal3-repro-run-2/result.json
```

校验原报告与复现报告：`script_sha256`、全部 `source_sha256`、四个评估数据集哈希、逐种子初始/最终参数 SHA、监测损失与所有预测指标、资源账户、三个 checkpoint 的归档字节 SHA，以及 `eligible_ood_delays=[]` / `selected_smallest_eligible_delay=null` 均完全相等。仅复现提交、原运行的 dirty `git_status`、输出路径和耗时字段属于预期差异；这些字段未参与科学数值比较。复现 checkout 在运行后仍无 Git 改动。

这排除了本机同一锁定软件/硬件环境下脚本未留存或随机数不可重放所造成的结果疑点；它不证明跨硬件逐位一致，也不解释 d=4 崩溃是额外循环时间还是随机干扰负载所致。CAL-0003 的“不合格”设计结论和 Gate 1 状态均保持不变。
