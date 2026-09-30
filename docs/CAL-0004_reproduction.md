# CAL-0004 六臂干净 checkout 复现

2026-09-24。状态：**PASS，范围为三配对种子 × 固定/可变训练分布 × d=1/4/8/16 工程校准**；没有运行多轨迹架构。原始结果为 `artifacts/calibration/gate1_cadence_v1.json`，配对审计为 `artifacts/calibration/gate1_cadence_pair_audit.json`。冻结源提交 `2067906112b5a9e848f9b39a3162c2e124478cd7`，原执行时工作树为 dirty，但两份产物分别记录的脚本 SHA 与冻结文件完全一致。

从该提交创建新的独立克隆 `/private/tmp/neurotopo-cal4-repro-20260924`，初始与运行后的 `git status --short` 均为空。使用 `/private/tmp/neurotopo-cal4-repro-venv` 和同一 `uv.lock` 执行 `uv sync --frozen --extra dev`；`uv run --frozen pytest -q` 得到 **100 passed**。在克隆中将输出写到仓库外：

```sh
uv run --frozen python scripts/calibrate_cadence.py \
  --output /private/tmp/neurotopo-cal4-repro-run/result.json
uv run --frozen python scripts/audit_cadence_pairing.py \
  --result /private/tmp/neurotopo-cal4-repro-run/result.json \
  --output /private/tmp/neurotopo-cal4-repro-run/audit.json
```

原报告与复现报告在排除 Git 提交/dirty 状态、checkpoint 路径、训练和推理墙钟后**逐字段相等**：全部六臂的监测损失、d=1/4/8/16 预测指标、初始及最终参数 SHA、训练 token/FLOP 代理、四套评估数据哈希、预定护栏与资格结果均一致；六个 checkpoint 的序列化字节 SHA 也完全相等。`source_sha256` 和 `script_sha256` 完全一致。独立重算的配对审计每种子的五个 d=0/1/2 各100次块、每臂每块691,200 token、前向线性代理和有效输入/目标 SHA 均与原审计逐字段相等。

这证明在该 macOS/Python/PyTorch 锁定环境中，校准结果及配对账目可从干净源完整重放。它不测跨硬件逐位稳定性、完整训练 FLOPs 或峰值内存，也不能把 d=4 改善唯一归因于块内节律变化；可变臂还接触了 d=2。Gate 1 未因本复现而改变。
