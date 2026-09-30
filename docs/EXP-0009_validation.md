# EXP-0009 运行前验收

2026-09-24。冻结提交 `c32b07b071ace8c3c57aa2e5b01d9d238d374ff8`。真实项目工作树含一个无关未跟踪的 `artifacts/architecture-motion-20260924/`，未纳入源或实验；在 `/private/tmp/neurotopo-exp9-run-20260924` 从冻结提交创建的新克隆中，`git status --porcelain` 为空。新 venv `/private/tmp/neurotopo-exp9-venv` 由同一 `uv.lock` 的 `uv sync --frozen --extra dev` 安装，`uv run --frozen pytest -q` 为 **119 passed**。克隆中的源码指纹、配置 SHA、预检脚本 SHA 与原仓库 `artifacts/validation/exp0009_preflight.json` 完全相符。

无任务结果的 preflight 对十个计划种子均为 PASS：M3/C0 初始参数逐参数哈希相同，均为5,704有效/14,920分配参数、H64；各自掩码行列度16、自环64，C0 强连通。十个 C0 图相对原四块边界的跨块边数为717–747，未按任务结果筛图。每臂日程1,500步、192,000训练例、3,456,000输入 token、97,271,808,000 前向线性 FLOP 代理；各模型的资源账户和评估 d=1/4/8 数据摘要留在预检 JSON。完整训练 FLOPs 与进程峰值内存未测。实验结果在此验收时尚不存在。

注册协议哈希与账本 `REGISTERED` 证据一致；下一状态只能在预检证据入账为 `VALIDATED` 后，由受监督 `loop_worker` 启动。输出路径通过绝对路径 override 指向原仓库新目录，使执行进程仍在干净克隆中，且不会覆盖既有产物。此文件记录工程验收，不预言 EXP-0009 科学判定。
