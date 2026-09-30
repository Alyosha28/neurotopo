# EXP-0004 v1 单种子独立复现

**复现结果：精确通过，仅限 seed 9101 的五个模型。** 相对于原始 registered run，全部指定的非计时结果字段与预测数组逐项相同，未发现不匹配。本记录检验实现与运行结果的可复现性，不改变 EXP-0004 的科学判定，也不代表其他九个种子。

## 固定上下文与执行

- 原始 `artifacts/EXP-0004/v1/manifest.json` 状态为 `COMPLETED`，记录了 registered seeds 9101–9110。原始运行与复现均使用提交 `eec9ca4b8d664d7945da554f5237a36153e09bb2`。
- 复现从 `/private/tmp/neurotopo-cycle2-repro-checkout` 的干净分离 HEAD 执行；运行前后 checkout 均干净。锁定环境位于 `/private/tmp/neurotopo-cycle2-repro-venv`，uv 缓存位于 `/private/tmp/neurotopo-uv-cache`。执行前确认 `neurotopo` 从该临时 checkout 导入。
- 环境：Python 3.13.15、PyTorch 2.8.0、NumPy 2.3.3、pytest 8.4.2；CPU、单线程。两份 manifest 记录的平台、Python、设备、线程数及这些包版本一致。
- 配置、基础配置和 `uv.lock` 的 SHA-256 分别为 `61e56aceb3c599825d4e83514b71ea2b8c1d7def033eff3c9bcc09c073449d92`、`fe8275c7b141e4d9cb4955276a027dc9f01e747d3660881c9e31cb7d9e629462`、`a4228ebafd79e9e2cafbacff4cae0cfd8ea1c5a768a8a95c6d5a03dea61b509b`。
- 两份 manifest 均记录 `validation.pass=true`。原始与复现 manifest 的 `validation.source_sha256` 完全相同，并与各自顶层 `source_sha256` 相同；覆盖 `src/neurotopo/` 下的十个 Python 源文件。配置、基础配置及锁文件摘要也与原始 manifest 一致。未在 clone 中重算或改写 validation suite 报告。

在临时 checkout 中运行冻结依赖和测试：

```sh
UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-cycle2-repro-venv uv sync --frozen --extra dev
UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-cycle2-repro-venv uv run --frozen pytest -q
```

结果：`31 passed in 14.69s`。随后执行注册的单种子命令，输出仅写到临时目录：

```sh
UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-cycle2-repro-venv uv run --frozen python -m neurotopo.study --config configs/exp0004_v1.json --seed 9101 --output /private/tmp/neurotopo-cycle2-seed9101
```

复现 manifest 记录 `status=COMPLETED`、`run_kind=REPRODUCTION_SUBSET`、`seeds_run=[9101]`。原始和复现输出分别来自 `artifacts/EXP-0004/v1/seed-9101/<model>/` 与 `/private/tmp/neurotopo-cycle2-seed9101/seed-9101/<model>/`。

## 逐项比较

每个模型均有 6000 条训练损失、5 个监测点和 16 个预测数组（noise payload，delay 8、16、32、64）。指标对象覆盖每个延迟的全部字段。比较了预测数组的键、dtype、shape 和原始数组字节；训练损失、完整监测记录及其中数据哈希；初始参数、mask 与评估数据哈希；所有资源与预算字段；以及去除计时字段后的完整 `result.json`。结果如下：

| 模型 | 指标 | logits/target 数组 | 完整损失与监测记录 | 初始/mask/数据哈希 | 资源及其他非计时字段 |
|---|---|---|---|---|---|
| `gru_small` | 完全相同 | 完全相同 | 完全相同 | 完全相同 | 完全相同 |
| `gru_wide` | 完全相同 | 完全相同 | 完全相同 | 完全相同 | 完全相同 |
| `sparse_rnn` | 完全相同 | 完全相同 | 完全相同 | 完全相同 | 完全相同 |
| `multi_gru` | 完全相同 | 完全相同 | 完全相同 | 完全相同 | 完全相同 |
| `random_gru` | 完全相同 | 完全相同 | 完全相同 | 完全相同 | 完全相同 |

五个模型均为精确匹配；总计 0 个不匹配。递归结果比较排除了 `inference_timing` 和 `runtime_seconds_including_monitor`；不比较 `final.pt` 序列化容器的字节或哈希。因此，结论只针对上述 seed、源提交、冻结配置和本次 CPU 单线程环境中的记录内容，不外推到其他种子、平台或科学效果。
