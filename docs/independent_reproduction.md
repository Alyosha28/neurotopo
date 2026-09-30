# EXP-0001 v2 独立复现记录

**复现结论：PASS（限定于 seed 0、五个模型）。** 从独立干净检出中的固定提交运行 seed 0 后，五个模型的预测结果和所有非计时、非 checkpoint 字段均与原始 seed 0 记录精确一致。该检查没有重跑 seeds 1–9，也不替代十种子正式结果或改变其科学判定。

## 固定上下文

- 提交：`60af9c628a7c1c74013b9afbdcf6253cfb8017d2`，分离 HEAD；运行前 clone 工作树干净。
- 临时检出：`/private/tmp/neurotopo-independent-review-60af9c6`。通过本地 `git clone --no-hardlinks` 复制 Git 对象后检出指定提交；没有把正式输出复制进 clone。
- 配置 SHA-256：`fe8275c7b141e4d9cb4955276a027dc9f01e747d3660881c9e31cb7d9e629462`；`uv.lock` SHA-256：`a4228ebafd79e9e2cafbacff4cae0cfd8ea1c5a768a8a95c6d5a03dea61b509b`。运行时六个 NeuroTopo 源文件的 SHA-256 均与原始 seed-0 manifest 一致。
- 环境：Python 3.13.15、PyTorch 2.8.0、NumPy 2.3.3、pytest 8.4.2、PyYAML 6.0.2；CPU、单线程。开发环境安装在 `/private/tmp/neurotopo-repro-venv`，uv 缓存在 `/private/tmp/neurotopo-uv-cache`。

先在独立 clone 安装冻结开发依赖并运行单元测试：

```sh
UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-repro-venv uv sync --frozen --extra dev
UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-repro-venv uv run --frozen pytest -q
```

测试结果：`12 passed in 14.48s`。随后按注册命令执行 seed 0，输出写入 `/private/tmp/neurotopo-independent-seed0`：

```sh
UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-repro-venv uv run --frozen python -m neurotopo.run --config configs/exp0001_v2.json --seed 0 --output /private/tmp/neurotopo-independent-seed0
```

输出 manifest 记录 `status=COMPLETED`、`run_kind=REPRODUCTION_SUBSET`、`seeds_run=[0]`，五个模型均完成。单元测试只在临时 clone 内运行；未运行会写出验证产物的 `scripts/validate.py`，也没有把复现输出写入原始 `artifacts/EXP-0001/v2`。

## 与原始 seed 0 的逐项比较

比较对象为原始 `artifacts/EXP-0001/v2/seed-0/<model>/result.json`、`predictions.npz` 与临时复现对应文件。五个模型均通过：

| 模型 | 全部预测指标与 `prediction_sha256` | logits/target 数组 | 完整训练损失轨迹及最终损失 | mask、数据与图结构 | 全部资源字段 |
|---|---|---|---|---|---|
| `gru_small` | 精确相同 | 精确相同 | 精确相同 | 精确相同 | 精确相同 |
| `gru_wide` | 精确相同 | 精确相同 | 精确相同 | 精确相同 | 精确相同 |
| `sparse_rnn` | 精确相同 | 精确相同 | 精确相同 | 精确相同 | 精确相同 |
| `multi_gru` | 精确相同 | 精确相同 | 精确相同 | 精确相同 | 精确相同 |
| `random_gru` | 精确相同 | 精确相同 | 精确相同 | 精确相同 | 精确相同 |

指标覆盖 delay 8 与 16 的全部记录字段。`predictions.npz` 中每个模型的 delay 8/16 `pre_logits`、`final_logits` 及对应 targets 均逐数组完全相等；记录中的 `prediction_sha256` 也完全相同。训练比较包括所有 400 个 loss 记录及 `final_train_loss`。资源比较包括两个延迟下的参数、状态、缓冲区、MAC/FLOP 估算和工作区字段；mask/data SHA-256 及图元数据也完全相同。

只从递归字段比较中排除 `checkpoint_sha256`、`construction_seconds`、`train_seconds` 和 `inference_timing`。因此，匹配结论不依赖 checkpoint 文件容器字节或机器计时值。

原始正式 manifest 记录完整 registered run 已完成，正式分析判定为 **INCONCLUSIVE**。本次结果只确认固定 seed 0 在指定环境和实现下可精确复现，不外推至另外九个种子、其他环境或模型的一般科学效果。
