# EXP-0005 v2 clean-checkout 子集复现

**结论：PASS（仅限 seed 9101 的计算复现）。** 在冻结提交的独立 detached checkout 中，三个模型的 seed 9101 replay 与归档结果在所有非计时字段上完全一致；预测文件中的 120 个数组也逐项按键名、dtype、shape 和原始数组字节完全一致。此结果不替代十个 seed 的完整实验，也不构成新的科学结论。

## 范围与运行环境

- 原始结果 `artifacts/EXP-0005/v2/results/manifest.json` 的状态为 `COMPLETED`。复现使用冻结提交 `43e51aa08c203b88884a3f920ca3d82b5622fbd5` 的本地 detached clone：`/private/tmp/neurotopo-exp5-repro-checkout`；checkout 干净，未创建项目内 `.venv`。
- 只复现 seed `9101`，模型为 `multi_gru`、`random_gru`、`gru_wide`，输出位于 `/private/tmp/neurotopo-exp5-seed9101-repro`。其余九个 seed 未重跑。
- Python 3.13.15、macOS 26.7 arm64、CPU、PyTorch 2.8.0、NumPy 2.3.3。uv 环境与缓存分别在 `/private/tmp/neurotopo-exp5-repro-venv` 和 `/private/tmp/neurotopo-uv-cache`。

## 命令与验证

首次直接运行 `uv run --frozen pytest -q` 时，uv 按默认依赖集合未安装 pytest，命令因找不到 `pytest` 退出。按项目配置在外部虚拟环境同步 dev extra 后，冻结测试通过：

```sh
UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-exp5-repro-venv \
UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache \
uv sync --frozen --extra dev

UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-exp5-repro-venv \
UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache \
uv run --frozen pytest -q
# 47 passed in 15.84s

UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-exp5-repro-venv \
UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache \
uv run --frozen python -m neurotopo.replay \
  --config configs/exp0005_v2.json \
  --seed 9101 \
  --output /private/tmp/neurotopo-exp5-seed9101-repro
```

Replay 的 delay32 联合准确率如下；分数以 1024 对为分母：

| 模型 | full | blank | 与归档结果 |
|---|---:|---:|---|
| multi_gru | 0.7666015625 | 0.6044921875 | 完全一致 |
| random_gru | 0.6728515625 | 0.4775390625 | 完全一致 |
| gru_wide | 0.859375 | 0.6455078125 | 完全一致 |

## 逐项比较

每个模型的归档结果与复现结果比较时，只排除 `inference_timing` 和 `runtime_seconds`。其余所有 JSON 字段均完全相同，包括所有条件的指标及 `prediction_sha256`、输入与目标标签哈希、资源记录、与 full 条件的 pair-overlap 记录、原始流 replay 记录、checkpoint 哈希、初始参数哈希和 mask 哈希。未发现非计时字段差异。计时字段按任务要求排除；三个模型的 `runtime_seconds` 均有小幅变化。

三个 `predictions.npz` 各有 40 个数组。所有 120 个数组的键集合、dtype、shape 和 C-order 原始字节均一致，标签数组和 logits 数组都没有差异。这里比较的是 NPZ 内数组内容，不要求 ZIP 容器字节完全相同。

两份运行 manifest 的登记提交和输入摘要一致，并且重新对 checkout 文件计算的摘要与 manifest 相符：

| 项目 | SHA-256 |
|---|---|
| EXP-0005 config | `b484767d0e78f18cd6ae7a7e8116d81a616598684d9ba3dcb1c4da8dc137be26` |
| `uv.lock` | `a4228ebafd79e9e2cafbacff4cae0cfd8ea1c5a768a8a95c6d5a03dea61b509b` |
| checkpoint manifest | `c00f200abd3321c3caa5689a692a3e8d998b355daefc7911ed38c049957a6995` |
| EXP-0004 source-run manifest | `3477f2cf6f8b3b82f2ab815be80ed91efa996e5f7e4d418fefe1a8902b540b94` |

登记的 source commit `eec9ca4b8d664d7945da554f5237a36153e09bb2` 也相同。运行 manifest 的 `source_sha256` 和 `validation.source_sha256` 均逐项相同，包含 12 个源码文件哈希；validation 标为通过。

seed 9101 的三个 checkpoint 均同时满足：原始结果中的 `checkpoint_sha256` 等于复现结果字段、登记 checkpoint manifest 中的哈希，以及 checkpoint 文件实算 SHA-256：

| 模型 | checkpoint SHA-256 |
|---|---|
| multi_gru | `8a523d477e9b67e2c268a8a4eaa7e1e5ea66e303a2b1ba2eec8ceb21c87aec6f` |
| random_gru | `04e3a22cf709085a93f679d3b5a3e38ad1454a48e84955bd1695fa5defa0548d` |
| gru_wide | `5181d8bc81f7fbf21e932f13770f4c4f849c529f7011c034e12c3a350d242329` |

## 解释边界

此 clean-checkout 子集复现确认指定冻结环境下 seed 9101 的输入、checkpoint、标签、指标、资源/重叠记录和预测数组可重复。它没有重跑 seed 9102–9110，也没有重新计算完整十 seed 汇总或改变 EXP-0005 的登记判定；因此只作为工程复现证据，不增加独立样本量或机制证据。
