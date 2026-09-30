# EXP-0006 v2 seed 9101 数值子集复现

**结果：精确通过，范围限于 seed 9101 的三个模型及 delay 32/64。** 原始 `artifacts/EXP-0006/v2/manifest.json` 首次检查时已为 `COMPLETED`。本记录验证冻结实现的数值可复现性，不代表其他种子，也不改变实验科学判定或阶段门状态。

## 冻结环境与执行

- 从本地仓库建立干净 detached checkout `/private/tmp/neurotopo-exp6-repro-checkout`，HEAD 为 `3ed1f14030fe69a660fe618b6fbd426780e04499`，运行前工作树干净。uv 环境和缓存分别位于 `/private/tmp/neurotopo-exp6-repro-venv` 与 `/private/tmp/neurotopo-uv-cache`。
- 原始 manifest 状态为 `COMPLETED`，其 `git_commit` 与复现 checkout 完全相同。配置、EXP-0005 checkpoint manifest、EXP-0004 源 manifest 和 `uv.lock` 的 SHA-256 均与原始 manifest 一致；`source_sha256` 列出的 14 个源文件全部匹配。提交来源分别为实验实现 `3ed1f14` 和原始 checkpoint 训练记录 `eec9ca4`。

| 输入 | SHA-256 |
|---|---|
| `configs/exp0006_v2.json` | `df440af2483d28686c0aacd65a8cb1765906eea6c4173d786a650a632f1e7449` |
| `configs/exp0005_checkpoint_inputs.json` | `c00f200abd3321c3caa5689a692a3e8d998b355daefc7911ed38c049957a6995` |
| `artifacts/EXP-0004/v1/manifest.json` | `3477f2cf6f8b3b82f2ab815be80ed91efa996e5f7e4d418fefe1a8902b540b94` |
| `uv.lock` | `a4228ebafd79e9e2cafbacff4cae0cfd8ea1c5a768a8a95c6d5a03dea61b509b` |

执行冻结安装与测试：

```sh
UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-exp6-repro-venv UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache uv sync --frozen --extra dev
UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-exp6-repro-venv UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache uv run --frozen pytest -q
```

结果为 `57 passed in 14.91s`。之后仅执行注册 seed 9101，并把输出写至临时目录：

```sh
UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-exp6-repro-venv UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache uv run --frozen python -m neurotopo.probe --config configs/exp0006_v2.json --seed 9101 --output /private/tmp/neurotopo-exp6-seed9101-repro
```

## 比较结果

三个模型的 `result.json` 在排除 `runtime_seconds` 及各 delay 的 `timing_seconds` 后，其余字段逐项相同。比较覆盖完整 task/probe 分数、train/validation/test 指标、数据集哈希、pair-bit 数组与哈希、固定探针设置、资源字段和 checkpoint 信息。三模型的 `source_replay` 均为 `EXACT`；每个模型的原始 pair-bit 与复现 pair-bit 完全相同。训练、验证、测试数据哈希在每个 delay 下均跨模型一致。

下表列出未择取的 uncued probe 指标和同批任务的行为分数；其余指标字段也在上述完整 JSON 比较中完全相同。

| 模型 | delay | probe uncued train / validation / test | task joint-pair / pre / final |
|---|---:|---|---|
| `multi_gru` | 32 | 0.991780599 / 0.992838542 / 0.992187500 | 0.763183594 / 0.791992188 / 0.957519531 |
| `random_gru` | 32 | 0.984944661 / 0.986328125 / 0.986490885 | 0.676269531 / 0.741699219 / 0.959960938 |
| `gru_wide` | 32 | 0.998616536 / 0.999023438 / 0.998209635 | 0.879394531 / 0.886230469 / 0.995605469 |
| `multi_gru` | 64 | 0.985677083 / 0.988281250 / 0.984375000 | 0.470703125 / 0.692871094 / 0.784912109 |
| `random_gru` | 64 | 0.968668620 / 0.963867188 / 0.962239583 | 0.352539062 / 0.535156250 / 0.831054688 |
| `gru_wide` | 64 | 0.996419271 / 0.992838542 / 0.996093750 | 0.679687500 / 0.720214844 / 0.971191406 |

六个 NPZ 文件均逐字节相同；每个文件包含 `mean`、`scale`、`coefficients` 三个 float64 数组，形状与字节数分别为 `(64,)` / 512、`(64,)` / 512、`(65, 16)` / 8,320。因文件 SHA 相同，train-only scaling 与全部 ridge 系数也均逐字节一致。

| 模型 | delay | 原始与复现 NPZ 相同文件 SHA-256 |
|---|---:|---|
| `multi_gru` | 32 | `e64d20db974bb5c2c8501c049f068d5c794d289d1b9a0f230b283d1552b1b3cb` |
| `multi_gru` | 64 | `614e2fe5114ca4f8e73ccbac39ed2a231cb1a80175d4cbe4c07810bfe57cfbfa` |
| `random_gru` | 32 | `d22fe3fae3899331fb70f57719f05a8b70182363318ac5935d64a6798d7ad777` |
| `random_gru` | 64 | `638fdfdcef4f92d8cf6266ff187b639812a0b853d6b94febe8a1281c7d5290f4` |
| `gru_wide` | 32 | `c6972ad4a591c1633a564ca44a3384f6166eff85d4650e809f3a043ee380bc1e` |
| `gru_wide` | 64 | `adb278d2326b9a0a3139f4966220f8acf4e8e7a9ae3f70da75eedf6e6ad46bfb` |

本次范围只有三个模型、两个 delay 和一个已注册种子；未重跑其余种子或全样本统计。该复现用于实现与数值核验，不构成新增科学证据。
