# EXP-0007 v2：seed 9101 数值子集复现

**结果：精确通过。** 原始 v2 manifest 首次读取时已为 `COMPLETED`（注册的 10 个种子均完成）。本复现仅覆盖 seed 9101 的三个模型；不代表其他种子或改变科学判定。

在干净 detached clone `/private/tmp/neurotopo-exp7-repro-checkout`（HEAD `c86c3bd4eca99f4120b50fbc3a71548771893c9b`）中，使用独立 uv 环境 `/private/tmp/neurotopo-exp7-repro-venv` 和缓存 `/private/tmp/neurotopo-uv-cache` 执行 `uv sync --frozen --extra dev`；`uv run --frozen pytest -q` 结果为 **70 passed**。之后只运行：

```sh
uv run --frozen python -m neurotopo.transfer_probe --config configs/exp0007_v2.json --seed 9101 --output /private/tmp/neurotopo-exp7-seed9101-repro
```

运行时设置了上述 `UV_PROJECT_ENVIRONMENT` 与 `UV_CACHE_DIR`。配置、checkpoint 清单、EXP-0004 源 manifest、锁文件哈希及 manifest 所列 16 个源文件指纹均与原始结果一致：

| 输入 | SHA-256 |
|---|---|
| `configs/exp0007_v2.json` | `a1769f0658df1605e12a96d553927d13515f8e9f51b73ece779408ff805a7334` |
| `configs/exp0005_checkpoint_inputs.json` | `c00f200abd3321c3caa5689a692a3e8d998b355daefc7911ed38c049957a6995` |
| `artifacts/EXP-0004/v1/manifest.json` | `3477f2cf6f8b3b82f2ab815be80ed91efa996e5f7e4d418fefe1a8902b540b94` |
| `uv.lock` | `a4228ebafd79e9e2cafbacff4cae0cfd8ea1c5a768a8a95c6d5a03dea61b509b` |

三个模型的完整 `result.json` 除 `runtime_seconds`、`timing_seconds` 外逐字段相同，**无其他差异**；三者 `source_replay` 均为 `EXACT`。delay32 探针准确率如下，且与源 JSON 一致：

| 模型 | H8A→32 | H32A→32 |
|---|---:|---:|
| `multi_gru` | 0.788085938 | 0.991699219 |
| `random_gru` | 0.715820312 | 0.960937500 |
| `gru_wide` | 0.914062500 | 0.987304688 |

范围内六个 heads × 三模型的 **18 个 NPZ** 均检查了成员键、dtype、shape、数组原始字节、文件原始字节和 SHA-256，全部完全一致；每个文件含 `float64` 的 `coefficients (65,4)`、`mean (64,)`、`scale (64,)`。数据集哈希 99/99、normalizer/head scaler 哈希 21/21 匹配；delay32/64 的六个模型资源对象也逐字段相同。无任何数值或非 timing 字段 mismatch。仅运行 seed 9101，未扩展其他种子或重算全样本统计。
