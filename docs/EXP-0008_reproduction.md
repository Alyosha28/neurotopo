# EXP-0008 v1 seed 9101 clean-checkout reproduction

状态：**PASS（注册运行的单 seed 复现子集）**。本记录仅核对 seed 9101 的确定性与输入/资源配对，不替代十个训练 seed 的正式分析，也不改变 EXP-0008 的科学判定。

## 执行范围

- 正式运行 `artifacts/EXP-0008/v1/manifest.json` 已为 `COMPLETED`，包含 seed 9101–9110。
- 从提交 `7242520918f8e7aeb2b6d1e8cc8d847cfb637f95` 创建 detached clean clone：`/private/tmp/neurotopo-exp8-repro-checkout`；克隆初始 `git status` 为空。
- UV 环境和缓存均放在仓库外：`/private/tmp/neurotopo-exp8-repro-venv` 与 `/private/tmp/neurotopo-uv-cache`；没有创建仓库内 `.venv`。
- `uv sync --frozen --extra dev` 成功；冻结测试结果为 **81 passed**。
- 按规约只运行 seed 9101：

  ```sh
  uv run --frozen python -m neurotopo.final_probe \
    --config configs/exp0008_v1.json --seed 9101 \
    --output /private/tmp/neurotopo-exp8-seed9101-repro
  ```

## 对照结果

正式运行与复现子集的 `config_sha256`、`lock_sha256`、checkpoint-manifest SHA-256、EXP-0004 source-run manifest SHA-256，以及全部 **18 个源文件 SHA-256** 均相同。配置哈希为 `22da812d2956e06bf9f745930322ad4dd4a39dd99511461b0580aead7b08772f`；锁文件哈希为 `a4228ebafd79e9e2cafbacff4cae0cfd8ea1c5a768a8a95c6d5a03dea61b509b`；checkpoint 清单哈希为 `c00f200abd3321c3caa5689a692a3e8d998b355daefc7911ed38c049957a6995`；EXP-0004 manifest 哈希为 `3477f2cf6f8b3b82f2ab815be80ed91efa996e5f7e4d418fefe1a8902b540b94`。

| 模型 | delay32 A 探针 revise | delay32 原模型 revise | delay64 A 探针 revise | delay64 原模型 revise | 检查点重放 |
|---|---:|---:|---:|---:|---|
| `multi_gru` | 0.99072265625 | 0.99267578125 | 0.85107421875 | 0.8720703125 | EXACT |
| `random_gru` | 0.96875 | 0.9560546875 | 0.90380859375 | 0.85546875 | EXACT |
| `gru_wide` | 0.994140625 | 0.998046875 | 0.97998046875 | 0.98291015625 | EXACT |

三个模型的 `result.json` 全部非计时字段逐项相同。三模型各自两个 delay 的完整数据、pair、base-prefix 与 final-target 哈希一致；A/B 探针的训练、验证、测试分数和原模型表现逐项一致。六组资源账户完全相同，原始检查点 SHA、初始参数 SHA 和 mask SHA 均相同，且两侧 source replay 都为 `EXACT`。12 个 NPZ 文件的归档 SHA-256 均相同，内部 `mean`、`scale`、`coefficients` 数组的名称、形状、类型和逐元素数值也全部相同。

未发现 mismatch。唯一预期的 manifest 范围差异是正式运行标记为 `REGISTERED` 并含十个 seed，复现标记为 `REPRODUCTION_SUBSET` 并仅含 seed 9101；墙钟耗时字段不参与比较。该子集只验证 seed 9101 的 clean-checkout 可复现性，不能单独支持十 seed 的统计判定。
