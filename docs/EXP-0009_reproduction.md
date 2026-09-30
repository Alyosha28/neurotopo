# EXP-0009 v1 独立复现记录：seed 9401

## 范围与结果

本记录只复跑预注册配对种子 9401 的 `multi_gru`、`random_gru`、`gru_wide` 三个模型。结果是一次干净 checkout 的单种子复现检查通过；它不代表其余九个种子已复现，也不构成 EXP-0009 的科学判定或十种子分布复现。

正式运行产物位于主工作树 `artifacts/EXP-0009/v1/`。其 `manifest.json` 状态为 `COMPLETED`，记录提交 `c32b07b071ace8c3c57aa2e5b01d9d238d374ff8`、空 Git 状态、9401–9410 十个正式种子。配置 SHA-256 为 `e07b5720a6281003314fa105c67bb872118d5ae006fa7905d69571cec04d2372`，`uv.lock` SHA-256 为 `a4228ebafd79e9e2cafbacff4cae0cfd8ea1c5a768a8a95c6d5a03dea61b509b`。manifest 登记的 23 个源码文件哈希均与正式运行源码和新 checkout 文件相符；`validation.pass` 与 `preflight.pass` 均为真，二者源码哈希均与 manifest 相符。干净 checkout 内的 validation suite SHA-256 为 `59e636bd092b4f8fe2dad71804d387af6e140425c5241c22ffb5834ef9a138e1`，preflight SHA-256 为 `a9218bda7c497c27c2e9f4325a8abd523344078526c575405ebd2dfc0bcc6ff6`。复跑 runner 接受了干净 Git 状态以及匹配的配置、suite、preflight 和源码。

## 环境与命令

独立 checkout：`/private/tmp/neurotopo-exp0009-seed9401-20260924`，HEAD 固定在上述提交，执行前后 Git 状态均干净。外部虚拟环境为 `/private/tmp/neurotopo-exp0009-seed9401-venv`，缓存为 `/private/tmp/neurotopo-exp0009-seed9401-cache`。环境与正式 manifest 完全一致：macOS 26.7 arm64、Python 3.13.15、PyTorch 2.8.0、NumPy 2.3.3、CPU、单线程；uv 为 0.12.9，依赖通过锁文件安装。

```sh
git clone --no-hardlinks . /private/tmp/neurotopo-exp0009-seed9401-20260924
git -C /private/tmp/neurotopo-exp0009-seed9401-20260924 checkout --detach c32b07b071ace8c3c57aa2e5b01d9d238d374ff8
UV_CACHE_DIR=/private/tmp/neurotopo-exp0009-seed9401-cache UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-exp0009-seed9401-venv uv sync --frozen --extra dev
UV_CACHE_DIR=/private/tmp/neurotopo-exp0009-seed9401-cache UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-exp0009-seed9401-venv uv run --frozen pytest -q
UV_CACHE_DIR=/private/tmp/neurotopo-exp0009-seed9401-cache UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-exp0009-seed9401-venv uv run --frozen python -m neurotopo.source_belief_study --config configs/exp0009_v1.json --seed 9401 --output /private/tmp/neurotopo-exp0009-seed9401-reproduction-20260924
```

锁定环境安装成功；冻结测试通过，`119 passed`。runner 正常结束，输出 manifest 标记 `REPRODUCTION_SUBSET`，且只含 seed 9401。复跑完整产物保存在 `/private/tmp/neurotopo-exp0009-seed9401-reproduction-20260924/`。

## 逐项比较

比较器对三个模型的 `result.json` 做递归比较，排除训练与推理计时字段，并单独报告 checkpoint 归档哈希；其余非计时字段均无差异。monitoring 中 300、900、1500 步的训练延迟、批次损失、验证指标及预测摘要均逐值相同。训练数据与调度摘要、各评估数据摘要、初始/最终参数哈希、掩码哈希、全部预测指标与预测摘要、图字段和全部资源账户均相同。

| 模型 | d=1 最终联合 KL | d=4 最终联合 KL | d=8 最终联合 KL |
|---|---:|---:|---:|
| `multi_gru` | 0.0032760444440646623 | 0.013913793457215051 | 0.06901331423742951 |
| `random_gru` | 0.0037014170171257993 | 0.013096614117070183 | 0.08718727118240566 |
| `gru_wide` | 0.003625931414979162 | 0.019267325824730994 | 0.12804247488018095 |

验证监控的最终联合 KL 也逐值相同：`multi_gru` 在 300/900/1500 步为 0.05074730446052943 / 0.006978730670860297 / 0.0030781887726925227；`random_gru` 为 0.050713012141676145 / 0.006890309858855259 / 0.003705815387210658；`gru_wide` 为 0.05995749887458724 / 0.007483269814470876 / 0.003594127674696939。每个监控点的其余验证指标、批次损失、训练延迟和预测摘要均已纳入逐值比较。

| 模型 | 图与原四块间边数 | 有效/分配参数 | 有效循环参数 | d=4 每样本 dense/active MAC |
|---|---|---:|---:|---:|
| `multi_gru` | block，度数 16，自环 64，跨块边 0 | 5,704 / 14,920 | 3,072 | 589,696 / 202,624 |
| `random_gru` | random，度数 16，自环 64，强连通，跨块边 731 | 5,704 / 14,920 | 3,072 | 589,696 / 202,624 |
| `gru_wide` | dense，度数 64，自环 64，跨块边 3,072 | 14,920 / 14,920 | 12,288 | 589,696 / 589,696 |

其余资源字段也完全一致：三个模型均为 64 个隐状态浮点数、d=4 序列长 42，dense 前向代理为每批 150,962,176 FLOPs。完整图描述、各延迟资源、所有结果指标和哈希保存在正式及复跑的 `seed-9401/<model>/result.json` 中。

预测文件逐数组载入比较：每个模型的 NPZ 含 18 个数组，键、形状、dtype 与数值全部相同。模型 checkpoint 用 `torch.load(..., map_location="cpu", weights_only=True)` 载入后，逐键比较 7 个张量，dtype、形状和值均完全一致。checkpoint 张量内容哈希、checkpoint 文件 SHA-256 与预测 NPZ 文件 SHA-256 分开核对；本次三模型的张量内容均一致，两个归档文件的字节 SHA-256 也都一致。每项的正式/复跑 SHA-256 与数组摘要保存在 `/private/tmp/neurotopo-exp0009-seed9401-reproduction-20260924/comparison-seed9401.json`。

训练与推理秒数、manifest 时间戳及总 wall time按复现规则排除，不作为差异判据。

## 限制与记录差异

这是一个配对种子、三个模型的一次确定性重跑，只验证该种子的实现与归档一致性；不能据此估计跨种子不确定性或替代十种子分析。`research/stage_gates.md` 仍写着 EXP-0009 尚无训练结果，与正式 manifest 的 `COMPLETED` 状态不一致；本记录保留该文档差异，不据此修改实验协议、研究判定或阶段门状态。
