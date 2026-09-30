# EXP-0011 固定8D动力学矩阵：设计制品，不是任务结果

2026-09-24。按v5及`docs/EXP-0011_v4_review.md`、独立代码复核，在**干净分离工作树** `/private/tmp/neurotopo-exp0011-freeze-20260924` 的提交 `c729d52b5938cdc53e804cf55ff8a1ea45eb7c6e` 上执行固定任务定义脚本；运行前 `git status --porcelain` 为空，矩阵与校验报告均不存在。该工作树使用 `/private/tmp/neurotopo-venv/bin/python`（NumPy2.3.3），固定源码/seed34017/QR规则无任务或模型结果参与。

| 顺序 | 实际命令 | 返回码 | 工具报告墙钟 | 核验 |
|---|---|---:|---:|---|
| 1 | `/private/tmp/neurotopo-venv/bin/python scripts/freeze_exp0011_matrices.py` | 0 | 0.167174167秒 | 创建`configs/exp0011_matrices.json`，原脚本写后读回数组SHA |
| 2 | `/private/tmp/neurotopo-venv/bin/python scripts/verify_exp0011_matrices.py` | 0 | 0.141366916秒 | 独立从seed/QR重算字节；报告`PASS`、形状/有限/正交/det/JSON/环境通过 |

矩阵 little-endian float64 C-order数组SHA-256：`7282b80d022f546d2993b6cee86ca9570b70bde0aad49510b8eac8354996b1d7`。完整矩阵JSON文件SHA-256：`e2f118379f3dde5a9ca32ce05fcf9865eb1167c50abe3812a10437d4ae3e36ec`；独立验证报告文件SHA-256：`f4d0be4e5097d57f0da6fd15f0420812564f75600c09c950f03825d5baefc5c9`。两文件从分离工作树以`open('xb')`排他复制到主项目相同相对路径、fsync，并再次逐字节SHA比较。`configs/exp0011_matrices.json`中的256个数值是后续唯一权威任务参数；生成器在原工作树记录NumPy/BLAS构建、线程环境、解释器、lock SHA、源码提交/脚本SHA及运行前Git状态。

以上是**任务定义冻结**，生成/验证用时单列，不并入后续模型实验3600秒执行预算；未读取任何由该矩阵生成的任务样本、oracle信息、控制或候选表现。数组不可因后续任务/模型结果而修改或重抽。v6正式协议/配置须引用以上SHA；独立预运行审查仍未批准研究训练。
