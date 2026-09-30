# BND-0001 独立数学复核与干净输出重放

2026-09-24。独立 `gpt-6-sol` / high / `fork_turns=none` 只读 BND 事前协议、脚本、JSON、前期 witness、图生成代码和十份正式 C0 结果；未读对话、未编辑或训练。它先用独立 NumPy 计数和 `i<j` dyad 枚举复算十图，确认 formal/witness/BND 的原 mask SHA、正式结果文件 SHA、64×64、1,024边、64自环、逐点入出度16、强连通及4×4块矩阵逐项一致。原互反有向边226–258、跨块边717–747；仅总 cross 上界958–960，固定完整块矩阵上界912–946，目标原 R+96为322–354，故两类均0/10排除，`BOUND_NONDECISIVE` 正确。

独立推导采用另一表达式：`Rmax,total=960−[(C mod 2)+((960−C) mod 2)]`；`Rmax,block=960−Σ_{a<b}|E_ab−E_ba|−Σ_a[(E_aa−16) mod 2]`。它与主脚本的 `2floor`/`2min` 公式等价，正确扣除四组内部自环及跨组方向不平衡。这些都是上界，未约束逐点度、谱、motif、路径或实际可达交换序列；独立审查拒绝任何“上界未排除所以目标可达”的推论。脚本解析正式 JSON 并哈希原始字节，但判定只访问 seed/model/mask SHA，未取能力分数。

环境障碍保留：首次 `uv run --frozen` 因 home 下 uv cache 的 `.git` 权限错误退出，仓库默认 `.venv` 的 PyTorch 又是不完整 namespace package，因而审查者未能在该环境直接跑原脚本。主代理提供已验证的 `/private/tmp/neurotopo-venv/bin/python` 后，审查者将原脚本产物写到 `/private/tmp/bnd0001_independent_replay.json`，运行 exit0、标签及0/0排除计数相同，临时文件与仓库 JSON 的字节 SHA-256 都是 `c7825d9a17d1587e921d9a1cc6b05a95576373d97816903e7739dc672300ff04`。该重放修复了环境路径问题，不抹去首次失败，也不增强上界的科学解释范围。
