# Git元数据存储修复（2026-09-28）

## 已确认的问题

原仓库位于NTFS/macFUSE挂载卷。对象目录出现I/O错误，部分loose对象无法解压，refs/objects中还出现macOS AppleDouble `._`附属文件。不能仅凭这些现象断定是硬件、驱动还是断开挂载导致；本次修复针对Git元数据，不声称修复NTFS文件系统。

## 恢复与存储位置

- 工作区仍为 `.`。
- 活动Git元数据为 `~/.local/share/neurotopo/Neurotopo.git`（本机APFS）。工作区`.git`是标准gitdir指针文件。
- 原损坏目录保留为工作区中的`.git.ntfs-corrupt-20260928`，仅在本地info/exclude中忽略，没有删除。
- 可读取原件备份与校验记录在 `~/.local/share/neurotopo/git-recovery/20260928`；无法读取的原路径逐项记录。
- 用EXP0022独立复核副本恢复完整主线，用EXP0020冻结副本补回原reflog旧提交及两个Codex临时快照；没有丢弃这些引用。
- 原config、hooks、index、refs和reflog均保留；core.worktree指向原工作区，未修改全局Git配置，没有对象alternates依赖。

## 验证

切换前候选库 `git fsck --full --strict` 通过；274个主线提交逐个与完整副本一致，切换前后worktree porcelain状态相同。原失败对象b0bb21fcbe01b7faeb92bf09f9c20868df21f1f0写入和读回一致。随后普通git add/commit成功，完成EXP0022归档（552e2dc）。旧失败记录保留，实验原始数据和源码未因修复而重写。

工作区和Git元数据现分开存放，完整仓库包含上述两处；标准Git命令从原工作区正常使用。原损坏备份用于追溯，不是活动对象库。
