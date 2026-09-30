# STR-0002 运行入口独立代码审查与修复

2026-09-24。独立 `gpt-6-sol` / high 多轮只读 `src/neurotopo/entry_policy.py`、`loop.py`、四个 dry-run 合同、测试与事前 `entry_policy_protocol.md`；未执行代码、改文件或启用 live。审查首先发现三类实质缺陷：独立 benchmark 被错误要求 `factor=null`、资源匹配只用一个自由文本字段、当前 workflow 示例在策略启用后缺 `--entry-contract`。主代理修为独立 benchmark 可选因素、状态/参数/执行计算三个必填计划，以及实际 enable→validate→begin 命令；结构通过仍非科学批准。

审查随后要求 QUESTION 记录保留同一已校验合同的内容寻址快照。首次硬链接方案存在 SIGKILL 双链接恢复窗口，改为同锁下 staging 写入/fsync 后原子 rename，已完整发布而事件未追加时可复用；已有 symlink 或多链接别名拒绝。单元测试实际发现一个静态审查漏掉的漏洞：对快照**完整路径**使用 `.resolve()` 会跟随末端 symlink，绕过检查。现只规范化父目录、保留文件名，新增 symlink 拒绝和完整快照重试测试。最终独立只读复核没有发现正常路径或此恢复路径的确定性阻断；对不遵守本地 loop 锁的外部恶意并发不作防护保证。

`artifacts/validation/str0002_entry_dryrun.json` 保存四例输入哈希、拒绝理由/接受轨道及事件变化；缺因素项目和项目参考 K 分别返回 `MISSING_FACTOR`、`PROJECT_ELIGIBILITY_UNRESOLVED`，没有追加事件；独立 benchmark 和领域参考 K 只进入 `QUESTION`。全量141项测试通过。

独立审查还核对了**当前 live ledger 的复制迁移演练**：`artifacts/validation/str0002_live_migration_rehearsal.json` 中原66事件 SHA 与原 live 相同；副本前66事件逐字节保留、第67事件仅启用策略，迭代7/DECIDED/反思及方向不变，副本无合同 begin 被拒且不追加；原 live 未动。审查认定这满足事前 live 启用验收。真正启用仍须主代理在代码提交后单独执行并复核事件。

后续已按此顺序执行：代码提交 `9b87dd1` 后，主代理在 live 账本追加 event-000067；`artifacts/validation/str0002_live_activation.json` 与独立只读核对确认前66事件逐字节保留、ITER-0007 仍 DECIDED/已反思，缺合同新 `begin` 被拒且不追加。该事件只启用结构入口，不批准任何科学主张或新训练。
