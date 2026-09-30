# DIR-0003 交接说明（2026-09-24，goal 轮 11/12）

## 一句话状态
匹配容量识别性研究已完成六项前置计算、论文初稿（Markdown + 2 图）、复现包（56 条哈希）与**四轮独立审查**；**论文判词尚未通过最终审查口径，不得对外引用**；剩余两项必修未完成。

## 可核验的完成物
- 计算脚本：`scripts/dir0003_{v1_exact_check,spec_curve,null_contrast,tanh_control,tanh_matched,moment_diagnostic,review_followups,selection_null,figures}.py`
- 机器输出：`research/design/dir0003_*.json`（精确无穷视界容量、四格规格曲线、臂间实现零、tanh 两版对照、方差分解、选择零秩检验）
- 论文与投稿件：`paper/dir0003_matched_capacity.md`、`paper/figures/*.svg`、`paper/README.md`、`paper/dir0003_repro_manifest.json`（56 条，逐条字节哈希；`python3` 校验零失配）
- 决定：`research/decisions/0074`–`0079`
- 审查：`research/reflections/DIR-0003-{PREDESIGN,REDESIGN,PAPER,PAPER-FINAL}/`，四套 `reflection check` 均 `valid: true`

## 当前结论（必须附带其限制引用）
1. **臂间实现零**（同分配、同 γ、仅输入权重实现不同）是比"同臂 γ 敏感度"正确的比较对象；旧口径下的 `FALSIFIED` 判词**已撤回**。
2. 同分位点比较下，观测长延迟对比约为零的 **1.5–1.7 倍**（主格），落在重叠区；四格同类比较均为 `OVERLAP`。
3. 选择准则**有信息**（秩分位 `0.862/0.996`，Fisher `39.4/241.2`），不是"选择产物"；两种几何的差别主要是**可用候选数**（中位 `906` vs `33`）。
4. 隔离说明：零的 `n=84` 有效独立数 ≤28/γ 层；观测是族内**极大值**而零是单次抽取 → 统计量不匹配仍未闭合。

## 剩余必修（按优先级）
1. **统计处理**：对"选择后统计量"给出等价检验、置信区间与多重性控制（当前全文无统计推断）。
2. **口径闭合**：构造**选择匹配零**（对同一族做同规模极大、并以随机选择准则作对照），据此重写判词。
3. **判词体系统一**：删除全部 superseded 标签与旧图注；四格零与有效自由度写入申报账本。
4. **重新冻结送审**：以上完成后重新冻结 packet 并接受一次独立审查；**在此之前论文判词不得对外引用**（尤其不得引用"band geometry 决定可识别性"——该表述已被自身数据否掉）。

## 复现入口
见 `paper/README.md`（含 `UV_CACHE_DIR`/`UV_PROJECT_ENVIRONMENT` 设置与逐脚本命令）。所有脚本确定性、CPU、无训练、无外部依赖（图为手写 SVG）。

## 边界（未变）
Gate 1 仍 `NOT_PASSED`；无 EXP 注册、无训练；非回归清单 12 条无一重开；不主张能力/生物学/新颖性结论；用户授权的放宽只覆盖**目标轨道与主张范围**，不覆盖证据与判词。
