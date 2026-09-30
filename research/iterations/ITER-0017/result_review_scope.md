# EXP0022v2独立结果审查范围

问题：等当前logit且等幅的状态干预，对未用于选择的未来后缀是否产生注册效应？输入为冻结52425d3的v2协议/配置/源码及artifacts/EXP-0022/v2全部产物。最强反驳：候选族很窄、OOD破坏、search过拟合、既有训练seed已暴露、共同旋转数值误差。完整保留RAW/INV/REL，不升级最小半径或方法优越性。
从新干净checkout以PYTHONPATH=src、/private/tmp/neurotopo-venv/bin/python单次运行equal_output_stress_reproduce，<=300秒，45cells全重做候选/searchselection/test；另独立核对primary统计及资源/数据隔离。不得重训、重试到成功或新增候选。共有forward/geometry实现这一边界明确披露。保存review.json/reproduction.json/reproduction_metrics.json/audit_execution.json，本轮原始数据不改。
