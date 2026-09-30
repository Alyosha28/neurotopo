# EXP0017 结果复核事前范围

2026-09-28，研究训练前保存。输入：冻结v2协议/config、manifest、所有24个checkpoint/predictions/results、固定代码与独立log-domain重放程序；先读协议再读结果。问题：学习/余量旗标、整体分支、总更新/时间、证据边界是否正确？最强拒绝：错误后验/数据配对或未学会控制造成虚假资格。接受：原始数组重算与检查点重放容差通过，记录完整，判词不跨Gate1；失败或无法核验如实保存，不重训/换seed。复核预算300秒，使用干净冻结checkout、PYTHONPATH=src、解释器/private/tmp/neurotopo-venv/bin/python。复核者写独立reproduction.json和review.json，主代理处理问题。
