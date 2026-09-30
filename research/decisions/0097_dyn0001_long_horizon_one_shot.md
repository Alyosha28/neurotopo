# 0097：一次等logit重整的长期轨迹诊断

2026-09-28。DYN-0001 v3在冻结26模型、原EXP0025前256步与版本2预生成的26条延长后缀上完成52单元×5条件、视界256/512/1024的描述性审计。v1未运行；v2以512条合批前向计算，未能复演原实验每批128条的轨迹，第一单元未完成，原始失败与全部预生成后缀保留。v3只修复执行批次，未改变干预、终点或后缀内容。

**INCONCLUSIVE / MECHANISM_NOT_IDENTIFIED；研究轨为DESCRIPTIVE_ONLY。** 对新REL25002的HOLD，NONE/FULL8/EQUAL8在三个视界的末步准确率均50%；同一原256步输入在EXP0025每8步重整时，FULL8/EQUAL8为100%。EQUAL8在t8 512/512有效，256条序列到t36曾与NONE输出符号分叉，但后来回到相同50%终点。可直接判断的是t8单次重整不足以维持EXP0025观察到的修复；本审计没有区分规律重复刷新与某个较晚单次刷新。

REL25003的UPDATE NONE末步准确率随视界从96.09%→92.38%→82.23%，t8单次EQUAL8约95.70%→92.19%→82.03%。12个新REL在NONE两任务平均末步准确率84.78%→83.41%→81.09%；一次EQUAL8对应84.77%→83.40%→81.08%。12新INV在三个视界各条件均100%。原型替换使一些REL种子的准确率大幅下降，全部原始轨迹及分项保留。

t8 Gram局部切向在REL25002/HOLD为0/512可计算（近退化或条件数超冻结门槛），不能据此验证其局部雅可比解释；REL25003/HOLD可计算256/512，一步切向与实际差相关约0.901，但长期HOLD仍50%。一次局部变化不保证正确的长期计算。监督二状态码本仍是强替代解释，没有吸引盆证明、独立新训练seed效果或Gate1贡献。

318工程测试、259025根工程检查、0研究优化更新、0新原型拟合；v3 135249920个后缀转移、41396次float64切向一步转移，run164.96s+analysis1.72s≤1800s，原256步NONE复演最大logit误差0。APFS原件与项目副本逐文件SHA一致。完整报告`research/evidence/DYN-0001_equal_logit_future_observability.md`，核验`research/evidence/DYN-0001_verification.json`，独立packet反思`research/reflections/DYN-0001/`。

EXP0025原判词INCONCLUSIVE/R_NOT_SUPPORTED/M_NOT_SUPPORTED和项目Gate1 NOT_PASSED不变。本轮不根据已暴露轨迹补选干预时间、后缀或阈值。
