# EXP-0016 v1：ORBIT 固定任务的分阶段控制训练

2026-09-26。用户在要求减少流程、加快小实验后明确“开始”。本协议在首次控制训练/样本前固定，取代本轮“等待全部Gate1问题解决才可探索”的执行安排；**不取代**DIR0007完整8种子资格/128种子效果合同，不放宽Gate1。登记为independent_benchmark，目的为控制诊断与固定配置的去留，非ORBIT候选效果。

## 问题、反驳与范围

固定EXP0015已通过信息门的16状态环形HMM上，最简单的可学习过滤器是否已达到原δ=.02nat以内？若未达到，具名实际对照B以及无正则U、强稠密D能否学会且仍有提升空间？

候选仍是ORBIT-SOFT的对齐软先验A；实际主控制B是同H16/1904参数、同强度同谱的共轭软先验。U=H16无正则，D=H64全输入/全读出。**A不实例化、不训练、不读候选分数。** 先例为固定HMM forward、已知软等变/RPP；不声称新机制。最小充分状态是16类后验，不主张单流无法表示。

最强拒绝：简单HMM使用已知发射律，比GRU有额外算法先验；其胜出不是公平架构因果比较。接受这一点：H仅作为原DIR0007已经写明的SIMPLE_ALGORITHM_CEILING停止保护。实测结果可取消本固定配置的大额研究计划，故有明确否决价值；成功也只推进到正式控制资格设计。项目软先验资格、Generic、资源匹配/正式功效仍须在效果前解决，不由本独立诊断裁定。旧G1-RS/EXP0013不重开。

## 冻结任务、流和模型

生成器逐字复用EXP0015已审实现generate/posterior（NumPy2.3.3；16状态，T=.7I+.2S+.1Sᵀ，发射正确率.6，错误类别均匀）。训练长度32，末端完整历史后验为软标签；评估IID32和OOD64。输入只有one-hot观测，H也仅从相同观测和共同已知发射律过滤；不读取latent z或真实T作为可训练模型输入/初值。模型目标为末端posterior交叉熵；主指标KL(oracle||prediction)，不能用真实标签噪声制造余量。

种子16001/16002/16003。每seed/step训练root=`1600000000+seed*10000+step`（step1..3000），batch64；独立评估root=`1700000000+seed*100+length`，每长度4096例。各模型同seed同step共享可重建训练batch与评估流。均与EXP0015以及旧草案的流分离；本轮流从运行起标为已暴露，后续不得作独立效果流。无LR/预算/噪声菜单。

- H：16×16可训练transition logits，初值全0；按列softmax，uniform初始后验，发射矩阵固定正确率.6，逐步Bayes更新。只训练256参数，完整有向转移可学习，不预设局部环结构。float32、零之外严格正概率，不clip后验/损失。
- B/U：标准PyTorch2.8 GRU H16、input16、linear读出16；1904参数。D同理H64、16784参数。每模型先设Torch seed，初始化全部GRU矩阵Uniform[-1/sqrt(H),1/sqrt(H)]、全部bias0、读出同区间；明确按named_parameters的weight_ih、weight_hh、bias_ih、bias_hh、readout_weight、readout_bias顺序。B/U同seed相同初值；不允许构造时的默认初始化改变随后显式初始化的随机序列。
- B正则：λ=1，对全部6门矩阵及读出矩阵、6门bias及读出bias，加`sum||θ−P_G'(θ)||²/1904`。`p=[0,5,11,2,14,7,3,12,8,1,15,6,10,4,13,9]`，Q=I[p,:]，G'=QGQᵀ；投影为`Q P_G(QᵀWQ) Qᵀ`，P_G为同时循环行列的16项平均；bias投影为均值常向量。U/D无该正则。没有X条件或双向共轭假设。

所有臂固定Adam(lr=.001,betas=.9/.999,eps1e−8,weight_decay0)、clip_grad_norm1、3000更新，CPU float32单线程确定性，零初始GRU状态，无early stop。记录每500步训练loss和耗时但不选checkpoint；只用末权重。

## 分阶段预算与唯一决策

第一阶段只训练H的三个种子，9000更新，全部保留。每seed两长度评估与独立样本bootstrap10000次（PCG64 root=1800000000+seed；每次抽4096个索引，IID/OOD共用索引，仅为固定模型评估样本CI；np.quantile linear双侧95%）。

若三个H种子都满足IID KL均值<.10且OOD KL的95%上界<.02，执行**INCONCLUSIVE / SIMPLE_ALGORITHM_CEILING**，关闭这个固定ORBIT任务配置，不训练B/U/D。该结论仅限这三个训练实例与指定任务，不称架构假设被普遍证伪。比旧H均值<.02保护更保守，阈值δ不变；这是新诊断协议而非追改旧合同。

否则完整执行第二阶段B/U/D×3seed，每臂每seed3000更新；最多额外27000更新，共36000研究更新。逐seed记录IID和OOD所有指标。诊断窗口：B/U/D全部IID均值<.15且各臂三seed均值<.10；B/U/D每seedOOD∈[.04,1]且各臂均值∈[.08,.8]。这些沿用旧草案的学习/余量刻度，但**三seed、缺R/E不构成八seed完整资格**。

工程/非有限/超时优先→INCONCLUSIVE/ENGINEERING_ABORT或INCOMPLETE_BUDGET，保留部分结果，不换seed重启。H停止保护满足→上面的SIMPLE_ALGORITHM_CEILING。若进入阶段二：学习门失败→INCONCLUSIVE/LEARNING_FAILURE；学习通过但任一OOD下界窗口不足→INCONCLUSIVE/CONTROL_CEILING；其余OOD上限失败→INCONCLUSIVE/CONTROL_FLOOR；全部通过→SUPPORTED/CONTROL_PILOT_WINDOW，仅允许准备完整资格，不能签发CONTROL_STRONG_ENOUGH或运行A。

worker训练+评估+分析硬限1200秒。未完成任何seed不准当成科学失败；未完成研究优化步骤也应计入尝试次数。第二阶段的开始只由已冻H判据控制，不由人为看曲线决定。参数/动态状态/样本/更新/步数/实际wall与CPU时间、model/checkpoint/prediction/data/source/config hashes全部披露；本轮没有资源匹配效果判词，FLOP/峰值内存若未测则明确未测。

## 实现验收、原件与复核

一次前瞻审查覆盖设计与候选/实际控制身份；实现完成后执行必要测试和项目全套verify，冻结干净提交，从VALIDATED受监督运行。工程模型检查使用固定非研究root19016和手写观测：已知T的HMM与float64 oracle一致（float32≤2e−6）；参数数目、B投影对称/幂等与梯度有限；B/U相同初值；H/U/D各最多100次tiny-overfit更新，总工程更新≤300，loss末≤初×.8。仓库既有回归overfit另列，不计研究结果。

保存每个完整臂的末checkpoint、完整训练loss、评估观测/target/logits/逐例KL和bootstrap分布；保存全部流hash，不缓存科学训练批用于调参。失败臂保存末可用checkpoint和已尝试更新数。输出路径不得覆盖。
独立复核不增加训练：冻结代码检查来源hash，从末checkpoint重放全部评估观测、float64独立矩阵oracle及KL/CI/分支，数值容差posterior2e−6、评分/CI1e−6；预算300秒。同一检查点重放不是重新训练复制，也不是独立样本复制。研究最多36000更新，研究+复核墙钟≤1500秒。

结果独立审查与packet-only任务后反思可由同一份最小结果包安排在各自阶段，主代理保留集成责任；不能用审查次数代替验证。本轮以实际训练结果及明确分支结案，Gate1仍按其正式证据条件裁决。
