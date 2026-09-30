# EXP-0013 v1 数值草案独立预运行审查

独立`gpt-6-sol`/high只读提交`3dc7801`的v1草案、已冻结矩阵/γ、现有ResearchLoop/worker及先例；没有生成H/cue/u/y样本、训练模型或修改文件。裁定 **`REVISE_BEFORE_IMPLEMENTATION`**。纸面full/末帧/截断前史oracle正确，根值无明显冲突，28,800＋72,000＋36,000更新算数正确；但v1不可执行为科学闸门。

决定性问题是现有`loop_worker`一次从VALIDATED跑到ANALYZED，没有控制资格独立审查后恢复RUNNING的入口。因此v1写的“独立裁CONTROL_STRONG_ENOUGH才训练RS4”无法由同一次worker实现。最小修订是把任务信息+SC/D控制资格和三臂正式效果拆成**两个完整ResearchLoop迭代**；后轮协议/根/预算/学习率选择函数须在首样本前冻结并与前轮哈希绑定，后轮VALIDATED核前轮独立REVIEWED/DECIDED及反思。旧资格失败正常封存，不能启动后轮。

其他硬缺口：统一LR用双控制总损失可牺牲其中一臂；D64分路输入弱于全输入稠密GRU，须按臂选最强控制LR或作同等能力论证；训练loss归约/权重、Adam超参/初始化/seed未冻结；时间预检缺全流程绝对上界，worker现无EXP0013 timeout绑定/失败封存。`δ=0.05`若称最低有用差，支持需**CI下界>δ**而非仅点估计≥δ/CI下界>0；PARTIALLY_SUPPORTED与操作失败优先级、边界等号、5seed复核只是稳定性守门均须锁定。早史置换改变状态、最终lesion影响输出不能合起来证明**早史对预测**有用；还需同一早史干预的最终读出/原oracle损失检验。CKA零分母、logit双中心化和H/cue频率门槛也须具体化。

强稠密单流若学到内部四槽不失格；即使以后正效，最多是这个适应性选择、四流结构对齐任务族的有限递归组织/连接分配效应，不是先例新颖性或一般推理。RIMs、PF-RNN、EviTrack先例定位与v1谨慎措辞基本相符。
