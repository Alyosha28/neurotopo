# SCR-0002 packet 的一手来源命题图

供只读 packet 反思者定位论证；这是已完成 action 的来源摘要，不表示外部代码已运行。

| 支撑命题 | 固定一手位置 | 严格边界 |
|---|---|---|
| PF-RNN 为共享参数的 K 个带权 latent 粒子；转移含学习噪声、观测评分递推、软重采样，主输出作用于加权平均粒子。 | [AAAI 原论文 PDF 第2–3页公式(1)–(3)、第3页预测式](https://ojs.aaai.org/index.php/AAAI/article/download/5952/5808)。 | 作者结果不证明固定身份槽，也不匹配 NeuroTopo 的资源/任务。 |
| 论文 `NoResample` 表格只列 PF-LSTM；代码 PF-GRU 每步调用 `self.resampling`，`softplus(var)` 控制 proposal 标准差。 | [论文第6页 Table 3](https://ojs.aaai.org/index.php/AAAI/article/download/5952/5808)；[固定代码 `pfrnns.py` 第79–93、134–171行](https://github.com/Yusufma03/pfrnns/blob/0a76d52a857f06d4128bbc7601c616d714b25734/pfrnns.py#L79-L171)。 | `PF-GRU NoResample` 是后续须自建并核验的对照，不是该固定提交现成功能或论文数值结果。 |
| 固定代码用 `BatchNorm1d(K)`，把张量排成 `(batch,K,hidden)`，模型初始化粒子状态与均匀 log 权重并将输入复制 K 份；主预测先合并粒子，之后经输出层。 | [`pfrnns.py` 第41–43行](https://github.com/Yusufma03/pfrnns/blob/0a76d52a857f06d4128bbc7601c616d714b25734/pfrnns.py#L41-L43)、[第154–160行](https://github.com/Yusufma03/pfrnns/blob/0a76d52a857f06d4128bbc7601c616d714b25734/pfrnns.py#L154-L160)；[`model.py` 第75–85、119–141行](https://github.com/Yusufma03/pfrnns/blob/0a76d52a857f06d4128bbc7601c616d714b25734/model.py#L75-L141)；[PyTorch `BatchNorm1d` 官方 API](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)。 | “K 为通道，默认每索引有 affine/运行统计”是源码布局加 API 语义的**推论**；未测作者代码的置换行为，不能把它称固定语义身份。 |
| RIMs 通过注意力选择少量模块更新并稀疏通信。 | [RIMs 原论文](https://arxiv.org/abs/1909.10893)。 | 不同于本候选全槽每步更新/后验权重；只列邻近先例，不称等价。 |

数学恒等式和非线性读出反例由 `research/evidence/SCR-0002_noresample_slot_equivalence.md` 的明确假设推导，不来自作者的实验数值。
