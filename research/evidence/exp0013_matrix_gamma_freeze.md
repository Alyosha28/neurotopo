# EXP-0013 零样本矩阵/γ冻结与独立重推

在干净checkout提交`ddce115`只运行一次`freeze_exp0013_matrices.py`，未生成H/cue/u/y或训练模型。固定两个6维Q/A后得`D_base(48)=161.95313705068236`；按事前公式`γ=√(3/D_base)=0.13610245059775603`，在[0.1,1.5]内，状态`SELECTED`。这只保证给定完整u历史下**期望**单流KL为3，不证明真实修订、早期历史必要性、单流余量或RS4效果。

权威报告`configs/exp0013_matrices.json` SHA-256`8b5ed7d51b6178fa5469e9d0b54e25f502eb267017f6ba41db404f37d59f1b6c`，receipt SHA-256`97c985fd6e10304ef68501e26ad2919352fb77fe9d434e872c6dc61e7bcf4aec`；A数组`<f8`SHA`52218cad0f2b5aa35b7a516be864ba28c62121bd8f1ada68d859b6e5afdb60ed`，γ字节SHA`36b66da32c4c19f3bb92b3c5c5a0fc1749fd85dc3395ff5d9eef6ec77570db15`。独立`gpt-6-sol`以另写的NumPy2.3.3 QR/显式矩阵幂加权公式重推，37/37检查PASS，Q/A与字节SHA、D_base/γ及环境/来源/receipt全部吻合；验证`artifacts/validation/exp0013_matrix_gamma_verify.json` SHA-256`bbb18353bbd03a919c50a5d806b8714a97dbc4ccd97d9684f2566eee95639bf0`。冻结时的源码提交是当前Git祖先，不能要求验证时HEAD不变。

矩阵设计成本与模型实验分列。下一步仍只能写完整数值协议并独立预审；决定0050禁止任何H/cue/u/y样本直至任务、截断前史/捷径、真实SC64/D64、资源、统计和复核门全部前瞻固定。EXP-0012`NO_MATRIX`与其停止不变。
