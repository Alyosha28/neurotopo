# CAL-0006 v3 独立复核

结论：在提交 `f170e9b3acea0ad7611746f5a61c9e1f86e980a4` 的干净 worktree 中，以冻结解释器重新生成固定评估流并加载六份 checkpoint 后，文件 SHA-256、参数 SHA-256、IID8/OOD12 最终 KL 均与正式报告相同。601 种子的完整 1200 步训练数据流摘要也相同。按预注册优先顺序，任务信息与 IID 学习护栏均通过，603/dense 的 OOD12 KL 为 `0.06496780739188451 < 0.08`，唯一主判词是 **CEILING**；配置不具备进一步设计审查资格。逐项机器可读结果见 `artifacts/calibration/CAL-0006_v3_reproduction.json`。

复核只读取正式报告、发布侧车和 checkpoint。正式产物未改；没有重新训练、改变阈值或选择新的评估条件。

| seed | 臂 | IID8 末次 KL (nat) | OOD12 末次 KL (nat) | checkpoint SHA-256 前 12 位 |
| --- | --- | ---: | ---: | --- |
| 601 | dense | 0.011143396236 | 0.082095809452 | `aaaf9941f4b3` |
| 601 | random | 0.020408969459 | 0.086208533462 | `5e6c0cfab248` |
| 602 | dense | 0.009120468254 | 0.086627010054 | `0cc6f53224c6` |
| 602 | random | 0.020569364771 | 0.115287453479 | `ec6b948b9235` |
| 603 | dense | 0.007544372593 | **0.064967807392** | `d0e6d122b745` |
| 603 | random | 0.018946648472 | 0.107510936220 | `f94e9cd4d09a` |

发布侧车给出的报告 SHA-256 与实际文件同为 `33b561b9b2e9eb22362f73f33f339d110fc7ab061a2280c1bf0fd19e9a05fda2`。配置 SHA-256 重算为 `598f6b849f51c893888d7cc089a92d92caffad50f83f54d0e00ec74e8f6bc12b`；脚本、协议和报告列出的所有源码文件哈希都与干净 checkout 相符。侧车记录 staging 写盘及 fsync 检查为 `12.673964250003337 s`，低于冻结的 `900 s` 上限；独占 hard link 发布元数据耗时 `0.0006444579994422384 s`，按协议排除在该预算外。报告累计六臂各 1200 次更新，共 7200 次臂更新、921600 个按臂计训练样本，与预算相符；这些训练计数来自正式报告，复核没有重演优化器更新。

评估 seed `6400001` 各生成 8192 例。IID8/OOD12 的前缀完全相同，摘要为 `53268a7973f2b8a890d9a1978d1ba104c71e1f5bf1000bd9a8840dc6a57311a7`；输入张量摘要分别为 `96eb136c468a6f67c94031bbbe37da75b899840e46bcf917b3d4d71423949d8b` 和 `1c3c87a5ad5f5fac14067b8f6df803f805076d5c28ad9f9a64d7fda968a94f54`。独立读取生成批次，IID8 oracle 真值 MAP 准确率为 `0.84912109375 ≥ 0.55`，OOD12 早末 MAP 修订率为 `0.4127197265625 ≥ 0.08`。601 种子的完整数据流按每步 `x`、`pre_joint`、`final_joint` 张量字节顺序累加，SHA-256 为 `c5465ff0834b35a67f8475b40601fe122de28c5184b4ddc84907f735587097c2`，与报告相同。

在该 checkout 中可用以下命令重算核心读数；路径中的项目根只作为输入。代码按冻结生成律生成数据，重新构建模型并严格加载 checkpoint，用 float64 求最终逐例 KL 均值：

```sh
cd /private/tmp/neurotopo-cal0006-clean-f170e9b
PYTHONPATH="$PWD/src" /private/tmp/neurotopo-venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from neurotopo.dynamic_filter_task import make_dynamic_filter_batch
from neurotopo.models import RecurrentModel
from neurotopo.run import configure

root = Path('.')
path = root / 'artifacts/calibration/CAL-0006_v3.json'
report = json.loads(path.read_text())
sidecar = json.loads(path.with_name('CAL-0006_v3_publication.json').read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
assert sha(path.read_bytes()) == sidecar['report_sha256']
assert sidecar['staging_fsync_checked_seconds'] <= 900
batches = {8: make_dynamic_filter_batch(8192, 8, 6400001),
           12: make_dynamic_filter_batch(8192, 12, 6400001)}
assert torch.equal(batches[8].x[:, :9], batches[12].x[:, :9])

def final_kl(model, batch):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(batch.x), 256):
            _, logits = model(batch.x[i:i+256], batch.pre_index)
            rows.append(F.softmax(logits, dim=1))
    p = torch.cat(rows).numpy().astype(np.float64)
    q = batch.final_joint.numpy().astype(np.float64)
    return float((q * (np.log(np.clip(q, 1e-12, 1)) -
                       np.log(np.clip(p, 1e-12, 1)))).sum(axis=1).mean())

for row in report['model_results']:
    seed = row['seed']
    for name, arm in row['arms'].items():
        checkpoint = Path(arm['checkpoint'])
        assert sha(checkpoint.read_bytes()) == arm['checkpoint_sha256']
        configure(seed, 1)
        model = RecurrentModel(4, 64, 4, groups=1 if name == 'dense' else 4,
                               topology=name, graph_seed=0 if name == 'dense' else 6500000 + seed)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        tensor_sha = sha(b''.join(p.detach().numpy().tobytes() for p in model.parameters()))
        assert tensor_sha == arm['final_parameters_sha256']
        values = [final_kl(model, batches[n]) for n in (8, 12)]
        assert all(abs(got - arm[key]['final_joint_kl_to_oracle']) < 1e-12
                   for got, key in zip(values, ('iid8', 'ood12')))
        print(seed, name, *values)

stream = hashlib.sha256()
for step in range(1, 1201):
    batch = make_dynamic_filter_batch(128, 8, 6000000 + 601 * 10000 + step)
    for tensor in (batch.x, batch.pre_joint, batch.final_joint):
        stream.update(tensor.numpy().tobytes())
assert stream.hexdigest() == report['model_results'][0]['train_stream_sha256']
print('seed-601 train stream', stream.hexdigest())
PY
```

这一复核证明冻结权重、当前数据生成和报告在上述检查点一致，不证明从头训练的每一步必然复现。KL 是固定合成任务的后验拟合指标；`CEILING` 是此任务配置的基准资格失败，不是 H3、任何多轨迹机制、生物效应或 Gate 1 的判决。稠密与随机臂的有效参数不同，不能据二者差距作匹配架构因果结论。
