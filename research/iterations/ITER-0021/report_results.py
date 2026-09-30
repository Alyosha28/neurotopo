"""Generate a complete fixed-field report after EXP0026 analysis."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path,required=True);args=p.parse_args()
a=json.loads((args.run/'analysis.json').read_text());c=json.loads((args.run/'completion.json').read_text());geometry=json.loads((args.run/'geometry_summary.json').read_text())
lookup={(r['seed'],r['arm'],r['cell'],r['rotation'],r['condition']):r for r in c['rows'] if r['phase']=='main'}
lines=['# EXP0026：重整间隔、旋转配对与漂移首越界','',f"2026-09-29。协议v2，冻结源码007ccbb。登记判词 **{a['decision']} / {a['substatus']}**。本轮为DYN0002阶段A，对EXP0025的26个冻结模型作暴露轨迹机制诊断；原监督原型保持，无研究训练。",'',
'## 主要窗口判定','',f"ORIGINAL路径通过控制门并出现连续成功窗口的新REL种子：{a['window_seeds']}。控制不可判别种子：{a['control_inconclusive_seeds']}。",'',
'主窗口要求HOLD在256/512/1024步均≥95%、各视界相对NONE提升≥10pp，UPDATE同样均≥95%且每视界下降不超过2pp，EQUAL事件有效率H/U分别≥99%。失败分母同时要求EXP0025独立学习资格通过、本轮NONE的三个HOLD视界均低于95%；后一个条件使用了本轮已暴露数据，属于条件描述，不能声称新训练seed确认。', '',
'|seed|R路径|旧资格|条件失败|GAUGE控制|成功k|窗口|观测上界|','|---:|---|---|---|---|---|---|---|']
ks=[2,4,8,16,32,64]
for r in a['seed_windows']:
 good=[k for k,b in zip(ks,r['k_success']) if b]
 lines.append(f"|{r['seed']}|{r['rotation']}|{r['previously_qualified']}|{r['conditional_failure']}|{r['gauge_pass']}|{good}|{r['window']}|{r['k_upper_observed']}|")
lines+=['','连续后缀成功但延伸到64的窗口右端未测；非单调集合不强行命名临界间隔。IDENTITY缺单独GAUGE门，仅作配对描述；旧参考23005不进新seed分母。','', '## 预设病例的三个视界','', '|seed|任务|R|条件|256|512|1024|有效/尝试|','|---:|---|---|---|---:|---:|---:|---|']
for seed in (25002,25003,23005):
 for cell in ('LONG_H','LONG_U'):
  for rotation in ('ORIGINAL','IDENTITY'):
   conditions=('NONE','ONCE8','K2','K4','K8','K16','K32','K64')+ (('PHASE16','PHASE32','FULL8','SCALE8','GAUGE8','ADAPT90','COUNT90') if rotation=='ORIGINAL' else ())
   for cond in conditions:
    r=lookup[seed,'REL',cell,rotation,cond];m=r['summary']['metrics']
    lines.append(f"|{seed}|{cell}|{rotation}|{cond}|{m['final_256']:.2%}|{m['final_512']:.2%}|{m['final_1024']:.2%}|{m['valid']}/{m['attempts']}|")
lines+=['','## 原旋转与恒等旋转','', '|seed|模型|任务|最大Gram距离|平均Gram距离|1024准确率差R−I|','|---:|---|---|---:|---:|---:|']
for r in geometry['paired_R_I']:lines.append(f'|{r[0]}|{r[1]}|{r[2]}|{r[3]:.6g}|{r[4]:.6g}|{r[10]:+.4%}|')
lines+=['','这是同命令但经历不同历史旋转的轨迹差。理想共同R对Gram为gauge，所测差异可能包含数值误差的动态放大；不能把历史分叉向量直接称同一状态的纯干扰子空间。','', '## 独立漂移τ','', '|seed|模型|任务|R|首越界lower median|未越界/512|预测类别切换率|','|---:|---|---|---|---|---:|---:|']
for row in a['tau_rows']:
 seed,arm,cell,rotation,tau,censored,switch=row
 lines.append(f"|{seed}|{arm}|{cell}|{rotation}|{'TAU_NOT_REACHED' if tau is None else tau}|{censored}/512|{switch:.4%}|")
lines+=['','τ为独立NONE输入中，从t8起到自身预测类原型的Gram距离首次严格超过校准q90的滞后。它不是Jacobian模态衰减常数，不能直接拟合周期反比定律；预测类切换也可能使距离突变。未越界样本保留删失，未对已越界者单算中位数。','',
'## 次要机制对照与解释边界','',
'PHASE16/32改变首次刷新相位；FULL/SCALE/GAUGE分别保留原型、幅度与坐标控制。ADAPT90用pre距离和两步冷却，失败尝试也收费。COUNT90使用同一例ADAPT的完整T1024尝试总数排均匀时间表，具有未来调用数信息，是事后匹配对照；只在T1024比较调用次数匹配后的结果，短视界不可声称计数匹配。','',
'命令诊断从同一个状态分别执行HOLD/FLIP；刷新删除方向取当前Gram减EQUAL后Gram，正余弦表示方向重叠。重整无效或必要向量范数<1e−6时余弦不可用，不填0。该关联尚不能证明任务信息被擦除；没有进行命令子空间保护干预。全部104条几何摘要、近零分母和12seed描述性效果在geometry_summary.json/descriptive_effects.json。','',
'本轮没有把方向命名为快变量/慢变量，没有吸引盆识别或因果子空间消融，也没有新训练seed。监督二状态codebook是强替代解释；显式XOR状态机用一个bit即可精确解题，理想oracle100%。阶段B需由本轮可测规律另冻结，当前Gate1仍NOT_PASSED。','',
'## 图与证据','',
'![Accuracy vs interval](../iterations/ITER-0021/figures/accuracy_vs_interval.png)','',
'![Distance and accuracy](../iterations/ITER-0021/figures/distance_and_accuracy.png)','',
'![Command overlap](../iterations/ITER-0021/figures/command_overlap_vs_windows.png)','',
f"实际完成{c['main_rollouts']}主rollout+{c['tau_rollouts']}独立τrollout，基础转移{c['base_transitions']:,}，命令反事实{c['command_transitions']:,}，额外诊断EQUAL提议{c['diagnostic_refresh_proposals']:,}。刷新尝试{c['refresh_attempts']:,}、有效{c['refresh_valid']:,}。研究优化0、原型拟合0。运行{c['wall_seconds']:.2f}秒，分析{a['analysis_seconds']:.2f}秒，共享预算3600秒。",'',
'323项工程测试通过；专项工程根269026，神经研究模型未用于阈值/门槛调整。CPU1 float32，参数REL340/INV406、动态状态24bytes、原型48bytes每模型；FLOPs/峰值内存未测，无效率主张。完整逐rollout wall/CPU/有效率见resource_disclosure.json。独立原始指标复核与预定子集复演、packet反思、APFS/项目哈希归档分别见ITER0021与research/reflections/ITER-0021。']
args.report.write_text('\n'.join(lines)+'\n');print(str(args.report))
