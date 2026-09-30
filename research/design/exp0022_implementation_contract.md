# EXP0022实现接口（注册后的代码分工参考）

core模块equal_output_geometry.py提供：prefix_states(model,prefix)->H[N,3,2]；suffix_data(seed,bits,count,namespace)->commands[N,S,32],common[N,S,32,3,3],targets[N,S,33]；candidate_bank(model,h,seed,r)->states[N,16,3,2],sham[N,3,2],valid[N,16],sham_valid[N]及logit/radius/norm errors、原始方向和eigbasis；rollout(model,states[N,K,3,2],suffix)->logits[N,K,S,33],gram[N,K,S,33,3]；select_candidates(search_logits,search_targets,valid)->indices[N],scores[N,16]，不可接收test数据。
run模块先生成prefix/search，候选search选索引后才生成test；每cell将selected/random0/sham无效prefix按NONE回退并显式标记，完整64prefix进入描述，guard按每prefix allcandidate和sham valid比例统计。test四条件保存全部logit/Gram；search保存每候选score、index、状态和输入，无需保存完整search logits。条件顺序NONE,RANDOM,SELECTED,SHAM。指标独立分析读取原始数组，不能根据结果改变注册。
预算没有必要把同一seed的test随机数据复制45份，可每seed存1份共享文件；测试条件的选择只依赖search，已生成但未传入选择器的test也不等于泄漏，但按注册执行时应晚于index冻结生成/载入。

实现前定义核对：未来全时点准确率是future_accuracy，即所有prefix×suffix×t1..32的正确率平均，不是逐时刻最小值；当前准确率是t0。无效candidate在搜索前用对应NONE状态回退（geometry原candidate和误差仍保留），因此16个search候选已包含需要的回退结果，不另增加NONE search rollout。
