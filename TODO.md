# todo

1. 新建一个分支用于上传匿名仓库，供审稿用。
2. 测试ASR波动。将论文中的Vicuna-7B-gcq, llama2-GCG, llama2-beast, llama3-beast, llama3-advprompter, vicuna-beast使用不同的随机种子重跑多次实验来验证PSKV相比于原始版本对攻击成功率没有影响，并通过理论分析来说明其不会有影响。
3. 通过理论分析说明Faster-GCG与我们的加速方法是正交的。同时解释由于FasterGCG没有公布代码，并且由于时间限制，因此我们无法提供关于FasterGCG的PSKV集成实现。
4. 说明即插即用的使用方法是仅需将攻击方法中模型的推理替换为使用kv cache版本的即可。以GCG为例，根据当前的代码解释最少只需要增加多少行就可以将普通的方法转变为使用PSKV的方法，同时说明我们的库的使用方法。
5. 以GCG为例，添加使用cuda或python的内存分析工具的分析代码来说明每层在每个时刻的内存使用情况。 
6. 说明vllm和sglang无法在无梯度的情况下使用。同时说明技术不可行性：给 serving engine 加梯度不是简单的事，需要重写 CUDA kernel 的 backward pass，这本身就是一个巨大的工程挑战。
即使加了梯度，PSKV 仍有独特优势：layer-wise lazy expansion 是 workload-specific 的优化，通用系统做不到。还有serving engine 的调度器、LRU 驱逐、radix tree 查找在同步攻击场景下都是纯开销。
7. 补充关于更长后缀和更长提示词的测试结果。你可以使用wild jailbreak数据集。
9.补充关于社会关切的使用说明，说明开源的情况：
Our work accelerates existing attacks, not creates new ones. PSKV is a pure inference optimization that does not introduce any new attack algorithm, novel vulnerability, or previously unknown attack vector. All six attacks evaluated in our paper are already publicly available with open-source implementations. PSKV reduces the computational cost of running these existing methods but does not lower the technical barrier to entry or expand the attack surface.
Responsible code release plan. We plan to adopt a gated release strategy:

The code repository will be released under a research-only license (e.g., the Llama community license or a custom non-commercial license) that explicitly prohibits use for malicious purposes.
We will require users to acknowledge an acceptable use policy before accessing the code, following the precedent set by HarmBench (Mazeika et al., 2024) and GCG (Zou et al., 2023).
The release will include only the acceleration framework (PSKV) as a modular library, without bundling pre-configured end-to-end attack pipelines or curated harmful prompt datasets.
We will include a prominent ethical use statement in the repository README and documentation.

Broader benefit to the safety community. We argue that the net societal impact of this work is positive. The primary bottleneck preventing comprehensive LLM safety evaluation is computational cost—safety teams at organizations often cannot afford to run thorough red-teaming evaluations at scale. By reducing this cost by 40-50%, PSKV directly enables more thorough safety audits, which benefits defenders more than attackers. This argument is consistent with the established norms in the adversarial ML community, where tools like HarmBench, GCG, and AutoDAN have been released openly to advance safety research.
We will add a dedicated "Ethical Considerations" section to the revised manuscript discussing these points.
10. 讨论整合梯度检查点与微量批处理技术并重新实现优化较大的方法.下面是一个claude-Opus输出的参考意见，并不代表我的观点，可以用作参考： We have conducted a systematic analysis of memory bottlenecks across all six attacks and found that the optimal mitigation strategy is attack-dependent, reflecting the diverse computational patterns in our benchmark.
For optimization-based attacks (GCG, GCQ, AutoDAN, BEAST), the gradient computation phase uses small batch sizes, making activation memory modest. The dominant bottleneck is the logits tensor during the candidate scoring phase, which scales as O(N_cand × L_target × |V|) and does not involve backpropagation. Therefore, gradient checkpointing provides limited benefit for these methods. Instead, we find that micro-batching of the scoring phase and chunked loss computation are more effective. For BEAST specifically, which is entirely gradient-free, micro-batching is the only applicable technique and enables larger beam widths within the same memory budget.
For gradient-free attacks, gradient checkpointing is inapplicable as no backward pass is performed. Micro-batching of the candidate evaluation phase is already implemented in our framework via the width_bs parameter, which controls the chunk size for scoring.  For gradient-free methods, micro-batching is functionally equivalent to sequential processing of smaller batches.

For model-based attacks (AmpleGCG, AdvPrompter), the situation is reversed. These methods involve standard LLM training (fine-tuning or RL), where activation memory from the generator's forward pass is the primary bottleneck. Here, gradient checkpointing is highly effective, reducing activation memory from ~8-12 GB to ~1-2 GB for the generator training phase. This is especially impactful for AdvPrompter's dual-model setup, where both the target LLM and generator must co-reside in GPU memory.

We have integrated both techniques into our framework and will include the detailed analysis and implementation in the camera-ready version. We note that these optimizations are orthogonal to PSKV and can be combined with it for further efficiency gains.

11.修正夸张的urgent bottleneck表述.
