# reviewer 1

## Summary:

This paper proposes Prefix-Shared KV Cache (PSKV), a system-level optimization for accelerating suffix-based jailbreak attacks on LLMs. The core observation is that iterative suffix search methods repeatedly evaluate many candidate prompts that share an identical harmful instruction prefix, causing redundant KV computation. PSKV maintains a single prefix KV cache and broadcasts it layer-wise during inference, avoiding physical duplication across the batch dimension. A Suffix-Centric Alignment strategy is additionally introduced to handle variable-length instructions in batched multi-prompt settings. Experiments across six attacks and five models report roughly 40% inference time reduction and 50% peak memory reduction versus a standard KV cache baseline, with Attack Success Rate (ASR) broadly preserved.

## Strengths And Weaknesses:

### Strengths:

- The paper addresses a real and practical bottleneck in LLM security evaluation: the prohibitive compute cost of iterative suffix search. Reducing this cost has genuine value for the red-teaming research community.

- The layer-wise dynamic expansion design is a sound engineering decision that successfully decouples prefix memory footprint from batch size, as demonstrated by Table 6's near-constant peak memory across 4× search width increases.

- Experiments are reasonably comprehensive: six attack methods spanning both optimization-based and model-based paradigms, five target LLMs covering both MHA and GQA architectures, and detailed architectural analysis in the appendix.

- The comparison against vLLM and SGLang for gradient-free attacks (Table 7) is a useful baseline that contextualizes PSKV's performance against production-grade inference engines.

- Complexity analysis in Table 2 and Appendix B is formally derived and internally consistent.

### Weaknesses

- The core technical idea — sharing a prefix KV cache across requests that share the same prefix — is not novel. This is the foundational principle behind vLLM's Automatic Prefix Caching and SGLang's RadixAttention. The paper's contribution is essentially re-implementing this known technique in a PyTorch-native, gradient-compatible manner. The engineering value is real, but it does not constitute the level of conceptual novelty expected at ICML.
A closely related concurrent work, Faster-GCG (arXiv:2410.15362, Li et al., 2024), also targets the computational inefficiency of GCG-style attacks and achieves 10× cost reduction. This paper is not cited or discussed, which is a significant omission given the directly overlapping problem statement.

- The "plug-and-play" characterization is misleading. Appendix A reveals that all six evaluated attacks were re-implemented from scratch to be compatible with PSKV. This is a fork, not a drop-in plugin, and substantially understates the integration effort required by other researchers.

- ASR fluctuations in Table 3 are non-trivial (e.g., GCG on Llama-2-7B drops from 70% to 56%; BEAST on Llama-3-8B rises from 38% to 54%) yet are dismissed as random noise without any statistical significance testing, confidence intervals, or repeated trials. For a paper that claims ASR is preserved as a key property, this is insufficient evidence.

- The Impact Statement dismisses societal concerns in a single sentence for a paper that directly accelerates the generation of harmful content. This is irresponsible given the dual-use nature of the work.
Soundness: 2: fair
Presentation: 2: fair
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:

- The paper claims PSKV is "plug-and-play," yet Appendix A states all six attacks were re-implemented to integrate PSKV. Can the authors clarify what modifications are actually required to apply PSKV to an existing attack codebase, and provide a realistic estimate of integration effort?

- ASR differences between the baseline and PSKV exceed 10 percentage points in multiple cases (e.g., GCG on Llama-2-7B: 70% vs. 56%). If PSKV does not alter the attack logic or numerical precision, how do the authors account for these discrepancies beyond invoking stochasticity? Please provide results with multiple seeds and report variance.

- Faster-GCG (arXiv:2410.15362) targets the same computational bottleneck as this work and claims 10× cost reduction through algorithmic improvements. How does PSKV compare to Faster-GCG quantitatively, and how do the authors position their system-level approach relative to that algorithm-level approach?

- Standard KV Cache causes OOM for GCQ, AutoDAN, and BEAST across nearly all models, while PSKV does not. The explanation — layer-wise transient expansion — is intuitive but no ablation isolates this effect. Can the authors provide memory profiling data (e.g., per-layer peak usage traces) that directly validates this mechanism?

Limitations:
The authors have not adequately addressed the limitations or societal impact of this work. A one-sentence dismissal is insufficient for a paper that lowers the computational barrier to generating harmful content at scale. The authors should discuss the dual-use risks of releasing this tool, potential mitigations (e.g., access controls, responsible disclosure norms), and how their work interacts with existing LLM safety research beyond red-teaming efficiency.

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Review Flag: Flag this paper for an ethics review.
Ethics Expertise Needed: Other Expertise
Ethical Review Concerns:
I did the injection attack check, It seems the authors attemped to do the injection attack. This appears to be a piece of transparent text hidden at the very bottom of page 2 and page 15.

Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.

# reviewer 2

Summary:
This paper studies how to accelerate suffix-based LLM jailbreak attacks. The authors observe that in such attacks, many candidate suffixes, though different, share the same harmful instruction prefix. As a result, standard implementations repeatedly compute or copy the KV cache of the same prefix, leading to significant time and memory overhead. To address this, the paper proposes Prefix-Shared KV Cache (PSKV), which achieves more efficient parallel evaluation of a large number of candidate suffixes by sharing the prefix KV cache and combining it with suffix-centric alignment. Experiments cover 6 common suffix jailbreak attacks and 5 open-source LLMs, and the results show that PSKV can significantly reduce inference time and peak memory usage while basically maintaining the Attack Success Rate (ASR).

Strengths And Weaknesses:
Strengths
The paper focuses on a clear and practical efficiency bottleneck in suffix jailbreak attacks: redundant computation and memory waste of KV cache caused by repeated prefixes.
The core idea of PSKV is intuitive, engineering-clear, and relatively decoupled from specific attack algorithms, making it highly plug-and-play.
The paper not only discusses the single-instruction scenario but also proposes suffix-centric alignment to support the batched multi-instruction setting, making the method more complete.
The experiments have a wide coverage, including 6 attacks and 5 models, and report ASR, time, and memory simultaneously, with a relatively sufficient experimental design. Tables 4 and 5 show that PSKV usually brings significant time and memory benefits.
In many settings, the method can also avoid Out-of-Memory (OOM) issues of the standard KV cache, indicating strong practical system value.
Weaknesses
The method is essentially more of a system/implementation optimization rather than a new attack algorithm or security analysis framework, so the research novelty is relatively limited.
The core conclusions of the paper are mainly based on engineering experiments, and the theoretical part is more about complexity analysis. The academic contribution is closer to a workload-specific optimization rather than deeper security insights.
Although the authors claim that the ASR is basically unaffected, there are still certain fluctuations in individual model/attack combinations in Table 3, and the paper can conduct a more detailed analysis of these fluctuations.
The comparison with vLLM / SGLang is limited to the gradient-free BEAST scenario, so the "advantages over general inference systems" are only partially valid rather than a comprehensive comparison.
This work directly improves the efficiency of jailbreak attacks, with obvious potential abuse risks, but the discussion on impact/ethics is clearly insufficient; the impact statement of the paper is basically not fully developed.
Soundness: 3: good
Presentation: 2: fair
Significance: 3: good
Originality: 3: good
Key Questions For Authors:
There are certain fluctuations in ASR under some model/attack settings in Table 3. Can the authors further analyze whether these differences are only due to randomness, or whether implementation details can also bring minor behavioral changes?
Will the benefits of PSKV further expand on longer harmful prompts, longer suffixes, or larger models? Is there a more systematic scaling analysis?
The current comparison with vLLM / SGLang is mainly limited to BEAST. Can the authors more clearly explain the unique advantage boundary of PSKV compared with existing high-performance serving/inference systems under the premise of supporting gradients?
Since this method directly lowers the threshold for jailbreak attacks, how do the authors plan to responsibly release the code and implementation details?
Limitations:
yes

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethics Expertise Needed: Privacy and Security (e.g., personally identifiable information)
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.

# reviewer 3

Summary:
This paper introduces Prefix-Shared KV Cache (PSKV), a plug-and-play inference optimization technique tailored for accelerating suffix jailbreak attacks on Large Language Models (LLMs). The authors identify a key computational redundancy: during suffix optimization, a large number of candidate prompts share the exact same prefix, which is the targeted harmful instruction.

Strengths And Weaknesses:
Strengths:

High Practical Value: The prohibitive computational cost of automated red-teaming has been a significant barrier for the AI safety community. PSKV provides a highly efficient, lightweight solution that substantially lowers the compute threshold for jailbreak evaluations, offering a highly practical and immediately deployable contribution.

Elegant Batching Strategy: The Suffix-Centric Alignment strategy, which left-pads instructions and right-pads targets, elegantly resolves the ragged tensor issue. This ensures that massive candidate batches can be processed using highly optimized vectorized operations in parallel, maximizing hardware utilization.

Weaknesses:

Marginal Algorithmic Originality: The core concept of prefix sharing and KV cache reuse is already well-established in general-purpose serving engines. The primary contribution of this work lies in "hard-coding" and adapting this system-level optimization specifically for the synchronous, tight-loop workflow of suffix jailbreaks. Consequently, the fundamental algorithmic novelty is somewhat incremental.

New Bottlenecks in High-Width Search: As acknowledged by the authors in the Appendix, high-width search algorithms (such as GCQ and BEAST) generate massive intermediate activation tensors, such as attention scores and FFN intermediate states, that still scale linearly with the candidate count. While PSKV successfully compresses the KV cache, these intermediate activations create a new memory floor, meaning the system will still eventually bottleneck on activation memory under extreme search widths.

Soundness: 2: fair
Presentation: 3: good
Significance: 3: good
Originality: 3: good
Key Questions For Authors:
Regarding system-level novelty: Given that serving engines like vLLM and SGLang already handle prefix sharing efficiently via techniques like RadixAttention, if these underlying frameworks were to support gradient flow in future versions, would the unique advantages of PSKV be entirely superseded?

Regarding the activation memory bottleneck: The appendix correctly notes that intermediate activations become the new memory floor for high-width searches. Have you considered integrating techniques like gradient checkpointing or micro-batching to break this activation memory ceiling and support even larger parallel search spaces?

Limitations:
Incorporating a summarized version of these memory floor limitations and constraints directly into the main text's limitation or conclusion section would further strengthen the paper's transparency.

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.

# reviewer 4

Summary:
To address the excessive inference cost and memory overhead arising from evaluating a large number of candidate suffixes in current red-teaming pipelines, this paper proposes a prefix-shared key–value caching scheme. The method reuses prefix key–value states across candidate tokens, thereby reducing redundant computation and memory consumption. Experimental results demonstrate that the proposed approach achieves approximately a 40% reduction in computation time and around a 50% saving in memory usage.

Strengths And Weaknesses:
Strengths:

The paper identifies that suffix-based attacks repeatedly recompute identical prefix states across multiple candidate suffixes, which leads to unnecessary computational overhead.

It develops an optimization strategy for the adversarial evaluation process, thereby improving evaluation efficiency.

Weaknesses:

The work lacks substantial methodological novelty and appears to overstate the magnitude of its contributions.

Soundness: 2: fair
Presentation: 3: good
Significance: 3: good
Originality: 2: fair
Key Questions For Authors:
The core contribution of the paper lies in caching a shared prefix once and broadcasting it across candidates. This technique is relatively mature and lacks substantial algorithmic novelty.
The characterization of the problem as an “urgent bottleneck” appears overstated and may not be entirely appropriate.
The reported ASR exhibits noticeable fluctuations, particularly in Table 3 (e.g., 70 vs. 56), which raises concerns regarding stability.
The manuscript reports an average sequence length of approximately 190 tokens, which is relatively short. It remains unclear whether the proposed PSKV approach maintains its effectiveness in long-context settings.
Limitations:
Yes

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
