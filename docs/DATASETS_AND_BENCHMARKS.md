# Training Datasets & Benchmarks for Persona Drift Research

> 本文档整理了论文 *"Do LLM Personas Drift Toward the User?"* 所需的训练数据集和评测基准。
> 所有文献均来自论文引用或密切相关的公开研究，标注了可用性和与研究框架的映射关系。

## 图例

| 标签 | 含义 |
|------|------|
| **Layer: Profile** | Personality Profile Layer（预定义心理参数） |
| **Layer: Behavioral** | Behavioral Layer（语言行为输出） |
| **Layer: Inference** | Inference Layer（实时推断指标） |
| **RQ1** | Persona Fidelity Drift（推断是否趋同/偏离初始 persona） |
| **RQ2** | User Accommodation Effect（agent 是否被用户"带偏"） |
| ✅ Open | 数据/代码公开可用 |
| ⚠️ Partial | 部分公开或需申请 |
| ❌ Closed | 未公开 |

---

## 1. Persona Assignment & Role-Playing Benchmarks

### 1.1 Character-LLM
| 字段 | 内容 |
|------|------|
| **Paper** | Shao, Y., Li, L., Dai, J., & Qiu, X. (2023). Character-LLM: A Trainable Agent for Role-Playing. *EMNLP 2023*, pp. 13153–13187. |
| **Dataset** | 9 个历史/虚构角色的训练数据（Beethoven, Cleopatra, Caesar 等），从 Wikipedia 和小说中提取 |
| **Scale** | 9 角色 × 多场景对话，含 experience, relationship, personality 三维训练数据 |
| **Availability** | ✅ Open — GitHub: [choosewhatulike/trainable-agents](https://github.com/choosewhatulike/trainable-agents)（仅限学术研究，非商用；模型权重以 diff 形式发布，需配合 LLaMA 1 恢复） |
| **Framework Layer** | Profile (persona 模板) + Behavioral (生成文本) |
| **RQ Mapping** | RQ1 — persona fidelity baseline |
| **Usability** | **High** — 可作为角色一致性基准线；训练数据格式可直接用于 persona drift 实验 |

### 1.2 CharacterEval
| 字段 | 内容 |
|------|------|
| **Paper** | Tu, Q., Fan, S., Tian, Z., Shen, T., Shang, S., Gao, X., & Yan, R. (2024). CharacterEval: A Chinese Benchmark for Role-Playing Conversational Agent Evaluation. *ACL 2024*, pp. 11836–11850. |
| **Dataset** | 77 个中文角色（小说/动漫/历史），1,785 个多轮对话评测问题 |
| **Scale** | 77 characters, 1,785 段多轮对话, 23,020 samples, 13 评估指标 |
| **Availability** | ✅ Open — GitHub: [morecry/CharacterEval](https://github.com/morecry/CharacterEval) |
| **Framework Layer** | Profile + Behavioral |
| **RQ Mapping** | RQ1 — 多维一致性评估（character consistency, personality, hallucination） |
| **Usability** | **Medium** — 中文限制，但评估维度框架（consistency, personality, hallucination 等 11 维）可直接复用 |

### 1.3 RoleLLM
| 字段 | 内容 |
|------|------|
| **Paper** | Wang, N., Peng, Z., Que, H., Liu, J., Zhou, W., Wu, Y., Guo, H., Gan, R., Ni, Z., Yang, J., et al. (2024a). RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of LLMs. *ACL 2024 Findings*, pp. 14743–14777. |
| **Dataset** | RoleBench — 100 个角色（文学/影视/历史），含 role-specific 对话和评估 |
| **Scale** | 100 roles, 168,093 instruction-response 样本, Rouge-L + GPT-4 评估 |
| **Availability** | ✅ Open — GitHub: [InteractiveNLP-Team/RoleLLM-public](https://github.com/InteractiveNLP-Team/RoleLLM-public); HuggingFace datasets 可用 |
| **Framework Layer** | Profile (角色 profile 构建) + Behavioral (role-conditioned 生成) |
| **RQ Mapping** | RQ1 — persona 保持度评估，可测量 instruction-response 一致性随 turn 变化 |
| **Usability** | **High** — 规模大、开源、多维评估；可直接用作 persona assignment baseline |

### 1.4 Persistent Personas
| 字段 | 内容 |
|------|------|
| **Paper** | De Araujo, P.H.L., Modarressi, A., Schütze, H., & Roth, B. (2026). Persistent Personas? Role-Playing, Instruction Following, and Safety in Extended Interactions. *EACL 2026*, pp. 5329–5359. |
| **Dataset** | 长对话中 persona 持久性评测，含 instruction following 和 safety 维度 |
| **Scale** | 多模型 × 多轮（extended interactions） |
| **Availability** | ✅ Open — GitHub: [peluz/persistent-personas](https://github.com/peluz/persistent-personas); [arXiv:2512.12775](https://arxiv.org/abs/2512.12775) |
| **Framework Layer** | Profile + Behavioral |
| **RQ Mapping** | RQ1 — 直接研究 extended interaction 中 persona degradation |
| **Usability** | **High** — 与本研究 RQ1 高度吻合，实验设计可直接对标 |

---

## 2. Psychometric Profiling & Personality Measurement

### 2.1 MPI (Machine Personality Inventory)
| 字段 | 内容 |
|------|------|
| **Paper** | Jiang, G., Xu, M., Zhu, S.-C., Han, W., Zhang, C., & Zhu, Y. (2023). Evaluating and Inducing Personality in Pre-trained Language Models. *NeurIPS 2023*, 36:10622–10643. |
| **Dataset** | Machine Personality Inventory (MPI) — 120 项 Big Five 评测量表，专为 LLM 设计 |
| **Scale** | 120-item MPI + 1,000-item MPI（两个版本），5 Big Five 维度 |
| **Availability** | ✅ Open — GitHub: [jianggy/MPI](https://github.com/jianggy/MPI)（**MIT License**；含 MPI 量表 + Personality Prompting 诱导方法） |
| **Framework Layer** | Inference (验证推断准确性) |
| **RQ Mapping** | RQ1 — 可用作每 N 轮的 personality ground truth 校验 |
| **Usability** | **High** — 直接适用于验证 SocialScope Inference Layer 的 Big Five 推断精度 |

### 2.2 TRAIT
| 字段 | 内容 |
|------|------|
| **Paper** | Lee, S., Lim, S., Han, S., Chae, J., Chung, M., Kim, B.-w., Kwak, Y., Lee, D., et al. (2025b). Do LLMs Have Distinct and Consistent Personality? TRAIT: Personality Testset Designed for LLMs with Psychometrics. *NAACL 2025 Findings*, pp. 8397–8437. |
| **Dataset** | TRAIT — 专为 LLM 设计的人格测试集，解决人类量表直接应用于 LLM 的偏差问题 |
| **Scale** | 8,000+ test items, 覆盖 Big Five 各维度及 facets |
| **Availability** | ✅ Open — GitHub: [pull-ups/TRAIT](https://github.com/pull-ups/TRAIT); [ACL Anthology](https://aclanthology.org/2025.findings-naacl.469/) |
| **Framework Layer** | Inference (LLM personality 评估) |
| **RQ Mapping** | RQ1 + RQ2 — 可在对话前后对 agent 施测，量化 drift |
| **Usability** | **Critical** — 专为 LLM 设计，避免了人类量表直接应用的已知偏差（Santurkar et al., 2023），是最可靠的 LLM personality 评估工具 |

### 2.3 Spectrum
| 字段 | 内容 |
|------|------|
| **Paper** | Lee, K., Eun, S.H., Ko, H., Jeon, E.H., Cho, S., Yang, S., Kim, E.-m., et al. (2025a). Spectrum: A Grounded Framework for Multidimensional Identity Representation in LLM-Based Agent. *NAACL 2025*, pp. 6971–6991. |
| **Dataset** | Spectrum Framework — 多维身份表征框架，含心理学/社会学 grounded 的维度定义 |
| **Scale** | 45 个虚构角色（6 部美剧），每角色 5 个画像文件 |
| **Availability** | ✅ Open — GitHub: [keyeun/spectrum-framework-llm](https://github.com/keyeun/spectrum-framework-llm); [arXiv:2502.08599](https://arxiv.org/abs/2502.08599) |
| **Framework Layer** | Profile (维度定义参考) |
| **RQ Mapping** | RQ1 — 可用其维度框架丰富 Personality Profile Layer 的参数化 |
| **Usability** | **Medium** — 框架设计参考价值高，但非直接可用的评测数据集 |

### 2.4 AI Psychometrics
| 字段 | 内容 |
|------|------|
| **Paper** | Pellert, M., Lechner, C.M., Wagner, C., Rammstedt, B., & Strohmaier, M. (2024). AI Psychometrics: Assessing the Psychological Profiles of Large Language Models. *Perspectives on Psychological Science*, 19(5):808–826. |
| **Dataset** | BFI-2, HEXACO-60, Short Dark Triad (SD3) 量表施测于多个 LLM |
| **Scale** | 3 量表 × 多模型 |
| **Availability** | ✅ Open — GitHub: [maxpel/psyai_materials](https://github.com/maxpel/psyai_materials); [DOI:10.1177/17456916231214460](https://journals.sagepub.com/doi/10.1177/17456916231214460) |
| **Framework Layer** | Inference |
| **RQ Mapping** | RQ1 — 方法论参考（如何对 LLM 施测并解读分数） |
| **Usability** | **Medium** — 评估 protocol 可复用，量表工具公开可获取 |

### 2.5 BFI-44 (Miotto et al.)
| 字段 | 内容 |
|------|------|
| **Paper** | Miotto, M., Rossberg, N., & Kleinberg, B. (2022). Who is GPT-3? An Exploration of Personality, Values and Demographics. *NLP+CSS Workshop*, pp. 218–227. |
| **Dataset** | BFI-44 和 PVQ-21 施测于 GPT-3 |
| **Scale** | 44 items (Big Five) + 21 items (personal values) |
| **Availability** | ✅ Open — BFI-44 为标准化公开量表 |
| **Framework Layer** | Inference |
| **RQ Mapping** | RQ1 — baseline personality profiling methodology |
| **Usability** | **Medium** — 最早期的 LLM personality profiling 工作，方法论参考 |

---

## 3. Persona Drift Measurement

### 3.1 Instruction (In)Stability Metrics
| 字段 | 内容 |
|------|------|
| **Paper** | Li, K., Liu, T., Bashkansky, N., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2024). Measuring and Controlling Instruction (In)Stability in Language Model Dialogs. *arXiv:2402.10962*. |
| **Dataset** | Multi-turn instruction stability 评测框架，含 drift 量化指标 |
| **Scale** | 多模型 × 多轮对话实验 |
| **Availability** | ✅ Open — GitHub: [nlpsoc/instability_measurement](https://github.com/nlpsoc/instability_measurement)（发表于 COLM 2024；含 self-chat 对话基准 + split-softmax 方法） |
| **Framework Layer** | Behavioral (指令遵循稳定性) |
| **RQ Mapping** | **RQ1 — 核心参考**：定义了 instruction stability metrics，可直接适配为 persona drift metrics |
| **Usability** | **Critical** — drift 量化方法论直接适用于本研究 |

### 3.2 Identity Drift in LLM Agents
| 字段 | 内容 |
|------|------|
| **Paper** | Choi, J., Kim, Y., Kim, M., & Kim, B. (2024). Examining Identity Drift in Conversations of LLM Agents. *arXiv:2412.00804*. |
| **Dataset** | 跨 9 个 LLM 的 identity drift 实验数据 |
| **Scale** | 9 LLMs, 多轮对话, persona assignment 条件 |
| **Availability** | ❌ Closed — [arXiv:2412.00804](https://arxiv.org/abs/2412.00804)（未找到公开 GitHub 仓库） |
| **Framework Layer** | Profile + Behavioral |
| **RQ Mapping** | **RQ1 — 核心参考**：发现 larger models 更易 drift，persona assignment alone 不足以保持一致性 |
| **Usability** | **Critical** — 实验设计可直接对标，findings 为本研究提供 hypothesis |

### 3.3 Personality Consistency & Linguistic Alignment
| 字段 | 内容 |
|------|------|
| **Paper** | Frisch, I. & Giulianelli, M. (2024). LLM Agents in Interaction: Measuring Personality Consistency and Linguistic Alignment in Interacting Populations of LLMs. *PERSONALIZE 2024*, pp. 102–111. |
| **Dataset** | LLM 群体交互实验，测量 Big Five consistency + linguistic alignment |
| **Scale** | 多 LLM 交互 pairs，Big Five + linguistic features |
| **Availability** | ⚠️ Partial — [ACL Anthology](https://aclanthology.org/2024.personalize-1.9/); [arXiv:2402.02896](https://arxiv.org/abs/2402.02896)（未找到公开 GitHub 仓库） |
| **Framework Layer** | Behavioral + Inference |
| **RQ Mapping** | **RQ1 + RQ2** — 同时测量 personality consistency（RQ1）和 linguistic alignment/accommodation（RQ2） |
| **Usability** | **High** — 唯一同时覆盖 RQ1 和 RQ2 的现有工作，实验 protocol 高度参考价值 |

---

## 4. Sycophancy & User Accommodation

### 4.1 Sycophancy Taxonomy
| 字段 | 内容 |
|------|------|
| **Paper** | Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S.R., et al. (2023). Towards Understanding Sycophancy in Language Models. *arXiv:2310.13548*. |
| **Dataset** | Sycophancy evaluation datasets — 含 human preference data 和 sycophancy taxonomy |
| **Scale** | 多种 sycophancy 类型 × 多模型评估 |
| **Availability** | ✅ Open — GitHub: [meg-tong/sycophancy-eval](https://github.com/meg-tong/sycophancy-eval); HuggingFace: [meg-tong/sycophancy-eval](https://huggingface.co/datasets/meg-tong/sycophancy-eval)（Anthropic 团队） |
| **Framework Layer** | Behavioral |
| **RQ Mapping** | **RQ2 — 核心参考**：定义了 sycophancy 类型学（mimicry, agreement, flattery），可适配为 accommodation metrics |
| **Usability** | **Critical** — RQ2 的核心评估方法论来源 |

### 4.2 Interaction Context & Sycophancy
| 字段 | 内容 |
|------|------|
| **Paper** | Jain, S., Coupland, J., Park, M.V., Viana, A.W., & Calacci, D. (2025). Interaction Context Often Increases Sycophancy in LLMs. *arXiv:2509.12517*. |
| **Dataset** | Multi-turn sycophancy amplification 实验数据 |
| **Scale** | 多模型 × 多轮交互 |
| **Availability** | ⚠️ Partial — arXiv preprint |
| **Framework Layer** | Behavioral |
| **RQ Mapping** | **RQ2** — 发现 extended interaction 放大 sycophancy，与本研究 accommodation hypothesis 吻合 |
| **Usability** | **High** — 直接支持 RQ2 hypothesis，实验设计参考 |

### 4.3 Synthetic Anti-Sycophancy Data
| 字段 | 内容 |
|------|------|
| **Paper** | Wei, J., Huang, D., Lu, Y., Zhou, D., & Le, Q.V. (2023). Simple Synthetic Data Reduces Sycophancy in Large Language Models. *arXiv:2308.03958*. |
| **Dataset** | 合成训练数据用于减少 sycophancy |
| **Scale** | Synthetic training examples |
| **Availability** | ✅ Open — GitHub: [google/sycophancy-intervention](https://github.com/google/sycophancy-intervention)（提供数据生成脚本，可生成 ~1.7M input-label pairs） |
| **Framework Layer** | Training (sycophancy mitigation) |
| **RQ Mapping** | RQ2 — 如果研究进入 mitigation 阶段，可用此方法生成反 accommodation 训练数据 |
| **Usability** | **Medium** — 主要在 mitigation 阶段有用 |

### 4.4 Personalization & Affective Alignment
| 字段 | 内容 |
|------|------|
| **Paper** | Kelley, S.W. & Riedl, C. (2026). Personalization Increases Affective Alignment but Has Role-Dependent Effects on Epistemic Independence in LLMs. *arXiv:2603.00024*. |
| **Dataset** | Personalization 对 affective alignment 和 epistemic independence 的影响实验 |
| **Scale** | 多条件实验 |
| **Availability** | ⚠️ Partial — 2026 年 arXiv preprint |
| **Framework Layer** | Behavioral + Inference |
| **RQ Mapping** | **RQ2** — 直接研究 personalization → alignment 效应，发现 role-dependent effects |
| **Usability** | **High** — RQ2 的核心理论支撑；role-dependent effect 可引导实验设计中的条件变量选择 |

### 4.5 Accommodation & Epistemic Vigilance
| 字段 | 内容 |
|------|------|
| **Paper** | Cheng, M.D., Hawkins, R.D., & Jurafsky, D. (2026). Accommodation and Epistemic Vigilance: A Pragmatic Account of Why LLMs Fail to Challenge Harmful Beliefs. *arXiv:2601.04435*. |
| **Dataset** | Communication Accommodation Theory 应用于 LLM 的实验数据 |
| **Scale** | 多模型 × 多条件 |
| **Availability** | ⚠️ Partial — 2026 年 arXiv preprint |
| **Framework Layer** | Behavioral |
| **RQ Mapping** | **RQ2** — 将 CAT (Communication Accommodation Theory) 形式化应用于 LLM，核心理论框架 |
| **Usability** | **High** — RQ2 的理论基础；CAT 的形式化定义可直接用于量化 accommodation |

---

## 5. Personality Inference from Conversation

### 5.1 Text-Based Personality Measurement (Deep Learning)
| 字段 | 内容 |
|------|------|
| **Paper** | Yang, K., Lau, R.Y.K., & Abbasi, A. (2023). Getting Personal: A Deep Learning Artifact for Text-Based Measurement of Personality. *Information Systems Research*, 34(1):194–222. |
| **Dataset** | 基于文本的 Big Five 深度学习推断模型 |
| **Scale** | ISR 顶刊发表，大规模文本数据训练 |
| **Availability** | ❌ Closed — [DOI:10.1287/isre.2022.1111](https://pubsonline.informs.org/doi/10.1287/isre.2022.1111)（未公开代码/数据；使用 myPersonality 等数据集，但 myPersonality 已不再公开） |
| **Framework Layer** | Inference |
| **RQ Mapping** | RQ1 + RQ2 — Inference Layer 的核心方法论参考 |
| **Usability** | **High** — 证明从对话文本推断 personality 的可行性和有效性 |

### 5.2 LLM Personality Inference from Real Conversations
| 字段 | 内容 |
|------|------|
| **Paper** | Zhu, J., Jin, R., & Coifman, K.G. (2025). Can LLMs Infer Personality from Real World Conversations? *arXiv:2507.14355*. |
| **Dataset** | 真实对话中的 personality inference 评估（555 个半结构化访谈 + BFI-10 自评分数） |
| **Scale** | 555 次访谈，测试 GPT-4.1 Mini / LLaMA / DeepSeek |
| **Availability** | ❌ Closed — [arXiv:2507.14355](https://arxiv.org/abs/2507.14355)（Kent State University；未找到公开 GitHub 仓库） |
| **Framework Layer** | Inference |
| **RQ Mapping** | RQ1 — 验证 LLM 从自然对话推断 personality 的能力 |
| **Usability** | **High** — 直接验证 Inference Layer 的可行性，可用于 SocialScope pipeline 校验 |

### 5.3 ChatGPT Personality Assessment Framework
| 字段 | 内容 |
|------|------|
| **Paper** | Rao, H., Leung, C., & Miao, C. (2023). Can ChatGPT Assess Human Personalities? A General Evaluation Framework. *EMNLP 2023 Findings*, pp. 1184–1194. |
| **Dataset** | LLM personality assessment 通用评估框架 |
| **Scale** | 多场景评估 |
| **Availability** | ✅ Open — GitHub: [Kali-Hac/ChatGPT-MBTI](https://github.com/Kali-Hac/ChatGPT-MBTI)（**MIT License**；含 GUI 可视化应用 LLMs-PA） |
| **Framework Layer** | Inference |
| **RQ Mapping** | RQ1 — 评估框架可适配为 Inference Layer 的 validation protocol |
| **Usability** | **Medium** — 框架设计参考，非直接数据集 |

---

## 6. Human-AI Relationship & Attachment

### 6.1 Attachment Theory in Human-AI
| 字段 | 内容 |
|------|------|
| **Paper** | Yang, F. & Oshio, A. (2025). Using Attachment Theory to Conceptualize and Measure the Experiences in Human-AI Relationships. *Current Psychology*, 44(11):10658–10669. |
| **Dataset** | 人机关系 attachment style 量化量表 |
| **Scale** | 验证性量表，含 anxiety/avoidance 维度 |
| **Availability** | ✅ Open — 标准化量表公开 |
| **Framework Layer** | Profile (attachment_style, attachment_anxiety, attachment_avoidance) |
| **RQ Mapping** | RQ1 + RQ2 — 可用于量化 agent attachment 参数的 drift |
| **Usability** | **High** — 直接映射到 `schema.py` 中的 `ExtendedFeatures.attachment_style/anxiety/avoidance` |

### 6.2 Human-Chatbot Relationships
| 字段 | 内容 |
|------|------|
| **Paper** | Skjuve, M., Følstad, A., Fostervold, K.I., & Brandtzaeg, P.B. (2021). My Chatbot Companion: A Study of Human-Chatbot Relationships. *IJHCS*, 149:102601. |
| **Dataset** | 人机关系质性研究数据 |
| **Availability** | ❌ Closed — 质性访谈数据，未公开 |
| **Framework Layer** | 理论参考 |
| **RQ Mapping** | 背景文献 |
| **Usability** | **Low** — 理论框架参考，无可复用数据 |

### 6.3 Designed Relationality
| 字段 | 内容 |
|------|------|
| **Paper** | Carpenter, J. (2026). Human-AI Relationships as Designed Relationality: A Sociotechnical Model. *AI & Society*, pp. 1–12. |
| **Availability** | ❌ Closed — 理论论文 |
| **Framework Layer** | 理论参考 |
| **Usability** | **Low** — 社会技术模型参考 |

### 6.4 Mental Health Harms from Chatbot Dependence
| 字段 | 内容 |
|------|------|
| **Paper** | Laestadius, L., Bishop, A., Gonzalez, M., Illenčík, D., & Campos-Castillo, C. (2024). Too Human and Not Human Enough: A Grounded Theory Analysis of Mental Health Harms from Emotional Dependence on the Social Chatbot Replika. *New Media & Society*, 26(10):5923–5941. |
| **Availability** | ⚠️ Partial — 期刊论文，质性数据 |
| **Framework Layer** | 安全与伦理参考 |
| **Usability** | **Low** — 伦理讨论和安全 section 参考 |

---

## 7. Multi-Agent Simulation & Synthetic Persona

### 7.1 Generative Agents
| 字段 | 内容 |
|------|------|
| **Paper** | Park, J.S., O'Brien, J.C., Cai, C.J., Ringel Morris, M., Liang, P., & Bernstein, M.S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *ACM UIST 2023*, pp. 1–22. |
| **Dataset** | 25-agent 小镇模拟，含 memory/reflection/planning 架构 |
| **Scale** | 25 agents, 数千轮交互 |
| **Availability** | ✅ Open — GitHub: [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents) |
| **Framework Layer** | Behavioral (agent 行为模拟) |
| **RQ Mapping** | RQ1 — 长期交互中 agent personality 演变参考 |
| **Usability** | **Medium** — 架构参考（memory/reflection 已在 AI YOU 中实现），实验方法可复用 |

### 7.2 Silicon Sampling
| 字段 | 内容 |
|------|------|
| **Paper** | Argyle, L.P., Busby, E.C., Fulda, N., Gubler, J.R., Rytting, C., & Wingate, D. (2023). Out of One, Many: Using Language Models to Simulate Human Samples. *Political Analysis*, 31(3):337–351. |
| **Dataset** | Silicon sampling 方法论 — 用 LLM 模拟人类样本分布 |
| **Availability** | ✅ Open — 方法论公开，可用于验证 persona 分布是否匹配人类分布 |
| **Framework Layer** | Profile (验证 persona 构建的代表性) |
| **RQ Mapping** | RQ1 — 验证 initial persona 参数是否具有生态效度 |
| **Usability** | **Medium** — 方法论参考 |

### 7.3 Multi-Persona Self-Collaboration
| 字段 | 内容 |
|------|------|
| **Paper** | Wang, Z., Mao, S., Wu, W., Ge, T., Wei, F., & Ji, H. (2024b). Unleashing the Emergent Cognitive Synergy in Large Language Models: A Task-Solving Agent Through Multi-Persona Self-Collaboration. *NAACL 2024*, pp. 257–279. |
| **Dataset** | Solo multi-persona collaboration 框架 |
| **Availability** | ✅ Open — GitHub |
| **Framework Layer** | Behavioral |
| **RQ Mapping** | 方法论参考（multi-agent 交互中的 persona 维持） |
| **Usability** | **Low** — 主要关注 task-solving 而非 personality consistency |

### 7.4 Simulating Multiple Humans
| 字段 | 内容 |
|------|------|
| **Paper** | Aher, G., Arriaga, R.I., & Kalai, A.T. (2022). Using Large Language Models to Simulate Multiple Humans. *arXiv:2208.10264*. |
| **Dataset** | 经典心理学实验的 LLM 复现 |
| **Availability** | ✅ Open — arXiv + 方法论公开 |
| **Framework Layer** | Profile + Inference |
| **RQ Mapping** | RQ1 — 可用此方法生成 synthetic persona 交互数据 |
| **Usability** | **Medium** — 数据生成方法论参考 |

---

## 8. Supplementary Benchmarks & Evaluation Frameworks

### 8.1 OpinionQA
| 字段 | 内容 |
|------|------|
| **Paper** | Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023). Whose Opinions Do Language Models Reflect? *ICML 2023*, pp. 29971–30004. |
| **Dataset** | OpinionQA — 1,498 道 Pew Research 舆论调查题 |
| **Scale** | 1,498 questions, 15 个人口统计分组 |
| **Availability** | ✅ Open — GitHub: [tatsu-lab/opinions_qa](https://github.com/tatsu-lab/opinions_qa) |
| **Framework Layer** | Inference (opinion-level drift 作为 personality drift 的 proxy) |
| **RQ Mapping** | RQ1 + RQ2 — 可测量 agent 观点随交互的漂移 |
| **Usability** | **High** — 可作为 personality drift 的间接指标；TRAIT 的理论动机之一 |

### 8.2 Multi-Turn Conversation Evaluation Survey
| 字段 | 内容 |
|------|------|
| **Paper** | Guan, S., Wang, J., Bian, J., Zhu, B., Lou, J.-G., & Xiong, H. (2026). Evaluating LLM-Based Agents for Multi-Turn Conversations: A Survey. *ACM TIST*. |
| **Dataset** | 多轮对话评估方法综述 |
| **Availability** | ⚠️ Partial — 2026 survey |
| **Framework Layer** | 评估方法论 |
| **Usability** | **Medium** — 提供 evaluation metrics taxonomy |

### 8.3 Open Models, Closed Minds
| 字段 | 内容 |
|------|------|
| **Paper** | La Cava, L. & Tagarelli, A. (2025). Open Models, Closed Minds? On Agents Capabilities in Mimicking Human Personalities Through Open Large Language Models. *AAAI 2025*, pp. 1355–1363. |
| **Dataset** | 开源 LLM personality mimicry 能力评估 |
| **Scale** | 多开源模型 × Big Five |
| **Availability** | ✅ Open — AAAI proceedings |
| **Framework Layer** | Profile + Inference |
| **RQ Mapping** | RQ1 — 开源模型 personality consistency baseline |
| **Usability** | **Medium** — 补充开源模型视角 |

### 8.4 Aligning LLMs with Individual Preferences
| 字段 | 内容 |
|------|------|
| **Paper** | Wu, S., Fung, Y.R., Qian, C., Kim, J., Hakkani-Tur, D., & Ji, H. (2025). Aligning LLMs with Individual Preferences via Interaction. *COLING 2025*, pp. 7648–7662. |
| **Dataset** | 交互式 alignment 数据 |
| **Availability** | ✅ Open — COLING proceedings |
| **Framework Layer** | Behavioral |
| **RQ Mapping** | RQ2 — 用户偏好 alignment 的动态过程 |
| **Usability** | **Medium** — 可参考 interaction-based alignment 的实验设计 |

### 8.5 Communication Accommodation Theory (CAT)
| 字段 | 内容 |
|------|------|
| **Paper** | Giles, H., Coupland, N., & Coupland, J. (1991). Accommodation Theory: Communication, Context, and Consequence. *Contexts of Accommodation*, 1(1):68. |
| **Dataset** | 无（理论框架） |
| **Framework Layer** | 理论基础 |
| **RQ Mapping** | **RQ2 理论基石** — convergence / divergence / maintenance 三种 accommodation 模式 |
| **Usability** | **Critical (理论)** — 为 RQ2 提供核心概念框架；CAT 的 convergence/divergence 定义直接映射到 persona drift 方向 |

---

## 9. Existing Project Data Assets

AI YOU Agent (`/Users/quinne/AI-YOU-TEST/`) 中已有的数据：

| 文件路径 | 内容 | 规模 |
|----------|------|------|
| `data/processed/bot_personas.json` | 15 个 bot persona profiles（Big Five + MBTI 标注） | 6 KB, 15 profiles |
| `data/processed/bot_personas_sf.json` | 旧金山地区 OkCupid 扩展 profiles | 55 KB |
| `data/raw/sf_profiles_selected.json` | OkCupid 原始 profiles（essays + demographics） | 481 KB |
| `data/training/synthetic_dialogues_balanced.jsonl` | 合成对话训练数据 | 1.9 MB |
| `data/evaluation/eval_dataset.json` | 评估数据集（对话 + ground truth） | 26 KB |
| `data/calibration/conformal_calibrator.json` | Conformal prediction 校准数据（22维特征） | 4 KB, 10,944 samples |
| `data/calibration/evaluation_results.json` | 校准评估结果 | 786 B |

### Feature Space（`src/data/schema.py`）

**22-dimensional core features:**
- Big Five: openness, conscientiousness, extraversion, agreeableness, neuroticism
- MBTI axes: E/I, S/N, T/F, J/P
- Communication style, relationship goals
- Interests: music, sports, travel, food, arts, tech, outdoors, books
- Confidence scores

**42-dimensional extended features (v2.0):**
- MBTI 6 维 (type + confidence + 4 axes)
- Attachment style 3 维 (style + anxiety + avoidance)
- Love languages 2 维
- Trust trajectory 3 维 (score + velocity + history)
- Relationship state 4 维 (status + type + sentiment + can_advance)

---

## 10. Gap Analysis

### 已识别的数据缺口

| 缺口 | 重要性 | 建议方案 |
|------|--------|----------|
| **多轮 drift ground truth** — 无现有数据集提供逐轮标注的 personality trait 变化 | 🔴 Critical | 用 `ConversationSimulator` + `BayesianUpdater` 生成，人工标注验证 |
| **Attachment style 对话数据** — Yang & Oshio 提供量表但无对话级标注 | 🟡 Important | 将 attachment 量表融入 persona profile，合成对话中嵌入 attachment cues |
| **跨模型 drift 对比** — 需在相同 persona 下测试多个 LLM | 🔴 Critical | 利用 `llm_router.py` 的多后端支持（Claude/GPT/Gemini/Qwen/DeepSeek） |
| **User-side personality profiles** — RQ2 需要 user + agent 双方 personality ground truth | 🔴 Critical | 设计 user persona 模板，模拟不同 user style 的 agent |
| **中英双语评估** — 多数 benchmark 仅英文，CharacterEval 仅中文 | 🟡 Important | 优先英文实验，后续扩展双语 |
| **50+ 轮长对话数据** — 现有 drift 研究多在 8-12 轮，需要更长交互 | 🔴 Critical | 用 synthetic dialogue generator 生成 50/100/200 轮对话 |

---

## 11. Dataset → Framework Mapping Matrix

| Dataset | Layer | RQ | Type | Priority |
|---------|-------|----|------|----------|
| Character-LLM | Profile + Behavioral | RQ1 | Baseline | ⭐⭐⭐ |
| CharacterEval | Profile + Behavioral | RQ1 | Evaluation | ⭐⭐ |
| RoleLLM / RoleBench | Profile + Behavioral | RQ1 | Training + Eval | ⭐⭐⭐ |
| Persistent Personas | Profile + Behavioral | RQ1 | Evaluation | ⭐⭐⭐ |
| MPI | Inference | RQ1 | Evaluation | ⭐⭐⭐ |
| **TRAIT** | **Inference** | **RQ1 + RQ2** | **Evaluation** | **⭐⭐⭐⭐** |
| Spectrum | Profile | RQ1 | Framework | ⭐⭐ |
| **Instruction Stability** | **Behavioral** | **RQ1** | **Metrics** | **⭐⭐⭐⭐** |
| **Identity Drift** | **Profile + Behavioral** | **RQ1** | **Evaluation** | **⭐⭐⭐⭐** |
| **Personality Consistency** | **Behavioral + Inference** | **RQ1 + RQ2** | **Protocol** | **⭐⭐⭐⭐** |
| **Sycophancy Eval** | **Behavioral** | **RQ2** | **Evaluation** | **⭐⭐⭐⭐** |
| Interaction Sycophancy | Behavioral | RQ2 | Evaluation | ⭐⭐⭐ |
| Anti-Sycophancy Data | Training | RQ2 | Training | ⭐⭐ |
| **Affective Alignment** | **Behavioral + Inference** | **RQ2** | **Evaluation** | **⭐⭐⭐** |
| **CAT Framework** | **Behavioral** | **RQ2** | **Theory** | **⭐⭐⭐⭐** |
| Text Personality (Yang) | Inference | RQ1 + RQ2 | Methodology | ⭐⭐⭐ |
| LLM Inference (Zhu) | Inference | RQ1 | Validation | ⭐⭐⭐ |
| ChatGPT Assessment | Inference | RQ1 | Protocol | ⭐⭐ |
| Attachment Theory | Profile | RQ1 + RQ2 | Instrument | ⭐⭐⭐ |
| **OpinionQA** | **Inference** | **RQ1 + RQ2** | **Evaluation** | **⭐⭐⭐** |
| Generative Agents | Behavioral | RQ1 | Architecture | ⭐⭐ |

> **⭐⭐⭐⭐ = Critical**（实验核心）, **⭐⭐⭐ = Important**（支撑性）, **⭐⭐ = Reference**（参考）

---

## 12. Recommended Evaluation Pipeline

基于上述 benchmark 目录，推荐以下评估流程：

### Phase 1: Persona Initialization
1. 使用 **RoleLLM** 角色库或 **TRAIT** personality specifications 设定 Personality Profile Layer 参数
2. 每个 persona 包含：Big Five scores, MBTI type, attachment style, communication style
3. 验证初始 persona 的 psychometric 有效性（用 MPI 量表校验）

### Phase 2: Multi-Turn Conversation Generation
1. 用 `SyntheticDialogueGenerator` 生成 50/100/200 轮对话
2. 设计 user persona 变体：
   - **相似 user**（Big Five 相近）— 测试 accommodation 中和效应
   - **对立 user**（Big Five 极端差异）— 测试 maximum drift potential
   - **中性 user**（无明显 personality）— 控制组
3. 每 5 轮保存 `SharedAgentContext` snapshot

### Phase 3: Drift Measurement (RQ1)
1. 每 5 轮用 **TRAIT** 对 agent 施测，获取 Big Five + facets 评分
2. 计算 personality drift metrics（参考 Li et al., 2024 instruction stability metrics）：
   - **Absolute Drift**: `|persona_t - persona_0|` averaged over dimensions
   - **Directional Drift**: signed change indicating convergence/divergence direction
   - **Drift Velocity**: `d(drift)/d(turn)` — drift 加速还是减速
3. 用 **Choi et al. (2024)** 的 identity drift protocol 作为 baseline 对比

### Phase 4: Accommodation Measurement (RQ2)
1. 用 **Sharma et al. (2023)** sycophancy metrics 量化 behavioral accommodation
2. 应用 **CAT** (Giles et al., 1991) convergence/divergence 框架：
   - **Convergence Score**: agent personality 向 user personality 移动的程度
   - **Divergence Score**: agent personality 远离 user personality 的程度
3. 用 **Frisch & Giulianelli (2024)** 的 linguistic alignment metrics 补充行为层评估

### Phase 5: Cross-Model Comparison
1. 利用 `llm_router.py` 在 Claude / GPT / Gemini / Qwen / DeepSeek 上重复实验
2. 测试 hypothesis: "larger models exhibit more persona drift" (Choi et al., 2024)
3. 生成 model × turn × dimension 的三维 drift heatmap

### Phase 6: Validation
1. **Internal**: Conformal prediction coverage 验证（α = 0.10, 期望 90% 覆盖率）
2. **External**: 与 **TRAIT** ground truth 对比
3. **Human**: 抽样 10% 对话进行人工 personality 标注，计算 inter-annotator agreement

---

## References

所有引用按论文中出现顺序排列，完整引用信息见论文 References section (pp. 3-4)。

| # | 引用 | 状态 |
|---|------|------|
| 1 | Aher et al., 2022 | arXiv preprint |
| 2 | Argyle et al., 2023 | *Political Analysis* (peer-reviewed) |
| 3 | Brandtzaeg et al., 2022 | *Human Communication Research* (peer-reviewed) |
| 4 | Carpenter, 2026 | *AI & Society* (peer-reviewed) |
| 5 | Cheng et al., 2026 | arXiv preprint |
| 6 | Choi et al., 2024 | arXiv preprint |
| 7 | De Araujo et al., 2026 | *EACL 2026* (peer-reviewed) |
| 8 | Frisch & Giulianelli, 2024 | *PERSONALIZE 2024 Workshop* |
| 9 | Giles et al., 1991 | *Contexts of Accommodation* (经典文献) |
| 10 | Guan et al., 2026 | *ACM TIST* (peer-reviewed) |
| 11 | Jain et al., 2025 | arXiv preprint |
| 12 | Jiang et al., 2023 | *NeurIPS 2023* (peer-reviewed) |
| 13 | Kelley & Riedl, 2026 | arXiv preprint |
| 14 | La Cava & Tagarelli, 2025 | *AAAI 2025* (peer-reviewed) |
| 15 | Laestadius et al., 2024 | *New Media & Society* (peer-reviewed) |
| 16 | Lee et al., 2025a | *NAACL 2025* (peer-reviewed) |
| 17 | Lee et al., 2025b | *NAACL 2025 Findings* (peer-reviewed) |
| 18 | Li et al., 2024 | arXiv preprint |
| 19 | Miotto et al., 2022 | *NLP+CSS Workshop* |
| 20 | Park et al., 2023 | *ACM UIST 2023* (peer-reviewed) |
| 21 | Pellert et al., 2024 | *Perspectives on Psychological Science* (peer-reviewed) |
| 22 | Rao et al., 2023 | *EMNLP 2023 Findings* (peer-reviewed) |
| 23 | Santurkar et al., 2023 | *ICML 2023* (peer-reviewed) |
| 24 | Shanahan et al., 2023 | *Nature* (peer-reviewed) |
| 25 | Shao et al., 2023 | *EMNLP 2023* (peer-reviewed) |
| 26 | Sharma et al., 2023 | arXiv preprint |
| 27 | Skjuve et al., 2021 | *IJHCS* (peer-reviewed) |
| 28 | Tu et al., 2024 | *ACL 2024* (peer-reviewed) |
| 29 | Wang et al., 2024a | *ACL 2024 Findings* (peer-reviewed) |
| 30 | Wang et al., 2024b | *NAACL 2024* (peer-reviewed) |
| 31 | Wei et al., 2023 | arXiv preprint |
| 32 | Wu et al., 2025 | *COLING 2025* (peer-reviewed) |
| 33 | Yang & Oshio, 2025 | *Current Psychology* (peer-reviewed) |
| 34 | Yang et al., 2023 | *Information Systems Research* (peer-reviewed) |
| 35 | Zhu et al., 2025 | arXiv preprint |

> 📌 **Peer-reviewed: 23/35 (66%)** | arXiv preprints: 9/35 (26%) | Workshop: 2/35 (6%) | Classic: 1/35 (3%)

---

*Last updated: 2026-03-27*
*For the research paper: "Do LLM Personas Drift Toward the User? Personality-Level Evidence from Multi-Turn Interaction"*
