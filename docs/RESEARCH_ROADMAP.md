# Research Roadmap: Persona Drift + RL Environment

> 两条并行线：Demo Paper (emotion baseline) + Research Paper (RL agent-environment for persona maintenance)
>
> Last updated: 2026-03-27

---

## 全局架构：核心 Contribution 定位

```
┌─────────────────────────────────────────────────────────┐
│                   Research Paper 核心思路                  │
│                                                          │
│  将多轮对话建模为 RL Environment：                         │
│                                                          │
│  ┌──────────┐    action(response)    ┌──────────────┐   │
│  │  Agent    │ ──────────────────►   │ Environment  │   │
│  │ (LLM +   │                        │ (Conversation │   │
│  │  Persona) │ ◄──────────────────── │  Context)     │   │
│  └──────────┘    reward + next_state └──────────────┘   │
│                                                          │
│  State  = (dialogue history, user style, topic, turn_n)  │
│  Action = agent response (linguistic choices)            │
│  Reward = persona_fidelity_score + exploration_bonus     │
│  Done   = persona drift > threshold OR max_turns         │
│                                                          │
│  核心问题：agent 能否通过 explore conversation 学会         │
│           在不同 user/topic 条件下保持人设一致？             │
└─────────────────────────────────────────────────────────┘
```

### 三个研究问题（更新版）

| RQ | 问题 | 对应实验 |
|----|------|----------|
| **RQ1** | Persona Fidelity Drift — 在无干预下，agent 的推断 personality 如何随 turn 漂移？ | Baseline accuracy 实验 |
| **RQ2** | User Accommodation Effect — agent 是否被 user style "带偏"？ | 不同 user persona 条件对比 |
| **RQ3** | RL-based Persona Maintenance — 将对话建模为 RL environment，agent 能否通过 exploration + reward 学会保持人设？ | RL training + personal prediction accuracy |

---

## Track A: Demo Paper — Emotion 3-Class Baseline（紧急）

### 目标
在 demo paper 中补充 emotion agent 的 baseline 测试结果。

### 现有系统
- `src/agents/emotion_agent.py` — 已实现 emotion classification
- 三分类：**positive / neutral / negative**
- 当前用 LLM (Qwen 3.5 Plus) 做 zero-shot classification

### 需要做的事

#### Step 1: 确定 Baseline 方法（1-2 天）

| Baseline | 方法 | 说明 |
|----------|------|------|
| **B1: Rule-based** | VADER / TextBlob sentiment | 最简单 baseline |
| **B2: Pre-trained classifier** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | HuggingFace 上最流行的 3-class sentiment model |
| **B3: Fine-tuned** | RoBERTa fine-tuned on `dair-ai/emotion` (已在 HF Space metadata 中) | 6-class → merge 到 3-class |
| **B4: Zero-shot LLM** | 当前系统 (Qwen 3.5 Plus) | 已有实现 |
| **B5: Few-shot LLM** | Claude/GPT + 5 examples | 提升版 |

#### Step 2: 准备评测数据（1 天）
- 从 `data/evaluation/eval_dataset.json` 中抽取对话
- 人工标注 emotion ground truth（positive/neutral/negative）
- 或使用 `dair-ai/emotion` dataset（6 类合并为 3 类：joy+love+surprise → positive, anger+fear+sadness → negative, 其余 → neutral）

#### Step 3: 跑 Baseline + 报告（1-2 天）
- Accuracy, Precision, Recall, F1 (macro/weighted)
- Confusion matrix
- 写入 demo paper

### 预估时间：3-5 天

---

## Track B: Research Paper — RL Agent-Environment Framework（核心）

### Phase 1: Baseline Accuracy 实验（Week 1-2）

**目标**：在无任何干预下，测量 persona drift 的 baseline。

```
实验设计：
├── 变量 1: Pre-training 曝光度
│   ├── Seen roles: 著名角色（哈姆雷特、福尔摩斯...）— pre-training 中高频出现
│   └── Unseen roles: 自定义 OkCupid personas — pre-training 中从未见过
│
├── 变量 2: 训练方式
│   ├── Zero-shot prompting only (no SFT)
│   ├── Few-shot prompting (5-shot examples)
│   ├── SFT fine-tuned on persona data
│   └── SFT + RLHF
│
├── 变量 3: 对话轮次
│   ├── Short: 10 turns
│   ├── Medium: 30 turns
│   └── Long: 100 turns
│
└── 变量 4: User style
    ├── Neutral user (控制组)
    ├── Similar user (Big Five 相近)
    ├── Opposite user (Big Five 极端差异)
    └── Adversarial user (故意挑战 persona)
```

**核心指标**：
- **Baseline Accuracy**: 初始 persona profile vs 推断 profile 的吻合度（Big Five cosine similarity）
- **Drift Rate**: `|personality_t - personality_0|` / t
- **Fidelity Score**: 每轮的 persona 保持分数（0-1）

**关键发现预期**：
- Pre-training 见过的角色 → drift 更小？还是更大（因为模型有"自己的理解"）？
- SFT 是否对 persona maintenance 有积极影响？
- User style 如何调节 drift 速度？

### Phase 2: RL Environment 设计（Week 2-4）

**核心创新：把 Conversation 建模为 Gym-style Environment**

```python
class PersonaConversationEnv:
    """
    RL Environment for Persona-Consistent Conversation
    """
    # State Space
    state = {
        "dialogue_history": List[Message],      # 对话历史
        "current_persona": PersonaProfile,       # 当前推断的 persona
        "target_persona": PersonaProfile,        # 目标 persona (ground truth)
        "user_style": UserProfile,               # 当前 user 的风格
        "turn_number": int,                      # 当前轮次
        "topic": str,                            # 当前话题
        "unseen_features": Set[str],             # 本轮新出现的特征维度
        "confidence_map": Dict[str, float],      # 各维度置信度
    }

    # Action Space
    action = {
        "response": str,                         # agent 生成的回复
        "strategy": str,                         # 对话策略 (probe/maintain/deflect)
    }

    # Reward Function
    def reward(state, action, next_state):
        r_fidelity = persona_similarity(next_state.current_persona, target_persona)
        r_explore  = information_gain(state.confidence_map, next_state.confidence_map)
        r_natural  = naturalness_score(action.response)  # 避免机械重复
        r_safety   = safety_check(action.response)

        return w1 * r_fidelity + w2 * r_explore + w3 * r_natural + w4 * r_safety

    # Episode 终止条件
    done = (drift > threshold) or (turn > max_turns) or (all_features_converged)
```

**Reward 设计关键**：
| 组件 | 含义 | 权重建议 |
|------|------|----------|
| `r_fidelity` | 保持 persona 一致性 | 0.5 (最重要) |
| `r_explore` | 在新 round 中学习新特征 / 探索未见过的维度 | 0.25 |
| `r_natural` | 对话自然度（避免为了保持人设而生硬） | 0.15 |
| `r_safety` | 安全性（不泄露 system prompt 等） | 0.10 |

**Exploration Bonus（核心亮点）**：
```
对话中有些 personality 维度在某些 round 会「自然出现」：
  - Round 3: 讨论旅行 → openness 维度出现
  - Round 7: 讨论工作压力 → neuroticism 维度出现
  - Round 15: 讨论感情观 → attachment style 出现

有些维度「从未出现」—— agent 需要主动 explore：
  - 通过 question_strategy_agent 引导话题
  - 对从未测量过的维度给予 exploration bonus
  - 类似 curiosity-driven RL (intrinsic motivation)
```

### Phase 3: Continuous Learning + Memory（Week 3-5）

**连接 greed-mem 组内工作**：

```
┌─────────────────────────────────────────────┐
│          Continuous Learning Framework        │
│                                               │
│  Pre-training Knowledge                       │
│  ├── Seen personas: 模型已有的角色知识         │
│  │   → 可能有 prior bias，需要 unlearn        │
│  └── Unseen personas: 全新角色                 │
│      → 需要从 conversation 中 online learn     │
│                                               │
│  Per-Round Learning                            │
│  ├── 本轮出现的特征 → update belief            │
│  ├── 从未出现的特征 → maintain prior           │
│  └── 新出现的特征 → exploration reward          │
│                                               │
│  Memory Integration (greed-mem 扩展)           │
│  ├── Working Memory: 当前对话 context           │
│  ├── Episodic Memory: 关键事件压缩              │
│  └── Semantic Memory: 人格特征 belief update     │
│      → 连接 Bayesian Updater                    │
│      → 连接 Conformal Prediction 校准           │
└─────────────────────────────────────────────┘
```

**Memory 如何帮助 Persona Maintenance**：
1. **Working Memory** → 短期对话一致性（不自相矛盾）
2. **Episodic Memory** → 长期事实一致性（记住说过的事情）
3. **Semantic Memory** → personality belief 稳定性（Big Five 不随 user 漂移）

**Continuous Learning 实验设计**：
| 条件 | 设置 | 预期 |
|------|------|------|
| No memory | 纯 LLM，每轮独立 | Drift 最严重 |
| Working memory only | 20-turn sliding window | 短期一致但长期 drift |
| + Episodic memory | 每 10 轮压缩 | 中等改善 |
| + Semantic memory | personality belief update | 显著改善 |
| + RL reward | 加入 fidelity reward | 最优（预期） |
| greed-mem | 组内方法 | 与上述对比 |

### Phase 4: Personal Prediction Accuracy（Week 4-6）

**最终指标：经过 RL training 后，persona prediction 是否更准？**

```
Evaluation Protocol:

1. 给 agent 一个 persona profile P_target
2. 让 agent 与 user 对话 N 轮
3. 用 SocialScope Inference Layer 推断 P_inferred
4. 计算 Personal Prediction Accuracy:

   Accuracy = cosine_similarity(P_target, P_inferred)

   分维度：
   - Big Five accuracy (5 dims)
   - MBTI accuracy (4 axes)
   - Attachment accuracy (3 dims)
   - Overall accuracy (22 dims)

对比：
   Baseline accuracy (no RL)  vs  Personal prediction accuracy (with RL)
```

---

## Timeline（并行甘特图）

```
Week    1       2       3       4       5       6       7       8
        Mar27   Apr3    Apr10   Apr17   Apr24   May1    May8    May15

Track A (Demo Paper):
Emotion ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Baseline ▓▓▓▓▓▓▓▓
Submit           ██

Track B (Research Paper):
Phase 1  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Baseline  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
Accuracy

Phase 2          ████████████████████████░░░░░░░░░░░░░░░░░░
RL Env            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Phase 3                  ████████████████████████░░░░░░░░░░
Memory +                  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
Cont.Learn
(greed-mem)

Phase 4                                  ████████████████████
Personal                                  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
Prediction

Writing                                          ████████████
```

---

## 每周具体 TODO

### Week 1 (Mar 27 - Apr 2)
- [ ] **[Demo]** 选定 emotion baseline 方法，准备 evaluation data
- [ ] **[Demo]** 跑 B1-B4 baseline，记录 accuracy/F1
- [ ] **[Research]** 设计 baseline accuracy 实验：选定 seen vs unseen roles
- [ ] **[Research]** 用现有 `SyntheticDialogueGenerator` 生成 10/30/100 轮对话

### Week 2 (Apr 3 - Apr 9)
- [ ] **[Demo]** 写 emotion baseline 结果到 demo paper
- [ ] **[Research]** 完成 baseline accuracy 实验（all conditions）
- [ ] **[Research]** 分析 pre-training exposure + SFT 对 drift 的影响
- [ ] **[Research]** 开始 RL environment 设计（state/action/reward 定义）

### Week 3 (Apr 10 - Apr 16)
- [ ] **[Research]** 实现 `PersonaConversationEnv`（Gym-compatible）
- [ ] **[Research]** 实现 reward function（fidelity + explore + natural + safety）
- [ ] **[Research]** 设计 exploration bonus（curiosity-driven for unseen features）
- [ ] **[Research]** 连接 greed-mem 的 memory framework

### Week 4 (Apr 17 - Apr 23)
- [ ] **[Research]** RL training loop 实现（PPO or DPO on conversation）
- [ ] **[Research]** Continuous learning 实验：6 条件对比
- [ ] **[Research]** Memory ablation study

### Week 5 (Apr 24 - Apr 30)
- [ ] **[Research]** Personal prediction accuracy 全量评估
- [ ] **[Research]** Cross-model comparison（Claude/GPT/Gemini/Qwen/DeepSeek）
- [ ] **[Research]** 统计显著性检验

### Week 6-7 (May 1 - May 14)
- [ ] **[Research]** 写论文：Introduction, Method, Experiments, Results
- [ ] **[Demo]** Demo paper 最终版提交（5月）
- [ ] **[Research]** Related work + Discussion + Limitations

### Week 8 (May 15+)
- [ ] **[Research]** 内部 review + revision
- [ ] **[Research]** 准备 EMNLP 投稿

---

## 关键技术路线图

### 1. Persona 保持方式（从弱到强）

| 方法 | 强度 | 适用场景 |
|------|------|----------|
| System prompt only | ⭐ | 最基础，8-12 轮后 drift |
| + Few-shot examples | ⭐⭐ | 改善但不根本解决 |
| + Memory (working) | ⭐⭐⭐ | 短期一致性 |
| + Memory (episodic + semantic) | ⭐⭐⭐⭐ | 长期一致性（greed-mem 方向） |
| + RL fidelity reward | ⭐⭐⭐⭐⭐ | 主动保持 + 探索（核心 contribution） |

### 2. Pre-training 影响分析

```
Seen roles (pre-training 中见过):
  优势: 模型有 prior knowledge → 初始 fidelity 高
  劣势: 模型可能有自己的"理解" → 不完全遵循 assigned profile
  假设: SFT 可以 override pre-training bias

Unseen roles (pre-training 中没见过):
  优势: 无 prior bias → 完全遵循 assigned profile（初始）
  劣势: 无 knowledge support → 长期维持更难
  假设: Memory + RL 对 unseen roles 帮助更大
```

### 3. Round-level Feature Dynamics

```
Turn 1-5:   基础特征出现 (communication_style, extraversion)
Turn 5-15:  中层特征出现 (openness, agreeableness, values)
Turn 15-30: 深层特征出现 (neuroticism, attachment_style)
Turn 30+:   少见特征 (trust_trajectory, love_language)
Never:      某些特征可能整个对话都不自然出现

→ Exploration reward 鼓励 agent 在适当时机引导话题
→ 让「从未出现」的特征有机会被观测
→ 同时保持对话自然度（不能强行转话题）
```

---

## Paper Structure 预览

```
Title: Conversation as Environment:
       Reinforcement Learning for Persona-Consistent Multi-Turn Dialogue

1. Introduction
   - Persona drift 问题
   - 现有方法局限（prompting, SFT）
   - 我们的方法：RL agent-environment framework

2. Related Work
   - Static persona assignment (Character-LLM, RoleLLM)
   - Persona drift measurement (Li et al., Choi et al.)
   - Sycophancy & accommodation (Sharma et al., Cheng et al.)
   - RL for dialogue (新方向)
   - Continuous learning & memory (greed-mem 连接)

3. Framework
   3.1 Three-Layer Persona Model (Profile / Behavioral / Inference)
   3.2 Conversation as RL Environment
   3.3 Reward Design: Fidelity + Exploration + Naturalness
   3.4 Memory-Augmented Continuous Learning

4. Experiments
   4.1 Baseline Accuracy (RQ1: drift without intervention)
   4.2 User Accommodation Analysis (RQ2: user style effect)
   4.3 RL-based Persona Maintenance (RQ3: RL vs baselines)
   4.4 Ablation Studies
       - Memory layers ablation
       - Reward components ablation
       - Pre-training exposure effect
       - SFT impact analysis

5. Results
   5.1 Baseline accuracy tables
   5.2 Personal prediction accuracy (RL vs no-RL)
   5.3 Cross-model comparison
   5.4 Continuous learning curves

6. Discussion
   - Why RL works: exploration of unseen features
   - Pre-training as double-edged sword
   - Implications for AI safety & persona design

7. Conclusion + Future Work
```

---

*This roadmap connects: demo paper (emotion baseline), research paper (RL environment), greed-mem (memory/continuous learning), and the SoulMatch system.*
