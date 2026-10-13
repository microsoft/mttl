# 补充实验清单：Clinical LoRA Adapter Leakage 论文（ACL 投稿）

> 优先级：**P0** = 不做基本无法投稿；**P1** = 决定能否进主会；**P2** = 加分项
> 建议执行顺序：E6 → E9 → E1 → E2 → E5 → E7 → E3 → E4 → P1 → P2
> 原则：做不完的实验，对应声明要从论文中删除，不要保留未兑现的内容。

---

## 总览

| # | 实验 | 优先级 | 解决的审稿问题 | 对应章节 |
|---|---|---|---|---|
| E6 | 修复 / 验证 Table 4 数字 | P0 | 数字自相矛盾、疑似 bug | §8.4, Table 4 |
| E9 | MIA 评估去混淆（canary 分布偏移） | P0 | DP ε=1 AUC<0.5 的异常 | §7, §8.1 |
| E1 | 现实训练区间 + 攻击对比 | P0 | loss 基线 AUC=1.0，所提攻击更弱 | §8.1 |
| E2 | 统计功效与置信区间 | P0 | TPR@0.1%FPR 分辨率不足 | 全部表格 |
| E5 | 全量微调 / 其他 PEFT 对照 | P0 | 标题问题无法回答 | §8（新增 RQ） |
| E7 | 正规 DP-LoRA 基线 | P0 | DP 实现不正确、基线过弱 | §6 D1, §8.4 |
| E3 | 真实临床数据 | P0 | 只有模板合成数据 | §7 |
| E4 | 多 backbone | P0 | 单模型结论难以泛化 | §7, §8 |
| E8 | A4 反演：实现或降级 | P0 | 正文描述与实现不符 | §5 A4, 附录 D |
| E10 | SGD vs AdamW + 子空间攻击修正 | P1 | 理论与实验脱节 | §4.1, §5 A2, 附录 G |
| E11 | AdaptGuard 消融 + 自适应攻击 | P1 | 防御价值未证明 | §6 D5, §8.4 |
| E12 | Canary exposure 规范化 | P1 | 256 候选空间过小 | §5 A3, §8.3 |
| E13 | 下游任务效用 | P1 | inverse NLL 非标准 | §7, 全部效用列 |
| E14 | A5 属性 / 性质推断 | P1 | 描述了但无结果 | §5 A5 |
| E15 | 目标模块与 PEFT 变体 | P2 | "target modules" 声明无结果 | §8.5 |
| E16 | 审计公开临床 adapter | P2 | 现实冲击力 + 兑现伦理承诺 | §9, Ethics |
| E17 | Prop 3 秩 1 记忆化演示 | P2 | 引言声明无支撑 | §4.2, 附录 C |

---

## P0：必须完成

### E6. 修复 / 验证 Table 4
**问题**
- "None" 行把 3 epoch 的 AUC（0.773）和 10 epoch 的 PHI recall（50%）混在一起。
- D2（r′=2）效用 0.047，NLL≈21，比 base model 还差，疑似 bug。
- DP ε=8 与 ε=1 的效用完全相同（0.330），很可能两者都退化成了 base model。

**做法**
- [ ] 所有行统一到同一配置（建议 r=16、|D|=10³、固定 epoch 或 early stopping），重新测 AUC、TPR 和 PHI recall。
- [ ] 新增 **Base model（无 adapter）** 行，作为效用下界和 AUC=0.5 的参照。
- [ ] 排查 D2：截断后保存的 `lora_alpha` 和 `r` 是否匹配。SVD 截断后应令 B′=U√S、A′=√S Vᵀ，并将缩放 α/r′ 设为 1，或者把缩放吸收进因子。
- [ ] 如果需要，D2 同时报告 r′∈{2,4,8}，画出截断曲线。

**产出**：修正后的 Table 4 + 一张隐私–效用散点图（x=效用，y=AUC 或 TPR@1%FPR）。

---

### E9. MIA 评估去混淆
**问题**：canary 如果只插入成员，含 canary 的 note 本身就更"罕见"，成员与非成员之间会出现分布偏移。DP ε=1 下 AUC 0.433 显著低于 0.5，是典型信号。

**做法**
- [ ] 选一种方案：(a) MIA 评估集排除所有含 canary 的 note；(b) 在非成员中对称地插入同分布的新 canary（不同的标识值）。
- [ ] 做一个 "blind baseline" 检查：只用不依赖模型的特征（长度、token 稀有度、zlib 熵）区分成员与非成员，AUC 应该约为 0.5（参考 Duan et al., 2024）。
- [ ] 用修正后的评估集重跑 Table 1 / 2 / 4。

**产出**：附录中的去混淆检查表（blind baseline AUC 及其 CI）。

---

### E1. 现实训练区间 + 攻击全面对比
**问题**：loss 和 Min-K% 已达到 AUC=1.0，A1 只有 0.773。需要找到 loss 基线不饱和的现实区间，并在那里证明 A1 或 fusion 的价值。

**设置**
- [ ] 训练时长：按验证集 loss 做 early stopping；1 epoch；3 epoch；10 epoch。另外每 N 步保存一次 checkpoint，画曲线。
- [ ] 数据量：|D| ∈ {10², 10³, 10⁴}。
- [ ] 任务格式（关键）：除了在 note 上做因果 LM 训练，再加一个**指令任务**（note→summary 或 note→QA），**只在输出上计算 loss**。这是真实临床 adapter 的常见形态。
- [ ] 攻击集合：Loss（Yeom）、Zlib ratio、Min-K%、Min-K%++、Reference（base model，即 A1）、SPV-MIA、LoRA-Leak 的参考攻击、A1+A2 fusion。

**指标**：AUC、TPR@1%FPR、TPR@0.1%FPR（需要 E2 的样本量），全部附 95% CI。

**预期 / 叙事**
- 画 "MIA AUC 随训练步数变化" 的曲线，并标出 early-stopping 点。
- 在非饱和区间比较 A1 与 loss；在过拟合区间解释 A1 为何变弱（成员 loss→0 后，base loss 的方差占主导）。
- 如果 A1 在任何区间都不占优，就不要把 A1 作为方法贡献，改为"评估协议"的一部分。

**产出**：新 Figure（AUC/TPR 与训练步数的关系，各攻击一条线）+ 重写后的 Table 1（early stopping 和过拟合两列）。

---

### E2. 统计功效与置信区间
**问题**：200 个非成员时，最小非零 FPR 为 0.5%；A2 只有 80 个样本。AUC 的 95% CI 约为 ±0.045，rank 之间的差异都落在噪声范围内。

**做法**
- [ ] 非成员 ≥ 5000 个（至少 1000 个才能报 1% FPR；要报 0.1% FPR 需要 ≥ 5000–10000 个）。成员数视 |D| 而定，|D|=10³ 时用全部成员。
- [ ] 3 个训练 seed（不同的数据顺序和 LoRA 初始化）。
- [ ] 1000 次 bootstrap 得到 95% CI，报告 mean ± CI。
- [ ] 画 log-log ROC 曲线（参考 Carlini et al., 2022）。
- [ ] rank 结论改用**等价性检验**（TOST，margin=0.05），而不是只说"差异很小"。

**产出**：所有主表附 CI；附录加一张 log-log ROC 图。

---

### E5. 全量微调与其他 PEFT 对照
**问题**：标题问"LoRA adapter 是否保护隐私"，必须有参照物。Mireshghallah et al. (2022) 与 Malekmohammadi & Farnadi (2024) 给出了相反的结论（adapter 或 LoRA 泄露更少），需要正面回应。

**设置**
- [ ] 方法：Full FT、LoRA（r=1/16/64）、IA3、Prefix/Prompt tuning（可选）、Head-only（可选）。
- [ ] **在相同效用下比较**：调节 lr 和 epoch，使各方法的验证效用对齐（例如都在某个 NLL 或任务指标附近），再比较 MIA 和抽取结果。
- [ ] 同时报告"同 epoch"和"同效用"两种对比。

**产出**：新 RQ 小节 + 表格（方法 × {效用, AUC, TPR@1%FPR, PHI recall, exposure}）。

---

### E7. 正规 DP-LoRA 基线
**问题**：目前是"per-step grad clip 1.0 + batch 1 × accum 8"，不是 DP-SGD；batch 8 下 DP 效用必然很差，与文献（Yu et al., 2022；Li et al., 2022）不符。

**做法**
- [ ] 使用 Opacus（或 dp-transformers）：逐样本梯度裁剪 + Poisson 采样 + PRV/RDP accountant。
- [ ] 调参：batch ∈ {256, 512, 1024}、lr、epoch、裁剪阈值 C。在 ε=8 下先把效用调到最优，再扫 ε ∈ {1, 3, 8, ∞}，δ = 1/n^1.1。
- [ ] 报告 canary 的 **group privacy**：重复 k 次的 canary 对应的实际保证约为 kε，需在文中讨论。
- [ ] 附录给出完整的 DP 超参数和 accountant 设置。

**产出**：修正后的 D1 行 + DP 的 ε–效用曲线。

---

### E3. 真实临床数据
**问题**：只有模板生成的合成 note，而且 D_pub 与训练数据同源，对攻击者和 AdaptGuard 都极为有利。

**选项（至少选一个，推荐 a）**
- [ ] (a) **MIMIC-IV-Note**：PhysioNet credentialed 访问；只攻击自己训练的 adapter，不发布 adapter；在 note 中插入 surrogate PHI 作为 canary。
- [ ] (b) **n2c2 / i2b2 2014 de-identification** 数据：已有 PHI 标注，可以用 surrogate 替换后评估 PHI recall。
- [ ] (c) 非临床对照域：例如带 PII 的 Enron 邮件，用来证明结论不是临床模板特有的。
- [ ] D_pub 改为**分布不同但同领域**的公开数据（例如 MTSamples、PubMed case reports），不再来自同一生成器。

**注意**：更新 Ethics 部分（DUA 合规、不发布基于真实数据训练的 adapter、展示的样例仍只用 surrogate canary）。

---

### E4. 多 backbone
- [ ] 至少 3 个 7–9B 模型：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct、Mistral-7B-Instruct 或 Gemma-2-9B-it。
- [ ] 规模轴：Llama-3.2-1B / 3B，观察泄露与模型规模的关系。
- [ ] 可选：一个医学领域模型（如 Meditron、BioMistral）作为 base。
- [ ] 核心结论（rank 无效、训练时长主导、防御排序）需要在所有 backbone 上复现。

---

### E8. A4 反演：实现或降级
**问题**：附录 D 写明报告的数字"没有 HotFlip 搜索，只用 template+greedy"，但正文用 Eq. 5 描述了一个优化算法。

**方案 A（实现）**
- [ ] 实现 Eq. 5：在松弛的 embedding simplex 上优化，用 GCG/HotFlip 投影回 token，从 D_pub 模板热启动。
- [ ] 消融：仅贪心 vs 仅子空间项 vs 仅 LM 项 vs 完整目标函数。
- [ ] 指标：PHI recall、exact span recovery、normalized edit distance（换成对 PHI 片段敏感的指标）。

**方案 B（降级，时间不够时选这个）**
- [ ] 把 A4 如实改名为 "greedy identifier extraction"，Eq. 5 移到 Future Work，并说明抽取数字是下界。

---

## P1：强烈建议

### E10. SGD vs AdamW + 子空间攻击修正
**目的**：把 Prop 1 从"空头主张"变成"解释性发现"。

- [ ] 用 SGD（含 momentum）和 AdamW 分别训练（调到相近效用），比较 A2 的 AUC。
- [ ] **A2 用 A−A(0)**：在 K5（seed 公开）设定下复现初始化 A(0)，投影到 rowsp(A−A(0))。
- [ ] **B 侧投影攻击（新）**：对候选 x 计算输出梯度 δ(x)，度量它在 colsp(B) 上的投影能量。B 从 0 初始化，完全由数据决定。
- [ ] 报告 frozen-A（D3）在 SGD 与 AdamW 下各自的效果。

**产出**：Table（优化器 × {A2 用 A, A2 用 A−A0, B 侧, fusion}）；Limitations 据此改写。

---

### E11. AdaptGuard 消融与自适应攻击
- [ ] **关键基线：只用 D_pub 训练的 adapter**（没有 teacher）。如果效用相近，AdaptGuard 就没有意义。
- [ ] 无噪声蒸馏（去掉 top-k 和 logit noise），用来隔离 privatization 的作用。
- [ ] 扫描噪声强度和 top-k 值，画出隐私–效用曲线。
- [ ] 扫描查询预算（D_pub 大小：100 / 300 / 1000 / 3000）。
- [ ] PATE 式分片聚合变体（如果论文中提到了，就必须有结果）。
- [ ] **自适应攻击**：攻击者知道防御机制，用 PHI 触发式 prompt（例如 "Patient name:"、"MRN:"）构造探测输入，测试 teacher 通道是否把 canary 泄露给 ΔW′。
- [ ] 表格中 "Guar." 列改为 "—"，不要写 "partial"。

---

### E12. Canary exposure 规范化
- [ ] 使用大候选空间：例如 7 位 MRN（10^7）、日期空间、姓名组合，按 Carlini et al. (2019) 的方法用采样估计 rank 或外推。
- [ ] 按重复次数 {1, 2, 4, 8, 16} 分别画 exposure 曲线，每个 rank 一条线。
- [ ] 重复记录放大：同一份完整 note 重复 {1, 2, 4, 8} 次，看 MIA 和抽取结果如何变化。
- [ ] 明确 Table 2 "Expo." 列对应的重复次数。

---

### E13. 下游任务效用
- [ ] 用真实任务指标替换 inverse NLL：摘要任务用 ROUGE-L + 事实性指标（如 MEDCON / NER-F1），QA 任务用 MedQA / PubMedQA 准确率。
- [ ] 所有表格的效用列统一使用该指标，NLL 放到附录。

---

### E14. A5 属性 / 性质推断
- [ ] 在中性临床 prompt 下采样 N 条生成，估计年龄分布、性别比例、诊断患病率、机构模板特征，与真实训练集比较（例如 TV 距离或 KL）。
- [ ] 对照组：base model 的生成，以及在另一个队列上训练的 adapter 的生成。
- [ ] 如果不做，就删除 A5 以及目标 G3/G4。

---

## P2：加分项

### E15. 目标模块与 PEFT 变体
- [ ] 目标模块：attention-only vs MLP-only（up/gate/down）vs all-linear。LoRA-Leak 报告 up/gate 层泄露最多，需要对照。
- [ ] 变体：DoRA、QLoRA（4-bit base）、rsLoRA。

### E16. 审计公开临床 adapter
- [ ] 在 HF 上筛选公开的临床或医学 LoRA adapter，做性质推断和抽取探测（无法验证成员身份，只报告抽取到的疑似 PHI 模式和队列性质）。
- [ ] 走伦理审查；发现高风险 adapter 时预先通知维护者（兑现 Ethics 部分中的承诺）；论文中不公开可识别的具体输出。

### E17. Prop 3 秩 1 记忆化演示
- [ ] 在单层上训练 r=1 adapter，使其逐字输出一条完整的**训练** note，报告成功所需的最短前缀和步数。
- [ ] 或者给出构造性证明（a 作为前缀检测器，b 作为残差流中的引导方向）。
- [ ] 两者都不做的话：把引言中 "we exhibit a rank-1 adapter that emits a complete held-out note" 删掉，Prop 3 降级为 conjecture。

---

## 附：每个实验都要记录的元信息

- backbone、LoRA 配置（r、α、目标模块、dropout）、优化器、lr、batch、epoch 或步数、seed
- 成员数和非成员数、候选 canary 空间大小
- 所有指标的 95% CI（1000 次 bootstrap）
- GPU 小时数（用于 Limitations 或复现说明）
- 删除所有内部路径（如 `projects/privacy_lora`），代码链接改为匿名仓库
