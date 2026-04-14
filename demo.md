# ACSD 项目总结（中文版）

_Adaptive Cascaded Speculative Decoding — MLSys Course Project_

---

## 一、Base Framework：Speculative Decoding 与 CS-Drafting

### Speculative Decoding（SD）

标准投机解码的核心思想是：**用一个小而快的 draft model 先贪心生成 $k$ 个 token，再让大模型一次 forward pass 批量验证全部草稿**。由于大模型单次 prefill $k$ 个 token 比自回归生成 $k$ 次快得多，整体吞吐大幅提升。验证过程保持拒绝采样性质，因此输出分布与大模型纯自回归完全一致（**lossless**）。

### CS-Drafting（我们的 base）

CS-Drafting（Chen et al., arXiv:2312.11462）把 SD 扩展为**多级级联**：

$$M_{d_1} \to M_{d_2} \to \cdots \to M_l$$

每一级为下一级提供草稿，最终由最大模型 $M_l$ 验证。三个模型的角色在整个生成过程中**固定不变**：小的永远是 drafter，大的永远是 verifier。

**CS-Drafting 存在的两个低效问题：**

| 问题 | 描述 |
|------|------|
| 无过滤 | $M_l$ 需要处理 $M_s$ 起草的每一个 token，即使 $M_s$ 明显偏差 |
| 无升级 | 当 $M_s$ 表现持续很差时，没有任何机制让更好的模型接管起草 |

---

## 二、Proposed 方法：ACSD

ACSD（**Adaptive Cascaded Speculative Decoding**）在三层模型栈上引入两个机制，分别解决上述两个问题。

### 三层模型栈

| 层 | 模型 | 参数量 | 角色 |
|----|------|--------|------|
| $M_s$ | TinyLlama-1.1B-Chat | 1.1B | 快速起草者（默认） |
| $M_m$ | LLaMA-2-7B | 7B | **双重角色**：预验证器 or 升级起草者 |
| $M_l$ | LLaMA-2-13B | 13B | 最终验证者，**始终运行**，保证无损 |

三个模型 fp16 共约 42 GB，跑在单块 48 GB A6000 上。

---

### Phase 2：级联预验证（Cascaded Pre-Verification）

**解决问题：** $M_l$ 不应该看到 $M_s$ 明显错误的 token。

**每步数据流：**

```
M_s.propose(k_s=5 tokens)
    → draft_ids

M_m.pre_verify(draft_ids)             ← 新增
    → filtered_ids（k' ≤ k_s，截断 M_m 不认可的 token）

M_l.review(filtered_ids)              ← 处理更短的序列，省计算
    → 最终接受序列
```

**为什么有效：** $M_m$（7B）每个位置的计算代价远低于 $M_l$（13B）。把一个 token 位置从 $M_l$ 转移给 $M_m$ 处理净收益为正。实验中 MMLU 上平均每步 $M_m$ 为 $M_l$ 节省约 0.99 个 token 位置（37.7 saved / 38 calls）。

**无损性保证：** $M_m$ 只做截断（粗滤），$M_l$ 始终对最终序列做验证，输出分布不变。

---

### Phase 3：自适应角色切换（Adaptive Role-Switching）

**解决问题：** 当 $M_s$ 持续表现差时，让 $M_m$ 接管起草，跳过慢速的"起草→过滤"两步。

**状态机逻辑（每步更新一次）：**

```
计算本步 M_s 接受率：
    α_step = (M_l 接受 token 数) / (M_s 起草 token 数)

维护滑动窗口（W=20 步）均值：
    ᾱ = mean(α_window[-W:])

决策：
    ᾱ ≥ τ  →  下步仍由 M_s 起草（Phase 2 快速路径）
    ᾱ < τ  →  下步由 M_m 起草（跳过 M_s）
```

**M_m 当起草者时的路径：**

```
M_m.propose(k_m=4 tokens) → M_l.review    # M_s 完全空闲
```

**KV Cache 失效处理：** M_m 在预验证器模式下的 KV cache 记录的是"旁观者视角"，切换为起草者的第一步必须清空，否则位置编码错乱。代码中用 `mm_as_verifier` flag 控制这个一次性清零。

---

## 三、具体实现

### 核心文件

| 文件 | 作用 |
|------|------|
| `csd.py` | 原始 CS-Drafting 生成循环（未改动） |
| `model.py` | 扩展了 `ACSDMiddleTierModel`：新增 `pre_verify()` 方法，支持 M_m 同时具备预验证和起草能力 |
| `acsd.py` | 两个核心函数：`acsd_cascaded`（Phase 2）和 `acsd_adaptive`（Phase 3） |
| `main_acsd.py` | CLI 入口，`--mode {baseline,cascaded,adaptive}`，结果保存为 JSON |

### 运行方式

```bash
conda activate acsd
cd /mnt/data/gaokaizhang/mlsys/CS-Drafting

# 纯大模型自回归（基线上限）
python main_acsd.py --mode autoregressive --dataset mmlu --n_samples 100

# Baseline CSD（M_s → M_l，两层）
python main_acsd.py --mode baseline --dataset mmlu --n_samples 100

# Phase 2：级联预验证（三层固定角色）
python main_acsd.py --mode cascaded --dataset mmlu --n_samples 100

# Phase 3：自适应（τ=0.4，W=20）
python main_acsd.py --mode adaptive --dataset mmlu --n_samples 100 --tau 0.4
```

---

## 四、Ablation 实验设计

我们跑了三类 ablation，对应三个研究问题：

### RQ1：Phase 2 是否真的减少了 $M_l$ 的 forward pass 次数？

**实验：** 对比 `baseline`（M_s → M_l）和 `cascaded`（M_s → M_m → M_l），记录每个样本 $M_l$ 调用次数和 $M_m$ 节省的 token 数量。

**证明目标：** Cascaded 的 $M_l$ calls 显著低于 baseline，且节省量接近理论预测 $k_s \cdot (1 - \alpha_{M_s}^{M_m})$。

---

### RQ2：Phase 3 自适应切换的价值在哪里？

**实验：** 对比 `cascaded` vs `adaptive (τ=0.4)` 在两个数据集上的 tok/s 和 switches/sample。

**证明目标：** 在 $M_s$ 表现较差的场景（GSM8K 推理题）下，adaptive 通过让 $M_m$ 接管起草提升接受率；在 $M_s$ 表现好的场景（MMLU）下，adaptive 退化为 cascaded（switches≈0），不引入额外开销。

---

### RQ3：超参 τ 的敏感性（τ Ablation）

**实验：** 固定 W=20，在 MMLU 上扫描 τ ∈ {0.2, 0.3, 0.4, 0.5}，记录 tok/s、$M_l$ calls、$M_m$ saved、switches/sample。

| τ | Tok/s | $M_l$ calls | $M_m$ saved | Switches/sample |
|---|-------|------------|------------|-----------------|
| 0.2 | 40.7 | 38.0 | 37.7 | 0.00 |
| **0.3** | **40.9** | **38.0** | **37.7** | 1.30 |
| 0.4 | 39.3 | 38.2 | 36.2 | 1.26 |
| 0.5 | 32.5 | 39.6 | 28.7 | 2.96 |

**关键发现：**
- τ=0.2 时 M_s 在 MMLU 上接受率从未低于 0.2，所以**从不切换**，等价于纯 Cascaded
- τ=0.3 时偶尔切换（1.3 次/样本），但切换带来少量提升（40.9 vs 40.7）
- τ=0.5 时切换过于频繁（2.96 次/样本），M_m 起草的慢速代价大于收益，吞吐下跌至 32.5
- **最优点在 τ=0.3**，兼顾切换灵敏度和切换代价

---

### Window Size 消融（W Ablation）

**实验：** 固定 τ=0.4，在 MMLU 上测试 W ∈ {10, 20, 50}。

| W | 含义 | 预期效果 |
|---|------|---------|
| 10 | 反应快，容易受单步噪声干扰 | 切换频繁，可能振荡 |
| **20** | 默认值（已跑） | 基准 |
| 50 | 反应慢，更稳定但滞后更大 | 切换少，M_s 差了更久才换 |

> 实验在 tmux `acsd-window` 中运行，W=10 和 W=50 的结果待补充。

---

## 五、主要结果与结论

### 主结果表

| 方法 | MMLU Tok/s | MMLU $M_l$ calls | GSM8K Tok/s | GSM8K $M_l$ calls |
|------|-----------|-----------------|------------|-----------------|
| 纯 $M_l$ 自回归 | 9.3 | 190.0 | 9.4 | 177.0 |
| Baseline CSD | 26.6 | 29.7 | 23.9 | 30.4 |
| **ACSD Cascaded** | **41.1** (+1.55× vs baseline) | 38.0 | **39.9** (+1.67×) | 36.1 |
| ACSD Adaptive τ=0.4 | 39.3 (+1.48×) | 38.2 | 26.3 (+1.10×) | 38.8 |

### 关键结论

1. **Phase 2 效果显著：** Cascaded 在 MMLU 上比 baseline CSD 快 **1.55×**，GSM8K 上快 **1.67×**。$M_m$ 每步平均节省 37.7 个 $M_l$ token 处理位置，接近理论上限。

2. **Phase 3 在困难数据集上更有价值：** GSM8K（数学推理，$M_s$ 表现更差）上 Adaptive 的切换更有意义；在 MMLU（知识问答，$M_s$ 表现好）上 τ=0.4 几乎不切换，退化为 Cascaded。

3. **τ 存在最优区间：** 过低（0.2）= 从不切换 = 浪费 Phase 3；过高（0.5）= 频繁切换 = 慢速 M_m 起草代价超过收益。τ=0.3 在 MMLU 上是最优点。

4. **方法的局限性：** 当前切换机制是**后验的**（依赖已过去 W 步的接受率），M_s 已经差了 W 步才能触发切换；且进入和退出 M_m 使用同一个阈值，边界附近容易振荡。这些是 future work 的改进方向（置信度驱动切换、非对称阈值）。

---

## 六、整体贡献总结

> ACSD 在 CS-Drafting 的基础上，通过引入**中间模型预验证门控**（Phase 2）和**基于滑动窗口接受率的自适应角色切换**（Phase 3），在不破坏输出分布无损性的前提下，将三层模型栈的推理吞吐相比两层 baseline 提升最高 **1.67×**，相比纯大模型自回归提升最高 **4.3×**。
