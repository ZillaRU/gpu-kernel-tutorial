# GPU Kernel 开发教程：总目录与学习路径

> 这页把整套教程串起来：先告诉你应该按什么顺序读，再告诉你每章读完后到底应该得到什么。

---

## 1. 这套教程适合谁

这套教程主要适合三类读者：

1. **刚开始学 Triton / CUDA / HIP kernel 的工程师**：需要先建立统一心智模型；
2. **已经能写 kernel，但调参还像碰运气的人**：需要形成“诊断 → 优化 → 验证”的闭环；
3. **想把 kernel 技能整理成 AI/工程团队可复用知识的人**：需要模块化、可导航、可复盘的结构。

如果你现在的状态是：

```text
我大概知道 GEMM / softmax / attention 是什么，
但一到真实优化时就不知道该先看哪里、先改什么，
```

那么这份学习路径就是为你设计的。

---

## 2. 全部文件总览

| 顺序 | 文件 | 主题 | 读完你应该得到什么 |
|---|---|---|---|
| 1 | [chapter1-tier1-tier2.md](./chapter1-tier1-tier2.md) | 执行模型、分块、内存层级 | 知道 GPU 在怎么执行、数据在怎么流动 |
| 2 | [chapter2-tier3-tier4.md](./chapter2-tier3-tier4.md) | 计算优化、融合、高级调度 | 知道什么时候该 fusion，什么时候该改调度 |
| 3 | [chapter3-tier5-arch.md](./chapter3-tier5-arch.md) | 架构特定优化 | 知道为什么同一 kernel 在不同 GPU 上最优解不同 |
| 4 | [chapter4-kernel-skills.md](./chapter4-kernel-skills.md) | 具体算子技能 | 能把前 3 章方法落到 GEMM / FA / Softmax / RMSNorm 等算子 |
| 5 | [chapter5-system-skills.md](./chapter5-system-skills.md) | 系统工作流 | 能用 profile → diagnose → optimize → verify → benchmark 的流程做优化 |

### 配套辅助文档

| 文件 | 用途 |
|---|---|
| [最小可运行模板集.md](./最小可运行模板集.md) | 为 5 章各补一个可作为起点的最小模板 |
| [术语表与通用坑清单.md](./术语表与通用坑清单.md) | 统一全书术语口径，并提供 code review / 自查 checklist |
| [章节依赖图与读者画像建议.md](./章节依赖图与读者画像建议.md) | 根据读者背景选择最省时间的阅读路线 |

### 图解索引（适合复习或断网速查）

如果你想先理解整套图谱的颜色和箭头分别代表什么，建议先看 [统一图例](./最小可运行模板集.md#0-统一图例先看颜色语义再翻图)。

| 图解主题 | 位置 | 适合解决什么困惑 |
|---|---|---|
| 统一图例：颜色与箭头语义 | [模板集附录 0](./最小可运行模板集.md#0-统一图例先看颜色语义再翻图) | 解决“蓝/绿/橙/紫/红在图里分别代表什么” |
| Tier 3 / Tier 4 决策图 | [第二章开头](./chapter2-tier3-tier4.md#1-先分清三个常见优化方向) | 解决“该继续 fusion 还是该重构调度” |
| bank conflict 与 padding 图 | [第三章第二节](./chapter3-tier5-arch.md#2-先讲一个所有平台都绕不过去的问题bank-conflict) | 解决“为什么转置常常要加 padding” |
| Tier 5 转译图 | [第三章第五节](./chapter3-tier5-arch.md#52-对教程作者来说最重要的写法规范) | 解决“如何把架构事实翻译成调参与路径选择” |
| GEMM：整张矩阵如何切成输出 tile | [第四章 2.2](./chapter4-kernel-skills.md#22-整张矩阵怎么被切块) | 解决“一个 program 到底负责哪块 C” |
| GEMM：一个 tile 怎么沿 K 维累加 | [第四章 2.3](./chapter4-kernel-skills.md#23-一个-tile-怎么沿-k-维算出来) | 解决“为什么 `acc` 要循环累加” |
| GEMM：单次 Triton 循环走读 | [第四章 2.7](./chapter4-kernel-skills.md#27-把一个-triton-gemm-循环完整走读一遍) | 解决“`a_ptrs / b_ptrs / tl.dot` 到底在做什么” |
| GEMM：地址表达图 | [第四章 2.8](./chapter4-kernel-skills.md#28-从-offs-到-ptrs地址表达到底在映射什么) | 解决“stride 为什么不能写反” |
| Flash-Attention：结构图 | [第四章 3.3](./chapter4-kernel-skills.md#33-先从结构图看flash-attention-在循环里到底做了什么) | 解决“为什么它不需要写出完整 `scores/P`” |
| Flash-Attention：在线 softmax 状态图 | [第四章 3.5](./chapter4-kernel-skills.md#35-图解为什么-acc-和-li-都要乘-alpha) | 解决“为什么 `acc` 和 `l_i` 都要重标定” |
| Softmax：一行归一化图 | [第四章 4.3](./chapter4-kernel-skills.md#43-图解一行-softmax-到底经历了什么) | 解决“softmax 为什么常常是 memory-bound” |
| RMSNorm：融合路径图 | [第四章 5.3](./chapter4-kernel-skills.md#53-图解为什么-rmsnorm-是很好的-fused-kernel-练手对象) | 解决“RMSNorm 为什么适合练 reduce + fusion” |
| Cross-Entropy：chunked logsumexp 图 | [第四章 6.3](./chapter4-kernel-skills.md#63-图解大-vocab-时真正发生的是-chunked-logsumexp) | 解决“大 vocab 时为什么不该先物化完整 softmax” |
| RoPE：pairing layout 图 | [第四章 7.3](./chapter4-kernel-skills.md#73-图解interleaved-和-split-half-到底差在哪) | 解决“RoPE 数学对了但 layout 仍可能错” |
| MoE：路由与数据流图 | [第四章 8.3](./chapter4-kernel-skills.md#83-图解moe-的性能损失可能发生在哪条数据流上) | 解决“为什么 MoE 不能只盯 expert GEMM” |
| FFN：训练 vs decode 图 | [第四章 9.2](./chapter4-kernel-skills.md#92-图解训练-ffn-和-decode-ffn-为什么像两种题) | 解决“为什么 FFN 不能用一套优化优先级通吃所有阶段” |
| 系统技能：优化闭环图 | [第五章开头](./chapter5-system-skills.md) | 解决“为什么优化不是线性打卡流程” |

如果你只是想离线快速翻图，而不想逐章跳转，也可以直接看 [最小可运行模板集里的图解速查附录](./最小可运行模板集.md#图解速查附录离线版)。

---

## 3. 推荐阅读顺序

### 路线 A：第一次系统学习（推荐）

严格按 1 → 2 → 3 → 4 → 5 阅读。

原因是这套依赖关系最自然：

```text
执行模型与访存基础
    ↓
计算优化与调度重构
    ↓
平台差异
    ↓
算子落地
    ↓
系统化工作流
```

### 路线 B：已经会写 kernel，但不会做闭环优化

推荐顺序：

1. `chapter5-system-skills.md`
2. `chapter1-tier1-tier2.md`
3. `chapter2-tier3-tier4.md`
4. `chapter4-kernel-skills.md`
5. `chapter3-tier5-arch.md`

这条路线更适合“先把方法论立起来，再回补具体技巧”。

### 路线 C：按问题反查

如果你已经在做具体项目，也可以按问题倒查：

| 你遇到的问题 | 优先读 |
|---|---|
| 我连 `BLOCK_M/BLOCK_N/BLOCK_K` 到底在控制什么都不稳 | 第一章 |
| 我知道 kernel 对，但不知道该 fusion 还是改调度 | 第二章 |
| 我在 4090 上和在 MI300X/H100 上行为完全不一样 | 第三章 |
| 我想直接看 GEMM / FlashAttention / Softmax / RMSNorm | 第四章 |
| 我每次优化都像试错，没法稳定复盘 | 第五章 |

---

## 4. 两周学习路径（推荐节奏）

下面给一个更像课程表的版本。默认每天 1.5~3 小时。

### 第 1 周：先把方法框架立住

#### Day 1~2：第一章

目标：

- 建立 Thread / Warp / Block / Grid 心智模型；
- 能解释 coalescing、bank conflict、SMEM/L2 的基本作用；
- 看懂最小 Triton 地址表达骨架。

最低产出：

> 你要能自己解释，为什么 tile 过大和过小都会慢。

#### Day 3~4：第二章

目标：

- 区分 Tier 3 和 Tier 4；
- 说清 fusion 的收益来自哪里；
- 说清 Split-K / Persistent / Stream-K 分别解决什么问题。

最低产出：

> 给你一个 kernel，你至少能先回答“这是该先看 fusion，还是先看调度”。

#### Day 5：第三章

目标：

- 不再把跨平台理解成“固定参数通吃”；
- 明白架构差异应如何转化成调参和实现路径选择；
- 知道 bank conflict 为什么是跨平台共性问题。

最低产出：

> 你要能解释为什么同一 kernel 在 NVIDIA 与 AMD 上最优参数不一定一样。

### 第 2 周：把方法落到算子与工作流

#### Day 6~8：第四章

目标：

- 用统一模板拆 GEMM、Flash-Attention、Softmax、RMSNorm、Cross-Entropy、RoPE、MoE、FFN；
- 形成“先判瓶颈，再选方法”的习惯。

最低产出：

> 看到一个算子时，你能先说出它更像 compute-bound 还是 memory-bound。

#### Day 9~10：第五章

目标：

- 建立 profile → diagnose → optimize → verify → benchmark 闭环；
- 学会一次只改一个主点；
- 学会留优化日志与回滚记录。

最低产出：

> 给你一个慢 kernel，你知道第一步不是“改代码”，而是“先做 profile”。

#### Day 11~14：复盘与自测

建议做法：

1. 回看每章自测题；
2. 随机抽一个算子，用第四章模板手工拆一次；
3. 自己设计一个优化闭环，写出：
   - 你会先 profile 什么；
   - 你准备先试哪一个 tier；
   - 你如何 verify；
   - 你如何 benchmark。

---

## 5. 每章之间的依赖关系

### 第一章是地基

不理解：

- 执行模型；
- 地址表达；
- coalescing / bank conflict；

后面很多“优化”都会变成死记结论。

### 第二章是方法分流器

第二章的价值在于把两条常被混淆的路分开：

- **继续优化计算与数据流**；
- **重构并行结构与调度方式**。

### 第三章是平台修正器

它会提醒你：

> 前两章讲的是共性，但最终参数和实现路径仍要服从硬件现实。

### 第四章是应用场

这里不是学新概念，而是把前面 1~3 章的方法，落到具体 operator 上。

### 第五章是闭环器

它把“知识点”变成“工作方法”。

如果没有第五章，你会懂很多优化名词，但很难稳定产出优化结果。

---

## 6. 推荐的阅读方法

为了更接近真实工程学习，而不是只做“纸面阅读”，建议每章都按下面 4 步走：

### Step 1：先读“本章技能卡”

先看：

- 本章对应哪个 tier；
- 本章最重要的产出是什么；
- 读完应该会回答什么问题。

### Step 2：只抓主线，不求一次记全

第一次阅读时，每章只抓：

- 主问题是什么；
- 为什么以前的直觉不够；
- 这一章给出的正确判断框架是什么。

### Step 3：自己复述一遍

如果你能不用看文档、自己复述出：

- GEMM 为什么通常先看 tile 与矩阵路径；
- Softmax 为什么是 memory-bound；
- Flash-Attention 为什么不是“普通 attention 快一点”；
- 为什么 verify 和 benchmark 不能颠倒；

那你才算真正读进去了。

### Step 4：用一页纸做章节摘要

每章都建议你自己写一页摘要，只写：

1. 核心判断句；
2. 3 个常见坑；
3. 进入下一章前必须会的东西。

这一步对形成长期记忆很有用。

---

## 7. 学完后你应该具备的能力

如果这套教程读完且能通过自测，理想状态下你应该能做到：

### 能力 1：看到 kernel 先分型

你能先判断：

- compute-bound；
- memory-bound；
- latency-bound / mixed。

### 能力 2：能选 tier，而不是只会乱调参数

你不会一上来就试很多配置，而是先判断：

- 该先看 Tier 1/2；
- 还是该先做 Tier 3 fusion；
- 或者已经到了 Tier 4/5 的问题。

### 能力 3：能识别“看上去对，其实不稳”的实现

例如：

- 地址表达写反；
- 边界 mask 缺失；
- 低精度 reduce；
- layout 假设错误；
- 跨平台判断写错。

### 能力 4：能做最小闭环优化

你能独立完成：

```text
profile -> diagnose -> choose one change -> verify -> benchmark -> keep/revert
```

---

## 8. 一份简化版毕业标准

如果你想验证自己是不是真的学会了，可以用下面 4 个问题做毕业测试：

1. 给你一个新算子，你能先判断它更像 compute-bound 还是 memory-bound 吗？
2. 给你一个 Triton GEMM 骨架，你能检查 stride、mask、accumulator 有没有明显错误吗？
3. 给你一个“4090 上快、MI300X 上慢”的案例，你能先从架构差异而不是“代码坏了”来解释吗？
4. 给你一个慢 kernel，你能写出第一轮优化日志模板，而不是直接开始瞎改吗？

如果这 4 题你都能答得比较稳，这套教程就算真正学进去了。

---

## 9. 最后给你的使用建议

这套文档最好的用法不是“从头读完一次就结束”，而是：

- **第一次**：按顺序通读，建立地图；
- **第二次**：带着你自己的项目问题回来反查；
- **第三次**：把第四章和第五章当作实际优化 checklist 用。

如果你后面继续扩写这套教程，我建议优先补两类材料：

1. **每章一个最小可运行模板**；
2. **每章一个 profile/verify/benchmark 的最小实验脚本**。

这样整套内容就会更接近“既能学，也能直接用”的知识库。
