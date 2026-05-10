# ds4.rs 项目总结报告

日期：2026-05-10
对象：[Anda Bot](https://anda.bot/) `/goal` 长程任务：用 Rust 重写 DeepSeek V4 Flash 的 C 推理引擎

## 1. 摘要

2026 年 5 月 8 日，项目以一条 `/goal` 指令启动：仔细研究 C 语言实现的 DeepSeek V4 Flash 推理引擎，在 `rs/` 目录下用 Rust 重写，并迁移 C 单元测试，目标是得到性能接近 C 版本、能够实际运行的 `ds4.rs`。

截至本报告撰写时，项目已经形成完整的 Rust CPU 推理引擎：核心引擎提交在 `ds4/rs` 下新增 20 个文件，Rust 源码合计 11,748 行；相对上游 `origin/main`，`ldclabs/main` 只保留与 `rs/` 相关的两个提交，其中核心实现提交为 `b029822 Add Rust inference engine (rs/) — full DS4 port with performance optimizations`，README 更新提交为 `affd8c5 docs: update rs/README.md for v0.2.0 — add screenshot, performance data, optimization table`。

这次任务的结果不是一个“玩具移植”，而是一个能加载真实 81 GB GGUF 模型、能在 Linux 服务器上完成 forward 和自回归生成的 Rust 推理引擎。最终服务器验证显示：在 64 vCPU、256 GiB RAM 的 AMD 机器上，Rust v0.2.0 贪心 decode 约 1.1 tok/s；同场景 C CPU reference 约 2.7 tok/s。Rust 版本在 CPU-only 路径上达到了可用状态，但当前 decode 速度仍落后于 C CPU reference。

同时，这次项目也是一次很有价值的 `/goal` 长程推理压力测试。它证明了 Anda Bot 加 DeepSeek 4 Pro 能在多天、多上下文、多轮中断的情况下，完成一个高度复杂的系统编程任务；也暴露出当任务依赖大模型文件、长时间运行、真实硬件验证和自动续跑时，当前 `/goal` 模式仍需要更强的资源约束感知、自动恢复、阶段验收和“不要过早宣布完成”的机制。

## 2. 分析材料

本报告基于以下材料整理：

- Rust 工程产物：[README.md](README.md)、[Cargo.toml](Cargo.toml)、[src](src)、[tests/integration_test.rs](tests/integration_test.rs)。
- 本地私有开发日志。原始开发记录不随公开仓库发布，本报告只保留统计结果和复盘结论。
- 当前 Git 状态：`ds4` 子仓库的 `main` 分支位于 `affd8c5`，相对上游只引入 `rs/` 目录相关变更。
- 本次复核命令：生成 tiny GGUF 后运行 `DS4_TEST_MODEL=/tmp/test_ds4_report.gguf cargo test --features test-dimensions`。

本地私有开发日志的规模也说明了任务长度：记录覆盖 2026-05-07T17:37:57Z 到 2026-05-10T12:05:06Z，累计约 7,302 条消息、3,775 次工具调用、7,401 个模型请求、3.91 亿输入 token 和 195 万输出 token。其中 47 段记录状态为 `completed`，2 段为 `failed`，3 段为 `working`，这些 `failed/working` 多数对应上下文中断、任务切换或手工继续点。

## 3. 最终工程产物

`ds4.rs` 是一个专门面向 DeepSeek V4 Flash 的 Rust CPU 推理引擎，不是通用 GGUF runner。它沿用 C 引擎的模型格式和架构假设，重点实现 CPU forward、KV cache、tokenizer、量化矩阵乘、CLI 和测试设施。

核心文件与职责如下：

| 文件                                                   |  行数 | 主要职责                                                                                             |
| ------------------------------------------------------ | ----: | ---------------------------------------------------------------------------------------------------- |
| [src/forward.rs](src/forward.rs)                       | 2,815 | 完整 forward pass：embedding、HC attention、MoE FFN、KV cache、batched prefill、spec decode 验证路径 |
| [src/quant.rs](src/quant.rs)                           | 1,786 | Q2_K、Q4_K、IQ2_XXS、Q8_K、Q8_0 的 block 格式、dequantize、AVX2/scalar dot product                   |
| [src/lib.rs](src/lib.rs)                               | 1,731 | 数值基础函数、RMS norm、SwiGLU、FP16/FP8、HC Sinkhorn、量化黄金测试                                  |
| [src/tokenizer.rs](src/tokenizer.rs)                   | 1,359 | JoyAI/GPT-2 byte encoding tokenizer、BPE、特殊 token、乱码修复相关 decode                            |
| [tests/integration_test.rs](tests/integration_test.rs) | 1,160 | tiny GGUF 集成测试、C cross-validation、forward/session/generation 测试                              |
| [src/gguf.rs](src/gguf.rs)                             |   797 | GGUF mmap 读取、metadata/tensor 解析、测试 GGUF 写出                                                 |
| [src/bin/ds4.rs](src/bin/ds4.rs)                       |   664 | 生产 CLI：one-shot、interactive chat、thinking mode、采样、调试开关                                  |
| [src/model.rs](src/model.rs)                           |   436 | 模型权重绑定、校验、测试模型构造                                                                     |
| [src/session.rs](src/session.rs)                       |   335 | 自回归会话、KV cache 生命周期、logits、speculative decode 调用                                       |
| [src/bin/gen_test_gguf.rs](src/bin/gen_test_gguf.rs)   |   305 | 生成 tiny GGUF 测试模型                                                                              |
| [src/bin/ds4-head-test.rs](src/bin/ds4-head-test.rs)   |   211 | C/Rust 层级对齐和 head-test 调试工具                                                                 |
| [src/constants.rs](src/constants.rs)                   |    91 | 生产维度和 `test-dimensions` tiny 维度切换                                                           |
| [src/bin/dump_logits.rs](src/bin/dump_logits.rs)       |    58 | logits/debug 辅助工具                                                                                |

核心引擎提交中，`rs/` 目录相对上游新增 20 个文件，Git diff 统计为 12,664 行增量；其中 Rust 源码为 11,748 行，README 和图片等文档/资产不计入源码行数。本报告和博客草稿属于后续公开叙事材料，不计入上述工程规模。

## 4. 技术实现要点

### 4.1 架构对齐

Rust 版本保留 DeepSeek V4 Flash 的固定架构参数：生产维度为 43 层、`n_embd=4096`、`vocab_size=129280`、256 routed experts、每次使用 6 个 experts、FFN expert hidden dim 为 2048。`test-dimensions` feature 则把模型缩小到 1 层、`n_embd=256`、`vocab_size=256`、4 个 experts，便于无真实 81 GB 模型时运行单元和集成测试。

forward 层基本复刻 C 引擎路径：

- token embedding 读取 F16 权重；
- HC split + Sinkhorn，生成 pre/post/comb 权重；
- Q/KV projection、RoPE、FP8 KV quantization；
- sliding-window raw KV 与 compressed KV cache；
- attention rows、grouped output、HC post；
- MoE router、hash routed experts、shared experts；
- IQ2_XXS gate/up、Q2_K down、Q8_0 shared FFN；
- 输出 head 和 sampling。

这种实现路径的关键价值是可审计：Rust 不是另起炉灶，而是在大量局部对齐测试、trace 工具和 C/Rust 数值对比之后，逐步吸收 C 代码的结构，再在热点路径做 Rust 化和并行化优化。

### 4.2 Tokenizer 与乱码修复

真实模型第一次在服务器上跑起来后，最早的主要故障不是 forward 崩溃，而是 tokenizer 和 prompt template 相关问题。

对话记录显示，服务器首先报出 `missing vocabulary size`。根因是 GGUF metadata 使用 `deepseek4.*` 前缀，而 Rust 解析器只查找 `ds4.*` 或 `llama.*`，甚至还跳过了部分 `deepseek4.*` key。修复后，模型可以读取 `vocab_size=129280, bos=0, eos=1`。

随后出现 `ĠæĢ³...` 或中文词片段循环等乱码。这个阶段先后确认了两个问题：

- 输出端需要把 GPT-2 byte encoding 正确反解码，否则 `Ġ`、`æĢ³` 这类内部 token 文本会直接显示出来。
- prompt template 不能把 `<｜User｜>`、`<｜Assistant｜>`、`<think>` 当普通文本再过 BPE；必须直接使用 GGUF 里的特殊 token id。

修复后，prompt 从类似 `<BOS>You are a helpful assistantUserhiAssistant在校期间及比如` 的错误形式，变为 `<BOS>You are a helpful assistant<｜User｜>hi<｜Assistant｜><think>` 这样的正确 token 序列。后续真实模型输出也从重复乱码，进入可读内容。

### 4.3 性能优化

项目中期曾讨论是否接入 BLAS。结论是 BLAS 对主要瓶颈帮助有限，因为核心耗时不在大块 F32 GEMM，而在低比特量化权重和 Q8 activation 的专用 dot product：IQ2_XXS、Q2_K、Q8_0、MoE expert 路径才是真正热点。因此优化重点转向 SIMD、批处理和 Rayon 并行。

最终 README 记录的 v0.2.0 优化包括：

- IQ2_XXS AVX2 SIMD：预计算 i16 LUT，使用 `_mm_madd_epi16` 等路径提升 matvec。
- Q2_K AVX2 SIMD：使用 `_mm256_maddubs_epi16` 和 srlv_epi32 类路径优化 2-bit 提取与乘加。
- Parallel expert rows：Rayon 并行 routed expert rows。
- Batched shared/routed experts：`rayon::join` 并发执行 shared 和 routed expert。
- Gate+up dual-channel fusion：一次遍历同时计算 gate 与 up，减少重复读取 Q8 activation。
- Wide multi-row IQ2XXS：多输出行共享 Q8 输入，提高 cache 复用。
- Parallel F16 gate/up/down：shared expert 的 F16 路径并行化。

这些优化并非一次成功。对话中出现过 `022b95d` 早期 SIMD 版本 prefill 卡住、`aplaplapla...` 循环输出、Q2_K SIMD 横向求和错误、IQ2_XXS signedness bug、speculative decode 接受率为 0% 等问题。最终可用版本的形成，依赖多条临时分支和服务器反馈：`perf-nosimd`、`perf-simd`、`perf-simd-iq2xxs`、`perf-q2k-fix`、`perf-both` 等分支承担了隔离验证的作用。

### 4.4 Speculative decode 的结论

Rust 版本实现了 8-layer draft 的 speculative decode，并通过 `--spec` 暴露为可选实验功能。但服务器诊断显示，对 DS4 真实模型来说，draft token 接受率为 0%。用户测试中每轮都是 `drafts generated: 3, accepted: 0`，decode 反而没有收益。

因此最终方案是把 speculative decode 改成默认关闭，仅通过 `--spec` 显式开启，并在 README 中标注为未来实验方向。这是一个重要的工程判断：实现了功能不等于默认启用，真实模型验收结果优先于“看起来先进”的算法选择。

## 5. 测试与验证

项目建立了三层验证体系。

第一层是无需真实模型的 Rust 单元测试。当前代码中共有 158 个 `#[test]`，分布为：

- [src/lib.rs](src/lib.rs)：33 个，覆盖基础数值、量化 matvec、黄金对比等。
- [src/tokenizer.rs](src/tokenizer.rs)：60 个，覆盖 byte encoding、pre-tokenizer、decode、特殊 token、乱码复现。
- [src/forward.rs](src/forward.rs)：5 个，覆盖 indexer、batched prefill 等局部 forward 行为。
- [tests/integration_test.rs](tests/integration_test.rs)：60 个，覆盖 tiny GGUF、forward/session/generation、C tokenizer cross-validation。

第二层是 tiny GGUF 测试模型。`gen_test_gguf` 可以生成 199 KB 级别的测试模型，让 `test-dimensions` 路径在普通机器上运行。这一层解决了“真实模型 81 GB，不适合本地测试”的核心矛盾。

第三层是真实模型服务器验证。由于开发电脑内存不足，真实模型验证迁移到 64 vCPU、256 GiB 的 AMD Linux 服务器上完成。服务器反馈推动了多个关键修复：metadata 前缀、prompt special token、prefill 并行、SIMD 正确性、spec decode 默认关闭等。

本报告撰写时额外做了一次测试复核：

```sh
cd ds4/rs
cargo run --features test-dimensions --bin gen_test_gguf -- /tmp/test_ds4_report.gguf
DS4_TEST_MODEL=/tmp/test_ds4_report.gguf cargo test --features test-dimensions
```

复核结果：98 个 lib 测试和 60 个 integration 测试全部通过。测试输出仍有一个已有 warning：`tests/integration_test.rs` 中 `read1` 未使用。复核过程中发现 3 个测试断言已滞后于当前 tokenizer 语义，并已修正：

- `token_to_bytes()` 对原始 CJK 字符按 C 行为跳过；实际 CJK token 应先经过 GPT-2 byte encoding。
- 最小 test vocab 不包含 newline byte token，因此 `>;\n` 在该测试 vocab 下编码/解码为 `>;`。
- `decode()` 当前会把 `HelloĠworld` 还原为真实文本 `Hello world`。

## 6. 性能结果

README 中记录的最终性能数据如下。

在 AMD EPYC 64-Core / 256 GB RAM Linux 服务器上：

| 指标                           |  Rust v0.2.0 | C CPU reference |
| ------------------------------ | -----------: | --------------: |
| Decode，greedy，81 GB q2 model | 约 1.1 tok/s |    约 2.7 tok/s |

需要注意：本报告当前只保留 Linux 服务器上的实测数据，未给出 macOS 实测结论。且 C 引擎的主力路径是 macOS Metal GPU，而 Rust 版本目前 CPU-only，因此这里的对比仅限 CPU reference 路径。

## 7. 已知限制

当前 `ds4.rs` 已经能实际运行，但仍有清晰边界：

- CPU-only，没有 GPU/Metal/CUDA 加速。
- CLI only，没有 OpenAI/Anthropic compatible server。
- 没有 disk KV cache，每次会话从空 KV 开始。
- Speculative decode 对 DS4 真实模型接受率为 0%，默认关闭。
- 真实 81 GB 模型与 C 的 logits 仍存在数值差距，README 记录 max logits 约 29.2 vs 16.8，怀疑与 43 层 Sinkhorn routing 的 FP accumulation 相关；不过 synthetic test model 可以做到 C/Rust identical logits，真实输出也已能保持可读和 coherent。

这些限制并不削弱项目结论，反而说明项目报告应该避免过度营销：Rust 版本是一个可运行、可测试、可优化的 CPU 推理引擎，但不是完整替代原 C+Metal server 的生产系统。

## 8. 对话时间线

### 阶段一：从零移植到“看起来完成”

初始 `/goal` 指令要求研究 C 推理引擎，在 `rs/` 下用 Rust 重写，并迁移测试。前几轮中，Anda Bot 快速建立 Rust crate、GGUF reader、基础常量、量化格式、测试模型生成器、tokenizer、forward/session 雏形等。早期 handoff 曾写出“Goal Complete”，但从后续事实看，这个完成判断偏乐观：当时 synthetic tests 和局部功能已完成，但真实 81 GB 模型还没有跑通。

用户追问“完成得怎么样，还要继续吗？”非常关键。它把任务从“能编译、有测试”拉回到“必须与 C 对齐”。这之后的工作转向 Q8_K、IQ2_XXS、Q2_K、HC state、FFN gate gap 等数值细节。

### 阶段二：C/Rust 数值对齐与测试体系强化

这一阶段主要处理 C/Rust 差异。项目通过 `ds4-head-test`、`dump_logits`、C tiny test、Rust tiny GGUF 等工具，把 forward 的中间量拆开对比。过程中修复了 Q8_K convention、IQ2_XXS sign decode、Q2_K dequantize nibble separation、expert down/shared FFN 测试等问题。测试数量从早期 59 个、94 个逐步扩展。

随后用户要求完善 Rust README，说明项目已经进入“可以给朋友试”的阶段，但用户同时强调自己的电脑跑不起来，这为后续硬件约束埋下伏笔。

### 阶段三：真实模型服务器验证与硬件约束

真实模型服务器验证是项目的转折点。用户在 64 vCPU、256 GiB AMD 服务器上运行 Rust 版本，先遇到 `missing vocabulary size`，随后遇到 prefill 长时间无首 token，再遇到首 token 乱码。这一阶段明确了两个事实：

- 开发电脑无法承载真实 81 GB 模型，真实验证必须在服务器上做。
- Synthetic test 通过不等于真实模型通过，真实模型暴露的问题更接近最终用户体验。

用户指出“你又把我的电脑搞挂了，我的电脑配置低，不能跑 gguf”。这是整个项目最重要的负面事件之一。它暴露出 `/goal` 模式在工具执行前缺少足够强的资源风险判断：即使文件路径可读，也不代表应该读取或加载。

之后出现了“终于看到 token”和“这回是真跑起来了”的节点。虽然当时仍有 prompt 乱码和速度慢的问题，但 Rust 引擎已经能完成真实模型加载、prefill、logits 有限性检查和输出。

### 阶段四：SIMD/并行优化与多轮回归

真实模型跑通后，用户把注意力转向最大瓶颈：先 SIMD 化 IQ2_XXS，再做 P1-P5 性能优化。优化路线包括：批量 block、Q2_K SIMD、gate+up fusion、宽向量化、多线程 shared/down projection。

这段是最像真实工程的部分：不是线性前进，而是不断出现回归、分支验证、撤销和重组。

- 早期 SIMD 版本导致 prefill 很久不出结果。
- P1-P5 版本在服务器上出现 `aplaplapla...` 循环输出。
- HC state 不连续、comb matrix transpose 误修、Q2_K SIMD horizontal sum、IQ2_XXS signedness 等问题逐个暴露。
- `--no-spec`、`--scalar`、`--trace-hc`、`--trace-ffn`、`--trace-moe` 等调试开关被加入，用于缩小问题范围。
- spec decode 被证明 0% acceptance，于是从默认路径移除，改为 `--spec` opt-in。

最终服务器输出达到可用状态：`你好！我是DeepSeek，很高兴为你服务`，10 tokens decode 约 1.1 tok/s。随后用户要求剥离对原项目文件的变更，只保留 `rs/` 目录，方便跟进上游 antirez 项目。最终历史被整理成相对 `origin/main` 只 ahead 一个 Rust engine commit 的结构，完整过程保存在 `archive/main-full` 归档分支中。

### 阶段五：README、截图与过程归档

最后阶段处理 v0.2.0 README 截图和性能数据，并把开发过程记录私下归档。这一步把工程成果、过程证据和复盘材料完整串起来。没有这些私有过程记录，本报告无法还原很多关键决策和失败路径。

## 9. /goal 长程推理能力评估

### 9.1 成功点

第一，任务保持能力很强。尽管上下文多次压缩、中断、失败和手工继续，系统始终能围绕“Rust 重写 DS4 推理引擎”这个核心目标继续推进。大量 handoff 虽然显得啰嗦，但有效保留了目标、路径、当前分支、未验证点和下一步命令。

第二，工程执行能力达到了高级系统编程水平。项目涉及 GGUF binary parsing、mmap、量化格式、SIMD intrinsics、Rayon 并行、MoE routing、KV cache、tokenizer、CLI、测试生成器和 C/Rust 对齐工具。对于一个完全由 LLM 编写的 Rust 引擎，11,748 行源码和真实模型可运行这一结果非常突出。

第三，调试策略逐渐成熟。早期有过过度自信和误判，但中后期开始形成正确工程节奏：小模型测试、真实模型验证、分支隔离、trace flag、server 反馈、回滚、再 merge。尤其是 speculative decode 默认关闭这一决策，体现了“以实测为准”的工程判断。

第四，人机协作边界清晰后，效率明显提高。开发电脑不能跑真实模型后，用户提供服务器执行结果，Anda Bot 根据日志继续修复。这种“LLM 写代码、用户提供真实硬件回执”的模式，是大模型系统工程任务的一种可行协作形态。

### 9.2 暴露的问题

最大问题是资源风险感知不足。81 GB GGUF 模型对低配开发机是危险操作，但 Agent 两次导致电脑卡死，说明 `/goal` 在长程任务中需要把“硬件约束”提升为强约束，而不是普通上下文提示。类似约束应该进入 session state、工具策略和 supervisor 检查。

第二个问题是自动续跑不足。用户明确提到前期 `/goal` 模式不够完善，过程中停止了没有自动继续。对话记录里也能看到多个 `working`、`failed`、“请继续”、“异常中断了，请继续”的节点。长任务需要可靠的 background task 生命周期、闲置检测、失败恢复和完成判定。

第三个问题是阶段验收不够严格。早期 handoff 写过“Goal Complete”，但真实模型尚未运行；后期 SIMD 也有“完成”后服务器翻车的情况。对于这类任务，完成标准应该分层：编译完成、tiny tests 通过、C/Rust synthetic 对齐、真实模型加载、真实 prompt 输出、性能目标达成。只有最后一层通过，才能称为项目完成。

第四个问题是有时会产生错误方向的修复。comb matrix transpose fix 导致 HC 爆炸更严重，Q2_K SIMD 初版也引入乱码。这不是 LLM 独有问题，真实工程也会发生；关键是 `/goal` 模式需要更自动化地建立“假设、验证、回滚”闭环。

### 9.3 对 Anda Bot 的改进建议

1. 为 `/goal` 增加强资源约束机制：文件大小、内存需求、模型加载风险、禁止本地读取大文件等信息应进入硬约束，并在 shell/file tool 调用前检查。

2. 引入阶段性 done gate：例如“真实模型未在目标硬件上跑通，不允许输出 Goal Complete”。每个目标可以声明验收矩阵，supervisor 按矩阵判断是否完成。

3. 增强自动恢复：长任务如果因上下文、工具、网络或命令失败中断，应自动生成 continuation plan 并继续，而不是等待用户说“请继续”。

4. 把 handoff 结构化：当前 handoff 文本可读但冗长。可以固定字段：Objective、Constraints、Current Branch、Changed Files、Verified Commands、Known Failing Cases、Next Command。

5. 对真实硬件验证建立远程回执协议：当 Agent 无法直接访问服务器时，可以让用户粘贴标准化输出；Agent 根据输出自动分类为 build error、runtime error、quality regression、performance regression。

6. 对高风险优化采用 branch-and-gate：SIMD、并行、spec decode 等优化默认在临时分支验证，只有真实模型质量和性能都过门槛才合入主线。

7. 记录负面知识：例如“BLAS 不适合当前低比特 MoE 热点”“spec decode 0% acceptance”“本地不要加载 81 GB GGUF”，这些都应进入 repo memory，防止未来重复探索。

## 10. 项目意义

这个项目的意义不只在于多了一个 Rust 版 DS4 推理引擎。更重要的是，它展示了一个复杂系统工程任务可以被 Agent 以长程方式推进：从阅读 C 代码、建立 Rust 模块、迁移测试、排查真实模型问题，到性能优化、文档整理、对话归档，整条链路都由 Anda Bot 与 DeepSeek 4 Pro 主导完成。

从结果看，“完全由 DeepSeek 4 Pro 编写”并不意味着没有人类参与。人的参与主要体现在目标设定、真实硬件测试、关键反馈、方向纠偏和最终验收上。真正有效的模式不是“人退出工程”，而是人把精力放在约束、验证和判断上，让 Agent 承担大规模编码、搜索、调试和文档整理。

从 `/goal` 模式看，这次任务已经证明它具备完成多日复杂目标的潜力；但如果要变成可靠的自动化工程能力，还需要把资源约束、阶段验收、失败恢复和真实环境验证做成一等公民。这个项目是一个非常好的里程碑，也是一份很清楚的改进路线图。

## 11. 附录：关键命令与最终状态

代码规模统计：

```sh
rg --files ds4/rs -g '*.rs' | xargs wc -l
# 11748 total
```

当前 tiny-model 测试复核：

```sh
cd ds4/rs
cargo run --features test-dimensions --bin gen_test_gguf -- /tmp/test_ds4_report.gguf
DS4_TEST_MODEL=/tmp/test_ds4_report.gguf cargo test --features test-dimensions
# lib: 98 passed
# integration: 60 passed
# doc-tests: 0 passed
```

当前 Git 摘要：

```text
affd8c5 (HEAD -> main, tag: v0.2.0-rs, ldclabs/main) docs: update rs/README.md for v0.2.0
b029822 Add Rust inference engine (rs/) — full DS4 port with performance optimizations
8e7575b (origin/main) Fix JSON parser nesting DoS
```

本报告撰写过程中还修正了以下测试断言，使当前 `test-dimensions` 测试与 tokenizer 的实际语义一致：

- [src/tokenizer.rs](src/tokenizer.rs)
- [tests/integration_test.rs](tests/integration_test.rs)

这些测试断言修正只调整测试期望，不改变推理实现。报告文件本身位于 `rs/` 下，便于公开阅读。

## 12. 试试 Anda Bot

如果你对这种长程智能体工作流感兴趣，欢迎试试 Anda Bot：

https://anda.bot/

也欢迎给项目一个 star：

https://github.com/ldclabs/anda-bot

我希望更多人拿它去做真正有难度的项目，而不是只让它写玩具 demo。智能体到底能走多远，最好的答案还是让它在真实工程里多跑几次。
