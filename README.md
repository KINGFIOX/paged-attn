# paged-attn — 从零学 PagedAttention

一份**自上而下、可运行**的 PagedAttention 学习笔记。每一步都是一个独立脚本，
跑得通就能验证对了，跑不通就能立即定位卡在哪一步。

PagedAttention（vLLM 的核心机制）解决的是 LLM 推理时 KV cache 的内存碎片化问题。
我们一步步把它拆开：先讲清楚 KV cache 为什么浪费，再实现"操作系统式"的分页
管理器、写 paged attention 前向、用 Triton 写 prefill / decode 两个内核，
最后塞进一个最小推理引擎里看连续批处理 + 前缀共享 + 抢占；并独立演示 CPU swap
作为另一种抢占策略。

## 仓库结构

```
paged_attn/
  01_standard_attention.py      # 标准 attention + 连续 KV cache，作为数值基线
  02_kv_cache_fragmentation.py  # 量化连续布局浪费多少 GiB，分页能省多少
  03_block_manager.py           # KV pool、block table、free list、refcount + COW
  04_paged_attention_naive.py   # 纯 PyTorch 版 paged attention（与 01 校验等价）
  05_paged_attention_triton.py  # Triton 内核版 paged DECODE，约 20× 加速
  06_mini_inference_engine.py   # 最小推理引擎：prefill+decode+前缀共享+重算抢占
  07_paged_prefill_triton.py    # Triton 内核版 paged PREFILL（Lq>1, 支持 chunked）
  08_swap_preemption.py         # CPU swap 池 + swap_out/swap_in，作为抢占的另一种方案
```

## 环境

GPU 机器（已在 NVIDIA A800 80GB / CUDA 12.7 driver 上验证）：

```bash
uv sync
```

会装好 `torch 2.5.1+cu124`、`triton 3.1.0`、`transformers`、`rich` 等。

## 推荐学习顺序

按文件名前缀 01→08 顺序跑。各步运行命令：

```bash
uv run python paged_attn/01_standard_attention.py
uv run python paged_attn/02_kv_cache_fragmentation.py
uv run python paged_attn/03_block_manager.py
uv run python paged_attn/04_paged_attention_naive.py
uv run python paged_attn/05_paged_attention_triton.py
uv run python paged_attn/07_paged_prefill_triton.py
uv run python paged_attn/06_mini_inference_engine.py
uv run python paged_attn/08_swap_preemption.py
```

（07 在 06 之前跑因为 06 依赖 07 的 prefill 内核。）

各步的学习要点：

### 01. 标准 attention + KV cache

- 用一个真实的 `MiniSelfAttention(Wq/Wk/Wv)` 层，把 prefill / decode 两阶段写清楚。
- 同一组 token 跑两条解码路径：
  - **无 cache**：每步把"prompt + 已生成 token"整段重过 `Wq/Wk/Wv`，全量 attention。
  - **有 cache**：prefill 一次后，每步只投影那 1 个新 token，老 K/V 从 cache 读。
- 数值等价（fp32 误差 \~1e-7）+ 成本对比：toy 配置 `d_model=128, prompt=32, decode=64`
  下 FLOPs **278M → 6M，~46×**；wall time \~3×（toy 尺寸下被 Python 开销稀释）。
- 顺带量化"连续 KV cache"在 max_len 预留下浪费 75% 的预留空间，引出 step 02。

### 02. 为什么需要分页：KV cache 浪费量化

- 取 Llama-2-7B / Llama-3-8B / Llama-2-70B 的真实 head 配置，模拟 64 并发请求
  （长度服从混合 log-normal，模仿真实对话流量）。
- 比较连续布局 vs 分页布局（block_size=16 / 128）：
  - Llama-2-7B 连续：128 GiB 预留 / 24 GiB 实际使用，**19% 利用率**。
  - Llama-2-7B 分页(16)：24.45 GiB 预留 / 24.24 GiB 使用，**99% 利用率**。
- 输出表格回答"小 block_size 是不是越小越好"——内部碎片、block table 开销、
  内存合并都在权衡里。

### 03. 分页内存管理器

把 OS 的虚拟内存搬到 GPU：

- `KVPool`：一个 `[num_blocks, 2, n_heads, block_size, head_dim]` 大缓冲。
- `BlockAllocator`：带 refcount 的 free list（LIFO，让 GPU 缓存友好）。
- `Sequence`：一个请求对应一个"虚拟地址空间"——`block_table` 把逻辑块号
  映射到物理块号。
- `BlockManager`：`new_sequence` / `append_tokens` / `fork` / `free_sequence`。
- **重点：copy-on-write**。`fork(parent)` 让 child 直接共享 parent 的 block，
  refcount 加一；只有当其中一方真的要往一个共享 block 里写时，才克隆这一个
  block，更新指针。这就是 beam search / n-best 不再爆显存的根因。

### 04. Paged attention 前向（pure PyTorch）

- 把 K/V 写进 paged pool：`store_kv(pool, seq, k, v, offset)`，每个 token 通过
  `(logical_block, slot) = (pos // block_size, pos % block_size)` 落位。
- 把 paged K/V 重新 gather 回 `[H, L, D]` 然后调 step 01 的 attention，作为
  paged 路径的参考实现。
- 与 step 01 的连续 KV cache 做端到端对照：误差 2e-7。
- 顺带验证 prefix sharing：3 个 child fork 同一个 prompt，prefix 部分的 attention
  输出在三个 child 上完全一致（误差仍 2e-7），证明共享 block 真的被三个序列同时
  正确读出。

### 05. Triton 内核版 paged attention（DECODE）

- 每个 Triton program 处理一个 `(sequence, head)`，loop 走 block table 上每一个
  物理块。
- 内核里就是 **Flash-Attention 风格的在线 softmax**：跟踪 running max `m` 和
  归一化 `l`，每读一个 block 增量更新 `acc`。
- 数值上和 step 04 一致（fp16 容差，rel\~6e-4）。
- 在 batch=8 / 8 heads / head_dim=64、序列 12–1000 这组样例上：
  - PyTorch reference per step: \~2100 µs
  - Triton paged decode per step: \~103 µs（**\~20× 加速**）

### 07. Triton 内核版 paged attention（PREFILL）

- decode 内核只能 `Lq == 1`。这里我们写 `Lq > 1` 的版本：一个 program 处理
  `(head, query_tile_of_BLOCK_M_rows)`，二维分块 + Flash-Attention 2 风格的
  在线 softmax。
- causal 用绝对位置实现（`k_abs <= q_abs`），这让它**自动支持 chunked prefill**：
  长 prompt 切成 64-token chunk，每个 chunk 传不同的 `query_start`，结果与一次
  跑完 bit-exact。
- 同样验证三种使用场景：fresh prefill、chunked prefill、`Lq=1 + query_start>0`
  形式的 decode。误差均 \~2e-3（fp16 容差）。
- Lq=2048 上 1.6× 超过 PyTorch 参考，但更重要的是它**直接吃 paged K/V**——
  这是 vLLM/SGLang 在生产里跑长 prompt 时唯一可行的路径。

### 06. 最小推理引擎（prefill + decode + 抢占）

把前面所有组件拼成一个能跑的玩具：

- 调度循环：每步从 `pending` 队列里准入能放下的请求，
  再把当前 `running` 里所有请求合成一个 `[B, H, D]` 的 query batch，
  调一次 Triton paged decode 内核，append 一个 token，
  完成数 == `max_new_tokens` 就退场。
- **每个新 prompt 第一次出现时**走 step 07 的 prefill 内核计算 prompt 注意力
  （结果丢弃，但这是真实引擎要付的算力）。
- 准入控制：分两本账——**每个请求**只预留自己的 decode 块，**每个 prompt cache**
  独立预留自己的 prompt 块（这样 fresh request 退场后 prompt cache 依旧有人记账，
  避免"孤儿块"导致后续 OOM）。
- 前缀共享：相同 `prompt_id` 的多个请求 fork 同一个 prompt sequence，
  refcount 与 COW 都自然冒出来。
- **抢占（重计算式）**：当 pending 队首是高优先级请求而 pool 已满，把 LIFO
  顺序下最新进入 running 的低优 victim 踢出（释放 KV、放回 pending、标记
  `needs_recompute`）；它再次入场时把 prompt 重新 fork、按位确定性地重建
  decode 部分的 K/V。因为我们的"模型"对每个绝对位置都 deterministic 播种，
  抢占重算前后 K/V **bit-exact**。
- demo 输出对照：同样工作负载下，开/关抢占的高优请求"到达→完成"延迟从
  **81 步 → 15 步**（5.4× 改进），代价 90 token 的重算开销。

### 08. CPU swap 抢占

抢占的另一条路：

- `SwappableBlockManager` 维护两个 `KVPool`，一个 GPU 一个 CPU。
- `swap_out(seq)`：把序列每一块 D2H 拷贝到 CPU pool，释放 GPU 块，更新 block
  table 指向新 CPU 块。要求每个块 refcount==1（共享块需要更复杂的群体迁移）。
- `swap_in(seq)`：反向。
- demo：GPU 池 4 个块装满 2 个序列，第 3 个高优请求要进来 → 把序列 0 swap 到
  CPU → 新序列入场 → 等新序列退场后 swap 回 → 用 fingerprint 校验 swap 完全
  无损（误差严格为 0）。
- 注释里给出"在 step 06 引擎里也接入 swap"的接线方案：先选 refcount==1 的
  victim 走 swap，没有再退回到 recompute。

## 数值精度小抄

| 步骤 | dtype | 对照 | 最大绝对误差 |
| --- | --- | --- | --- |
| 01 with-cache vs no-cache (含 Wq/Wk/Wv) | fp32 | no-cache streaming | 1.2e-7 |
| 04 paged vs contiguous | fp32 | step 01 | 2.4e-7 |
| 04 prefix-shared child prefix | fp32 | step 01 | 2.4e-7 |
| 05 Triton decode vs PyTorch reference | fp16 | step 04 | 9.8e-4 |
| 07 Triton prefill, full | fp16 | step 04 | 2.0e-3 |
| 07 Triton prefill, chunked | fp16 | step 04 | 2.0e-3 |
| 08 swap_out + swap_in round-trip | fp32 | 原始数据 | **0** |
| 06 抢占前 vs 抢占后 token 计数 | — | 自身 | 一致 |

## 还没写、读完后可以自己加的

- **GQA**：05 / 07 的内核假设 `n_kv_heads == n_q_heads`。改成 GQA 只需把 `pid_h`
  分成 `q_head_id` 和 `kv_head_id = q_head_id // (n_q_heads/n_kv_heads)`。
- **多请求 prefill batching**：07 一次跑一个序列。生产 vLLM 用 varlen 接口
  `(packed_q, cu_seqlens_q, cu_seqlens_k)` 把一个批次的 prefill 一并发射。
- **swap 抢占接入引擎**：把 step 08 的 `SwappableBlockManager` 替换 step 06
  的 `BlockManager`，在 `_preempt` 中优先尝试 swap，失败再走 recompute。
- **量化 KV cache**：把 K/V 从 fp16 压成 int8 / fp8，pool buffer 改成 uint8，
  Triton 内核里在读出后做 dequant。

## 参考

- vLLM 论文：[Kwon et al. 2023, "Efficient Memory Management for Large Language
  Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180)
- vLLM 源码：`vllm/attention/ops/paged_attn.py`, `vllm/core/block_manager_*.py`
- Flash Attention 在线 softmax：[Dao et al. 2022](https://arxiv.org/abs/2205.14135)
- Flash Attention 2（2D tiling，prefill kernel 的灵感）：
  [Dao 2023](https://arxiv.org/abs/2307.08691)
