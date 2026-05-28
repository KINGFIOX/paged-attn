# paged-attn — 从零学 PagedAttention

一份**自上而下、可运行**的 PagedAttention 学习笔记。每一步都是一个独立脚本，
跑得通就能验证对了，跑不通就能立即定位卡在哪一步。

PagedAttention（vLLM 的核心机制）解决的是 LLM 推理时 KV cache 的内存碎片化问题。
我们一步步把它拆开：先讲清楚 KV cache 为什么浪费，再实现"操作系统式"的分页
管理器，最后写一份 pure-PyTorch 的 paged attention 前向，用它和连续 KV cache
做数值对照。

> 当前版本只走 naive PyTorch paged attention，没有 Triton/CUDA 内核。后续要加
> 速时可以把 `paged_attention_single` 换成融合内核，block table / KV pool 的
> 数据结构可以保持不变。

## 仓库结构

```
paged_attn/
  standard_attention.py       # step 01：标准 attention + 连续 KV cache，作为数值基线
  kv_cache_fragmentation.py   # step 02：量化连续布局浪费多少 GiB，分页能省多少
  block_manager.py            # step 03：KV pool、block table、free list、refcount + COW
  paged_attention_naive.py    # step 04：纯 PyTorch 版 paged attention（与 step 01 等价）
```

## 环境

GPU 机器（已在 NVIDIA A800 80GB / CUDA 12.7 driver 上验证）：

```bash
uv sync
```

会装好 `torch 2.5.1+cu124`、`transformers`、`rich` 等。

## 推荐学习顺序

按 step 编号顺序跑。各步运行命令：

```bash
uv run python -m paged_attn.standard_attention      # step 01
uv run python -m paged_attn.kv_cache_fragmentation  # step 02
uv run python -m paged_attn.block_manager           # step 03
uv run python -m paged_attn.paged_attention_naive   # step 04
```

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
- `BlockManager`：`new_sequence` / `reserve_slots` / `append_kv` / `fork` /
  `free_sequence`。`reserve_slots(seq, n)` 只做"在 block table 里腾位置"，
  不碰 K/V 数据；真正写数据走 `append_kv(seq, k, v)`，它内部先 `reserve_slots`
  再把 K/V 按 `(pos // block_size, pos % block_size)` 落位到 pool 里。
- **重点：copy-on-write**。`fork(parent)` 让 child 直接共享 parent 的 block，
  refcount 加一；只有当其中一方真的要往一个共享 block 里写时，才克隆这一个
  block，更新指针。这就是 beam search / n-best 不再爆显存的根因。

### 04. Paged attention 前向（pure PyTorch）

- 高层调用直接 `mgr.append_kv(seq, k, v)`：一次完成"腾位置 + 写 K/V"。
- step 04 额外保留了底层工具 `store_kv(pool, seq, k, v, offset)` —— 它写
  *已经预留好*的任意 offset，适合用来观察非追加写入时 block table 如何定位。
- 把 paged K/V 重新 gather 回 `[H, L, D]` 然后调 step 01 的 attention，作为
  paged 路径的参考实现。
- 与 step 01 的连续 KV cache 做端到端对照：误差 2e-7。
- 顺带验证 prefix sharing：3 个 child fork 同一个 prompt，prefix 部分的 attention
  输出在三个 child 上完全一致（误差仍 2e-7），证明共享 block 真的被三个序列同时
  正确读出。

## 数值精度小抄

| 步骤 | dtype | 对照 | 最大绝对误差 |
| --- | --- | --- | --- |
| 01 with-cache vs no-cache (含 Wq/Wk/Wv) | fp32 | no-cache streaming | 1.2e-7 |
| 04 paged vs contiguous | fp32 | step 01 | 2.4e-7 |
| 04 prefix-shared child prefix | fp32 | step 01 | 2.4e-7 |

## 参考

- vLLM 论文：[Kwon et al. 2023, "Efficient Memory Management for Large Language
  Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180)
- vLLM 源码：`vllm/attention/ops/paged_attn.py`, `vllm/core/block_manager_*.py`
- Flash Attention 在线 softmax：[Dao et al. 2022](https://arxiv.org/abs/2205.14135)
- Flash Attention 2（2D tiling，prefill kernel 的灵感）：
  [Dao 2023](https://arxiv.org/abs/2307.08691)
