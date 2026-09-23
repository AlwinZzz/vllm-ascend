# DSV4 QB TP 命名字典与术语详解

> **配套文档**：《DSV4 QB TP 改动详解（代码级导读）》（同目录 `deepseek_v4_qb_tp_implementation.md`）、《DeepSeek-V4 并行方案迁移：改动清单与设计》（`deepseek_v4_parallel_migration.md`）
> **用途**：阅读上述文档与相关代码时的变量/模块/算子命名速查；附录收录两个高频疑问的专题解释（attn_sink、wq_a 复制）。

---

## 1. `qr` 详解（整条链路的枢纽）

DSV4 的 Q 不是一步算出来的，而是**低秩分解两步走**，`qr` 是中间产物：

```
hidden_states x [T, 4096]
   │
   ├─ wq_a（4096 → 1536，每卡复制）
   ▼
q_a（multistream 路径叫 wq_a_result）[T, 1536]     ← "压缩后的 Q"（未归一化）
   │
   ├─ q_norm（RMSNorm）
   ▼
qr [T, 1536]        ← ★ q reduced / q latent：归一化后的低秩压缩 Q
   │
   ├──────────────→ wq_b（1536 → 32768，QB TP 切的就是这步）→ q [T, 64, 512]
   └──────────────→ Indexer（C4A 层选稀疏 top-k 也吃 qr）
```

- **命名含义**：q**r** = q reduced / q latent，"压缩表示的 Q"（Q-LoRA latent），每 token 一行 1536 维。
- **重要性**：它同时是 wq_b 的输入和 Indexer 的输入。QB TP 的全部动作 = "把 qr 聚到全组 → 各卡算自己 16 头 → 把 head 还给 qr 属主"。
- **铁律**：函数返回的 `qr` / `qr_pertoken_scale` **永远是本地 token 原值**（Indexer 要用），聚合版是独立变量 `qr_qb`，二者绝不混用。

## 2. 数据流变量名（按计算顺序，qb=4、decode 为例）

| 变量 | 含义 | 形状 | token 域 |
|---|---|---|---|
| `hidden_states` | 层输入 | `[T_loc, 4096]` | 本地 |
| `q_a` / `wq_a_result` | wq_a 输出（压缩 Q，未归一化） | `[T_loc, 1536]` | 本地 |
| `qr` | q_a 过 q_norm 后的压缩 Q | `[T_loc, 1536]` | 本地（留给 Indexer） |
| `qr_qb` | qr 的 AllGather 聚合版（含全组 token） | `[4·E, 1536]` | 聚合域 |
| `cos` / `sin` | 本层 rope 旋转表（按 token 位置查表） | `[T_loc, 64]` | 本地 |
| `cos_qb` / `sin_qb` | cos/sin 聚合版（别人的 token 用它们自己位置的旋转角） | `[4·E, 64]` | 聚合域 |
| `q_b_quant` / `q_b_scale` | 实际喂进 wq_b 矩乘的激活及 scale（bf16 路径 = `qr_qb` 本身、scale=None；w8a8 = int8+scale） | 同 `qr_qb` 行数 | 聚合域 |
| `qr_pertoken_scale` | w8a8 融合算子产出的 qr 量化 scale（属于**本地** qr，随 qr 返回给 Indexer） | `[T_loc, …]` | 本地 |
| `q` | wq_b 输出 unflatten 后的 Query；qb 域内是 head 分片，归还后恢复完整 | `[4·E, 16, 512]` → `[T_loc, 64, 512]` | 聚合 → 本地 |
| `num_tokens` / `T_loc` | 本卡实际 token 数（归还时裁 padding 用） | 标量 | — |
| `E` / `exchange_num_tokens` | 静态交换容量 = `get_potential_max_tokens()`（decode 桶上限） | 标量 | — |
| `k` / `qb_tp_size` | QB 并行度（配置值） | 标量 | — |

## 3. head 数三个易混名

| 名字 | 定义 | 何时有别 |
|---|---|---|
| `n_heads` / `num_heads` | 模型总 head 数 = 64 | 常量 |
| `n_local_heads` | `64 // tensor_parallel_size`（**标准 TP** 维度的本地头数） | qb 场景 tp=1 → 恒为 64 |
| `qb_n_local_heads` | 本次新增：wq_b 之后每卡实际头数 = qb 开启时 `64 // qb_tp_size`（=16），关闭时 ≡ `n_local_heads` | 所有 `unflatten(-1, (·, head_dim))` 统一改用它 |

## 4. 模块 / 权重命名（Attention 内）与并行形态

| 名字 | 是什么 | 大小（每层） | 并行形态 |
|---|---|---|---|
| `wq_a` | 压缩投影 4096→1536 | ~6.3M | **复制**（原因见附录 B） |
| `q_norm` | 对 q_a 的 RMSNorm（有 weight） | 1536 | 复制 |
| `q_norm_without_weight` | head 内 RMSNorm（无 weight，只有 eps），由 `apply_dsa_q_rms` 调用 | 0 | 无参数 |
| `wq_b` | 升维投影 1536→32768，**QB TP 的切分对象** | ~50M | qb 组列切 1/k（或标准 TP 列切） |
| `cv_wq_a` / `cv_wq_b` / `cv_wkv` | `CVLinearWrapper` 包装：拆成 quantize(Vector)+matmul(Cube) 供多流调度；检测到自定义 op 自动退化为直通+完整 forward | — | 跟随被包装层 |
| `wkv` / `kv_norm` | 共享 KV latent 投影 4096→512 及其 norm | ~2M | **复制**（附录 B） |
| `wo_a` / `wo_b` | 分组 O-LoRA：每组 8192→o_lora（列切）、合并回 4096（行切） | 大 | 标准 TP 或 OTP 组 |
| `attn_sink` | 每 head 一个的 softmax"泄压"标量（附录 A） | 64 个 float32 | qb/DP 场景全量；标准 TP 按本地头数创建 |
| `indexer` | C4A 闪电索引器；**内部也有一个同名 `wq_b`**（ReplicatedLinear），prefix 选择规则必须排除 `"indexer"` | 小 | 复制 |
| `compressor` | C4A/C128A 的 KV 压缩器 | 小 | 复制 |
| `gate`（MoE） | 路由打分 4096→256 | ~1M | 复制（各卡路由结果必须一致，是 MoE TP 的前提） |

## 5. QB 机制命名（本次新增）

| 名字 | 是什么 |
|---|---|
| `qb_tensor_parallel_size` | 配置字段（`additional_config.finegrained_tp_config` 下），0/1=关 |
| `qb_tp_enable()` | 全局判定函数（utils.py） |
| `qbtp` 组 / `_QB_TP` / `get_qb_tp_group()` | 按 DP rank 连续分块的通信组及访问器 |
| `DSV4QBColumnParallelOp` | 绑定 qb 组的列切 Linear op（只换 `comm_group`，计算纯本地） |
| `custom_op` | Ascend Linear 上挂的策略对象；`get_parallel_op(prefix,…)` 按前缀选择，其 `tp_rank/tp_size` 决定权重切段与 forward 行为。dsa init 用 `isinstance(qb_op, DSV4QBColumnParallelOp)` **按层检测权重是否真被切** |
| `QBTPExchange` / `qb_exchange` | 交换器类 / 每层持有的单例引用（None=关闭） |
| `gather_tokens` / `return_heads` | 聚 token（AllGather）/ 还 head（all-to-all） |
| `_qb_gather_input` / `_qb_return_heads` | impl 上的直通封装（关闭态恒等返回） |
| `_qb_profile_run` | profiling 零输入预热（供图捕获录制 collective） |
| `_static_buf` / `_buffers` | 静态缓冲及字典（key=用途+名称+尾维+dtype，地址终身稳定） |
| `all_gather_into_tensor` / `all_to_all_single` | torch.distributed raw 集合通信（连续缓冲版；不用 GroupCoordinator 包装，因其每次分配新内存、破坏图重放地址稳定性） |
| `get_potential_max_tokens()` | 静态交换容量 E（runner 初始化一次算定），使所有图桶形状恒定、免运行时同步 |

## 6. 算子命名

| 名字 | 作用 |
|---|---|
| `unflatten(-1, (H, D))` | 把 `[T, H·D]` 末维展开成 `[T, H, D]` |
| `DeviceOperator.apply_dsa_q_rms(q, eps, q_norm_without_weight)` | head 内无权重 RMSNorm |
| `torch.ops._C_ascend.inplace_partial_rotary_mul(x, cos, sin, partial_slice=[nope_head_dim, head_dim])` | 原地部分 rope：每 head 仅后 `rope_head_dim` 维参与旋转，`interleave` 交错模式 |
| `npu_rms_norm_dynamic_quant` | norm+int8 动态量化融合算子（w8a8 路径；qb 首期拒绝，因其量化发生在聚合前、scale 与行序错位） |
| `npu_quant_matmul` / `npu_dynamic_quant` | int8 量化矩乘 / 动态量化 |
| `_is_w8a8_dynamic(layer)` | 判定 Linear 是否挂 W8A8 动态量化方法 |
| `nope_head_dim` / `rope_head_dim` | head_dim 的两段：不旋转段 / 旋转段（`partial_slice` 分界） |

## 7. prefix 命名（选择规则的判定依据）

| prefix 形态 | 归属 | 是否被 qb 切 |
|---|---|---|
| `model.layers.N.self_attn.wq_b` | 主模型 attention | ✅ |
| `…self_attn.indexer.wq_b` | Indexer 自己的同名层 | ❌（排除 `indexer`） |
| `mtp.…self_attn.wq_b` | MTP draft 层 | ❌（排除 `mtp`，draft 保持复制） |
| `…self_attn.wo_a / wo_b` | o_proj | 走 OTP/标准 TP，与 qb 无关 |

**最易混提示**：`qr`（本地，给 Indexer）vs `qr_qb`（聚合，给 wq_b）；`cos/sin`（本地，KV 的 rope 用）vs `cos_qb/sin_qb`（聚合，q 的 rope 用）。所有前向改动本质上就是把这四个名字在各调用点正确分开。

---

## 附录 A：`attn_sink`——"泄压标量"是什么意思

### A.1 问题：softmax 的"总和必须为 1"是强制约束

```
o = Σᵢ wᵢ·vᵢ ，  wᵢ = exp(sᵢ)/Σⱼ exp(sⱼ) ，  Σᵢ wᵢ ≡ 1
```

无论 query 在 KV 里有没有"值得看"的内容，100% 的注意力都必须分配出去。当某 token 查询历史发现**没有任何相关 key** 时，softmax 仍把概率摊在一堆不相关 token 上，输出 = 无关 value 的加权平均——纯噪声，却照样混进残差流。

### A.2 解法：给 softmax 分母加一个可学习的"排水口"

```
o = Σᵢ [ exp(sᵢ) / ( Σⱼ exp(sⱼ) + exp(sink) ) ] · vᵢ
```

- 真实 token 权重之和 **< 1**，"消失"的概率质量流向 sink——等价于存在一个 **value 为零向量的虚拟 token**，query 可以把注意力"倒进"它；
- 无相关内容时（所有 sᵢ 小）：分母由 `exp(sink)` 主导 → 所有 wᵢ→0 → **输出干净地衰减到 ≈0**，而非噪声均值；
- 有强相关内容时：`exp(sᵢ) >> exp(sink)` → sink 几乎不起作用，退化为普通 softmax。

"泄压"比喻的由来：归一化约束像必须保压的密闭容器，概率质量无处可去；sink 是泄压阀，模型在"无处安放注意力"时打开它排掉多余质量，避免输出被噪声撑爆。`sink` 数值由**预训练学出**：训练中"无内容可看"的情形越多，学到的 sink 越大。

### A.3 为什么每 head 一个

64 个 head 分工不同（语法/指代/位置…），"无处可看"的时机各异，故每 head 一个独立可学习标量：

```python
# models/deepseek_v4.py:762
self.attn_sink = nn.Parameter(torch.empty(attn_sink_heads, dtype=torch.float32))
```

### A.4 对稀疏注意力（CSA）尤其重要

C4A 层先由 Indexer 挑 top-k KV 块再做 FA——候选集是"选出来的"，不保证相关。烂候选面前，普通 softmax 只能硬分配权重，sink 允许直接"弃权"。

### A.5 shape 契约（它在各并行方案里反复出现的原因）

kernel 要求 **sinks 长度 == 本次 FA 收到的 Q head 数**：

| 场景 | FA 收到的 Q | sinks 传法 |
|---|---|---|
| cann-recipes v1（PR #3） | 全 64 头（AllGather 后） | 参数改为全量 64 个 |
| cann-recipes v2（大范围 TP） | 本地 16 头 | 参数仍 64 个（checkpoint 不变），传 kernel 时切 16 个 |
| vllm-ascend 标准 TP | 本地 `n_local_heads` 头 | 参数直接按本地头数创建（:761-762） |
| 本次 qb TP | 归还后全 64 头（tp=1） | 全量 64 个，零改动 |

## 附录 B：为什么 `wq_a`（以及 `wkv`、`gate`）要每卡复制

一句话：**它们太小、消费者太多、且切了反而新增通信——切分的性价比为负。**

### B.1 体量对比：切分收益与权重大小成正比

| 矩阵 | 形状 | 参数量/层 | decode 每 token 读取（bf16） |
|---|---|---|---|
| `wq_a` | 4096×1536 | ~6.3M | ~12.6MB |
| **`wq_b`** | 1536×32768 | **~50M** | **~100MB** |
| `wkv` | 4096×512 | ~2M | ~4MB |
| `gate`（MoE） | 4096×256 | ~1M | ~2MB |

decode 时延 ≈ Σ权重读取/带宽。切 `wq_b` 省 75MB/层；切 `wq_a` 只省 ~9MB/层——不到前者的 1/8，却要付出同样的工程复杂度。MLA 的低秩分解结构本身就是"小矩阵（wq_a）+ 大矩阵（wq_b）"，**天然把值得切的部分集中到了 wq_b 一个矩阵里**。

### B.2 切 wq_a 会凭空引入通信

`wq_a` 若按输出维（1536）列切，每卡只得到 qr 的一个分片，而它的**所有消费者都需要完整 qr**：

- `wq_b`：列切形态要求输入完整 1536 维（列切 Linear 的定义：完整输入 → 分片输出）；
- Indexer：用完整 qr 做稀疏打分；
- w8a8 路径：norm+quant 融合算子作用于完整 qr。

于是每层要多一次 qr 的 AllGather（T×1536）+ 一个同步点。对比现状：`wq_a` 复制 → qr 每卡天然完整 → `wq_b` 列切的输入侧**零通信**（这正是列切的优势），只有输出侧需要一次"还 head"。**切小矩阵的代价是給大小矩阵都加上通信；切大矩阵的代价只有输出侧一次交换。**

### B.3 多消费者需要一致的本地副本

qr 同时喂 wq_b、Indexer、（w8a8 下）量化融合算子。复制计算保证：

- 三个消费者在**同一卡上零依赖**地拿到同一份 qr；
- QB TP 的正确性前提——"返回给 Indexer 的 qr 永远是本地原值"——不需要任何额外约定；
- Indexer 各卡独立算出的 top-k 索引天然一致（输入相同、计算确定），稀疏选择的跨卡一致性免费获得。

同理 `gate` 复制：MoE TP 要求**各卡路由结果完全一致**（各自对相同 token 算部分和，最后一次 AllReduce），gate 复制是这一前提的构造性保证。

### B.4 `wkv` 复制还有一层架构原因

`wkv` 输出的 KV latent 是**全部 64 头共享**的一份（MLA 本质），FA、compressor、KV cache 写入都以"每卡持有完整 KV latent"为前提。切它意味着每卡 KV cache 只有分片，FA 前还得 gather KV——把全模型最贵的数据（长序列 KV）搬进通信，完全本末倒置。它同时也是全层最小的矩阵（~2M），复制的显存代价可忽略。

### B.5 一般规律

> **复制的**：小矩阵、输出被多个消费者全量需要的矩阵、路由/索引类"必须跨卡一致"的矩阵（wq_a、wkv、gate、q_norm/kv_norm）。
> **切分的**：大矩阵、输出可按分片消费或可低成本归还/归约的矩阵（wq_b 列切、wo_a 列切 + wo_b 行切、专家 FFN W13 列切 + W2 行切、lm_head/embed 按 vocab 切）。
