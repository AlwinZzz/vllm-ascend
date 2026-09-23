# DSV4 QB TP 改动详解（代码级导读）

> **仓库/分支**：vllm-ascend `releases/v0.26.0rc`
> **功能**：新增 `finegrained_tp_config.qb_tensor_parallel_size`——DP 部署下把 DSV4 的 `wq_b` 矩阵切到多卡
> **改动规模**：7 个文件（1 个新增），+189 / -21 行
> **阅读方式**：本文按"一次启动从头到尾会发生什么"的顺序讲 6 个阶段，每处改动都给出 **改前 → 改后 → 为什么** 三段式。

---

## 0. 先用一张图看懂这次改动在干什么

**问题**：DP 部署（4 卡各跑各的请求）时，每卡的 `wq_b`（QBMM，`1536 × 32768`，约 50M 参数/层）都是完整的一份。decode 是 GEMV，时延 ≈ 读权重时间，4 张卡重复读 4 遍同样的 100MB，纯浪费。

**方案**："All_Gather 聚token → 算 1/4 → all_to_all 还head"：

```
【改前】每卡独立：
  卡0: token A 的 qr ──wq_b(完整64头权重)──→ A 的 64 头 Q      权重读 100%
  卡1: token B 的 qr ──wq_b(完整64头权重)──→ B 的 64 头 Q      权重读 100%
  （卡2/卡3 同理）

【改后】4 卡协作（以卡 0 视角）：
  ① 聚token: AllGather 把 A,B,C,D 的 qr 都收到每卡       [4, 1536]
  ② 本地算:  wq_b 只留 head0-15 分片，对 4 个 token 都算  [4, 16, 512]   权重读 25%
  ③ 还head:  all-to-all——把 B/C/D 的分片发给属主卡，
             同时收回卡1/2/3 算的 A 的 head16-63 分片     [1, 64, 512]
  之后 FA / o_proj / MoE 完全不变（A 的 Q 又变回完整 64 头）
```

**为什么划算**：每卡计算量没变（4 个 token × 16 头 ≈ 1 个 token × 64 头），但权重读取 ÷4——decode 恰恰是权重带宽 bound。代价是两次 KB~MB 级小通信。

**为什么叫"局部"TP**：只有 wq_b 一个矩阵被切，token 归属、KV cache、FA、o_proj、MoE 的 DP 语义一概不动。

---

## 1. 改动全景：按生命周期看 6 个阶段

一次带 qb TP 的推理启动，改动代码按以下顺序被触发：

```
阶段1  解析启动参数、准入校验          ascend_config.py + utils.py
阶段2  建 qbtp 通信组                  distributed/parallel_state.py
阶段3  建模型：wq_b 权重按 head 切 1/4  ops/linear_op.py + models/deepseek_v4.py(校验)
阶段4  （首次前向）惰性建交换器与静态缓冲 attention/qb_tp_exchange.py（新文件）
阶段5  每层前向：聚token → 算 → 还head   attention/dsa_v1.py（3 个调用点）
阶段6  图捕获预热：零输入跑一遍通信      attention/dsa_v1.py（profiling 分支）
```

| 阶段 | 文件 | 改动量 |
|---|---|---|
| 1 | `ascend_config.py` | +32 行（字段 + 5 条校验 + 通用校验列表） |
| 1 | `utils.py` | +4 行（`qb_tp_enable()`） |
| 2 | `distributed/parallel_state.py` | +19 行（组变量/创建/getter/销毁） |
| 3 | `ops/linear_op.py` | +39 行（新 op 类 + prefix 选择 + 类型注解） |
| 3 | `models/deepseek_v4.py` | +22 行（4 条构建期校验） |
| 4 | `attention/qb_tp_exchange.py` | **新文件 ~130 行**（交换器） |
| 5/6 | `attention/dsa_v1.py` | +94/-21 行（init 检测、3 个辅助方法、3 个调用点、profiling） |

---

## 2. 阶段 1：配置解析与准入校验

### 2.1 新增配置字段（ascend_config.py，`FinegrainedTPConfig.__init__`）

**改后**（在既有 5 个字段后追加 1 行）：

```python
self.olora_tensor_parallel_size = finegrained_tp_config.get("olora_tensor_parallel_size", 0)
self.qb_tensor_parallel_size = finegrained_tp_config.get("qb_tensor_parallel_size", 0)   # 新增
```

**为什么**：该仓所有"切 DP 的模块级 TP"（oproj/olora/lmhead/embedding/mlp）都挂在 `additional_config.finegrained_tp_config` 下，qb 作为第五个成员加入，用户侧配置方式完全一致：

```bash
--additional-config '{"finegrained_tp_config": {"qb_tensor_parallel_size": 4}}'
```

### 2.2 五条准入校验（ascend_config.py，同函数）

**改后**（新增整块；每条都对应一个"不拦就会算错/挂死"的真实冲突）：

```python
if self.qb_tensor_parallel_size > 1:
    enabled_configs.append(...)
    # ① 与标准 TP 互斥
    if vllm_config.parallel_config.tensor_parallel_size > 1:
        raise AssertionError("qb_tensor_parallel_size currently requires tensor_parallel_size == 1, ...")
    # ② 仅 graph mode
    if vllm_config.model_config and vllm_config.model_config.enforce_eager:
        raise AssertionError("qb_tensor_parallel_size is only supported in graph mode")
    # ③ 仅 PD 场景 D 节点
    if vllm_config.kv_transfer_config is None or not vllm_config.kv_transfer_config.is_kv_consumer:
        raise AssertionError("... only supported in pd scenario and can only be used in D node.")
    # ④ 与 DSA CP 互斥
    if bool(additional_config.get("enable_dsa_cp", False)):
        raise AssertionError("... mutually exclusive with DSA context parallel.")
    # ⑤ head 数整除
    if num_heads and num_heads % self.qb_tensor_parallel_size != 0:
        raise AssertionError("num_attention_heads=... must be divisible by ...")
```

| 校验 | 不拦会发生什么 |
|---|---|
| ① tp==1 | 标准 TP 已把 token 复制、head 按另一根 rank 轴切了；再叠 qb，权重分片与 token 归属在 rank 网格的两个不同轴上，无法对齐 → 静默算错 |
| ② graph mode | 静态交换缓冲依赖 ACL graph 的"捕获前预热、重放地址不变"机制；eager 的 dummy_run 不走完整 attention 模块，缓冲不会被预热 |
| ③ PD D 节点 | 静态容量 E 按 decode 桶上限设计；P 节点大 chunk prefill 会超容量（沿用 oproj TP 现行边界） |
| ④ CP 互斥 | CP 按序列维切 token 并环形交换，与 qb 的"聚 token/还 head"叠加后 token 归属彻底乱掉；且 CP 下 wq_b 本来就是 Replicated，无分片可谈 |
| ⑤ 整除 | head 不能均分则"连续分片 + 按 rank 序拼回"复原不了原始 head 编号 → FA 拿到错序的 Q |

**另**：`qb_tensor_parallel_size` 被加进 `module_tp_sizes` 列表，自动继承两条既有通用校验——"仅 MoE 模型可开 finegrained TP"、"必须整除 data_parallel_size"。

### 2.3 判定函数（utils.py）

**改后**（与四个既有函数并列同构）：

```python
def qb_tp_enable() -> bool:
    return get_ascend_config().finegrained_tp_config.qb_tensor_parallel_size > 1
```

**为什么 `> 1` 而不是 `> 0`**：组大小为 1 等于没切，直接视为关闭（与 `olora_tp_enable` 口径一致）。

---

## 3. 阶段 2：建通信组（parallel_state.py）

四处小改，全部复刻既有 finegrained 组的写法：

```python
# ① 模块级变量
_QB_TP: GroupCoordinator | None = None

# ② init_ascend_model_parallel 内：读配置 → 建组
qb_tp_size = get_ascend_config().finegrained_tp_config.qb_tensor_parallel_size
global _OTP, _LMTP, _EMBED_TP, _MLP_TP, _QB_TP        # global 列表加 _QB_TP
if qb_tp_size > 1:
    _QB_TP = _create_or_get_group(qb_tp_size, "qbtp")

# ③ 访问器
def get_qb_tp_group() -> GroupCoordinator:
    assert _QB_TP is not None, "qb tensor parallel group is not initialized"
    return _QB_TP

# ④ destroy_ascend_model_parallel 内：销毁块
```

**意义与细节**：
- `_create_or_get_group` 按 **DP rank 连续分块**建组（dp=4、qb=4 → 一个组 {0,1,2,3}；dp=8、qb=4 → 两个组 {0-3},{4-7}），与源仓 `qb_tp_group` 拓扑语义一致。
- 其内部 `_group_cache` 按 size 缓存：qb 与 oproj 同开且尺寸相同时**物理复用同一个 ProcessGroup**（拓扑相同，省一份 HCCL 资源）——这是既有机制，qb 免费获得。
- 组必须在**建模型之前**就绪：阶段 3 的权重分片要在构造 Linear 时读组内 rank。

---

## 4. 阶段 3：wq_b 权重切分（linear_op.py）

### 4.1 先理解这个仓的 Linear 补丁体系（背景知识）

该仓用 `AscendColumnParallelLinear` 替换 vLLM 的 Linear，其构造函数第一件事：

```python
self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "column", ...)
self.output_size_per_partition = divide(output_size, self.tp_size)   # 每卡权重行数 = 总量/tp_size
```

`get_parallel_op` 内部按 **prefix 字符串 + 配置开关** 选一个 op 类；op 类的 `comm_group` 属性决定 `tp_rank/tp_size` 从哪个组取——**进而决定权重加载时切哪一段、forward 时怎么算**。已有的先例：`"wo_a" in prefix and oproj_tp_enable() → DSV4OProjColumnParallelOp`（绑定 OTP 组）。

所以给 wq_b 换组 = 新增一个 op 类 + 一条 prefix 选择规则，**权重加载器、量化 scale 创建全部自动跟随**（它们都按 `output_size_per_partition` 和 `tp_rank` 工作）。

### 4.2 新增 op 类

```python
class DSV4QBColumnParallelOp(CustomColumnParallelOp):
    """Bind DSV4 wq_b weight sharding to the fine-grained QB TP group. ..."""

    @property
    def comm_group(self):
        return get_qb_tp_group()          # ← 唯一实质差异：绑定 qb 组

    def apply_impl(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = self.quant_method.apply(self.layer, input_, bias)   # 纯本地矩乘
        output_bias = self.bias if self.skip_bias_add else None
        return output_parallel, output_bias
```

**意义**：
- **权重侧**：`tp_rank = qb 组内 rank`、`output_size_per_partition = 32768/4` → weight_loader 只加载 head `[r*16,(r+1)*16)` 对应的行段，量化方法的 scale 参数也按同尺寸创建。**checkpoint 无需任何转换**（与源仓 `qb_tp_rank` 切法一致）。
- **计算侧零通信**：通信不放在 Linear 里，因为交换必须和 rope/indexer 的编排耦合（见阶段 5），放进 Linear 会把 attention 的业务逻辑漏进通用层。
- **免费获得 multistream 兼容**：`CVLinearWrapper._detect_communication` 看到非 Replicated 的 custom_op 后，`quantize()` 自动直通、`matmul()` 自动回退完整 forward——DSA 的 CV 拆分优化路径不需要任何改动。

### 4.3 prefix 选择规则

```python
# DSV4 main-model wq_b under fine-grained QB TP; the indexer keeps its own
# replicated wq_b and draft (MTP) layers stay replicated in this phase.
if "wq_b" in prefix and "indexer" not in prefix and "mtp" not in prefix and qb_tp_enable():
    return DSV4QBColumnParallelOp(layer)
```

| 排除项 | 原因 |
|---|---|
| `indexer` | Indexer 内部也有一个叫 `wq_b` 的 **ReplicatedLinear**（index head 用，语义完全不同），前缀形如 `...indexer.wq_b`，必须排除 |
| `mtp` | 设计约定：首期 draft（MTP）模型保持 wq_b 复制。draft 的 token 布局独立，且这样主/草稿模型的 collective 序列天然隔离，不会出现"一半 rank 进通信、一半不进"的死锁 |

（DSpark 模型没有 `wq_b` 命名、Kimi/V3.2 用 `q_b_proj` 命名，均已核实不受影响。）

### 4.4 模型侧校验（models/deepseek_v4.py）

三处小改，全是"把隐式假设变成显式报错"：

```python
# Attention.__init__：heads / groups 对 tp 整除（原来直接 // ，配错时报错点飘忽）
if self.n_heads % tp_size != 0:
    raise ValueError(f"num_attention_heads={self.n_heads} must be divisible by tensor_parallel_size={tp_size}.")
if self.n_groups % tp_size != 0:
    raise ValueError(f"o_groups={self.n_groups} must be divisible by tensor_parallel_size={tp_size}.")

# DeepseekV4MoE.__init__：MoE TP 形态（tp>1 且 EP 关）下的两条新校验
if self.tp_size > 1 and not parallel_config.enable_expert_parallel:
    if self.enable_eplb:
        raise ValueError("EPLB is not supported when MoE runs in tensor parallel mode ...")
    if config.moe_intermediate_size % self.tp_size != 0:
        raise ValueError("moe_intermediate_size=... must be divisible by tensor_parallel_size=...")
```

**EPLB 互斥的意义**：MoE TP 下每卡持有全部 256 个专家（各 1/4 中间维），负载天然均衡，EPLB 的冗余专家复制机制既无意义、其权重布局也与 TP 分片冲突——对齐源仓 v2 的 `force_eplb` 拒绝逻辑。

---

## 5. 阶段 4：交换器（新文件 attention/qb_tp_exchange.py）

整个特性的运行时核心，两个公开方法 + 一个单例入口。

### 5.1 `gather_tokens(qr, cos, sin)`——聚 token

```python
def gather_tokens(self, qr, cos, sin):
    exchange_num_tokens = get_potential_max_tokens()        # 静态容量 E
    num_tokens = qr.shape[0]
    if num_tokens > exchange_num_tokens:
        raise ValueError("qb static exchange capacity must cover local tokens, ...")
    return (
        self._all_gather_tokens(qr,  "qr",  exchange_num_tokens),
        self._all_gather_tokens(cos, "cos", exchange_num_tokens),
        self._all_gather_tokens(sin, "sin", exchange_num_tokens),
    )

def _all_gather_tokens(self, tensor, name, exchange_num_tokens):
    send = self._static_buf(("qb_send", name, tail, dtype), (E, *tail), ...)
    send.zero_()                       # 清掉 padding 尾部残值
    send[:num_tokens].copy_(tensor)    # 实 token 填入
    gathered = self._static_buf(("qb_recv", name, tail, dtype), (k*E, *tail), ...)
    dist.all_gather_into_tensor(gathered, send, group=get_qb_tp_group().device_group)
    return gathered                    # [k*E, ...]，rank 主序
```

**每个细节的意义**：

| 细节 | 为什么必须这样 |
|---|---|
| 三个张量一起聚 | `qr` 是 wq_b 输入；`cos/sin` 是 rope 的按 token 位置数据——聚合域里是别人的 token，必须用**它们位置**的 cos/sin 旋转。凡"按 token 行走"的数据都要跟过去；q_b_norm 权重跨 head 共享，不用动 |
| pad 到静态容量 E | AllGather 要求各卡等长；E=`get_potential_max_tokens()`（runner 初始化时一次性算好的 decode 桶上限），**所有图桶共用同一形状** → HCCL 在 graph 重放下不失步。源仓为此需要运行时 MAX 同步 + `.cpu()` + is_decode 短路 + 按层缓存四个补丁，这里全部不需要 |
| 静态缓冲、惰性分配、地址终身稳定 | ACL graph 重放要求 collective 的收发地址与捕获时一致；每次新分配会导致失步。与 `_forward_o_proj`（OTP 路径）已验证的模式一致 |
| 用 raw `dist.*` 而非 GroupCoordinator 包装 | 后者的 list 式接口每次调用分配新内存，同样破坏地址稳定性（dsa_v1 OTP 注释明确记录过这个坑） |
| 缓冲 key 含 `name` | **cos 和 sin 尾维相同**——若共用 recv 缓冲，sin 的 gather 会覆盖 cos 的结果。按名称分槽杜绝此 bug |
| `send.zero_()` | 保证 padding 行恒为零：wq_b 无 bias，零行进零行出，聚合域计算不被残值污染 |

### 5.2 `return_heads(q, num_tokens, n_local_heads)`——还 head

```python
shape = (k, E, n_local_heads, head_dim)
send = self._static_buf(...); send.copy_(q.reshape(shape))   # q: [k*E, H/k, hd] → [k, E, H/k, hd]
recv = self._static_buf(...)
dist.all_to_all_single(recv.view(-1), send.view(-1), group=...)
return recv[:, :num_tokens].transpose(0, 1) \
           .reshape(num_tokens, k * n_local_heads, head_dim).contiguous()
```

**逐步含义**（设 k=4，本卡是 rank0）：

```
send 分块:  [rank0的token行 | rank1的token行 | rank2的 | rank3的] × 本地head0-15
all-to-all: 第 j 块发给 rank j；同时收到 4 个卡发来的"我的 token"的分片
recv 布局:  [来自rank0的head0-15 | rank1的head16-31 | rank2的head32-47 | rank3的head48-63]
[:, :T_loc]: 裁掉 padding
transpose+reshape: [T_loc, 64, hd]  ← head 顺序天然正确
```

**head 顺序为什么不用重排表**：ColumnParallel 是**连续**分片，rank i 恰好持有 head `[i·16,(i+1)·16)`，所以"按来源 rank 序拼接"就是"按 head 编号拼接"。

### 5.3 单例 `get_qb_tp_exchange(k)`

所有层共享一个交换器实例（即一套静态缓冲）。**为什么安全**：层间严格串行执行，graph 重放同样按录制顺序串行，缓冲不会交叠使用。**为什么值得**：qr 缓冲（k·E·1536·2B）是大头，按层各建一套在 61 层模型上会浪费数百 MB。

---

## 6. 阶段 5：attention 前向接入（dsa_v1.py，改动最重）

### 6.1 init：按层启用检测 + W8A8 拒绝

**改后**（`AscendDSAImpl.__init__`，在 `self.attn_sink = kwargs["attn_sink"]` 之后插入）：

```python
self.qb_exchange = None
self.qb_n_local_heads = self.n_local_heads
qb_op = getattr(self.wq_b, "custom_op", None)
if isinstance(qb_op, DSV4QBColumnParallelOp):
    if _is_w8a8_dynamic(self.wq_b):
        raise ValueError("qb_tensor_parallel_size does not support W8A8-dynamic wq_b yet; ...")
    self.qb_exchange = get_qb_tp_exchange(qb_op.tp_size)
    self.qb_n_local_heads = self.num_heads // qb_op.tp_size
```

**两个关键决策**：

1. **不看全局开关，看本层权重是否真被分片**（custom_op 类型）。假如用 `qb_tp_enable()` 全局判定，MTP/DSpark 等被 prefix 规则排除、权重仍是完整 64 头的模块也会执行交换——gather 之后用完整权重算出 64 头，再按 16 头布局做 all-to-all，**shape 直接错乱或静默算错**。按层检测让"权重形态"和"forward 行为"永远一致。
2. **W8A8-dynamic 明确拒绝**而不是绕着走：该路径 `npu_quant_matmul(qr, self.wq_b.weight, ...)` 直调权重绕过 custom_op，且其 norm+quant 融合算子在**聚合之前**就把 qr 量化了——scale 属于本地 token，行序却即将变成聚合域，二者错位。**报错优于算错**；bf16 与 weight-only 量化（量化逻辑在 `quant_method.apply` 内部、作用于聚合后的输入）不受影响。

### 6.2 三个辅助方法（直通封装）

```python
def _qb_gather_input(self, qr, cos, sin):
    if self.qb_exchange is None:
        return qr, cos, sin                    # 关闭态：原对象直通，零开销
    return self.qb_exchange.gather_tokens(qr, cos, sin)

def _qb_return_heads(self, q, num_tokens):
    if self.qb_exchange is None:
        return q
    return self.qb_exchange.return_heads(q, num_tokens, self.qb_n_local_heads)

def _qb_profile_run(self, num_tokens, dtype, device):
    """零输入完整走一遍 gather → wq_b → rms → rope → return（供图捕获预热）"""
```

**意义**：三个调用点（multistream / prefill / decode）共用同一套代码，不需要每处写 if/else 双份逻辑；关闭态下这两个函数就是恒等映射——**这是"未启用时行为逐 op 等价"的结构保证**。

### 6.3 调用点 ①：`_mla_prolog_multistream`（多流路径，prefill/decode 共用）

**改前**（norm/quant 段）：

```python
if is_prefill:
    qr = self.q_norm(wq_a_result)
    q_b_quant, q_b_scale = self.cv_wq_b.quantize(qr)
    qr_pertoken_scale = None
elif is_w8a8:
    qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(...)
    q_b_quant, q_b_scale = qr, qr_pertoken_scale
else:
    qr = self.q_norm(wq_a_result)
    q_b_quant, q_b_scale = qr, None
    qr_pertoken_scale = None
```

**改后**：

```python
if is_prefill:
    qr = self.q_norm(wq_a_result)
    qr_qb, cos_qb, sin_qb = self._qb_gather_input(qr, cos, sin)   # ← 聚
    q_b_quant, q_b_scale = self.cv_wq_b.quantize(qr_qb)           # 量化作用于聚合域
    qr_pertoken_scale = None
elif is_w8a8:
    ...（不变；qb 下此分支不可达，init 已拒绝）
    cos_qb, sin_qb = cos, sin                                     # ← 别名，保证下方共享 rope 点变量有定义
else:
    qr = self.q_norm(wq_a_result)
    qr_qb, cos_qb, sin_qb = self._qb_gather_input(qr, cos, sin)
    q_b_quant, q_b_scale = qr_qb, None
    qr_pertoken_scale = None
```

**改前**（matmul 段与尾部）：

```python
q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))
...
torch.ops._C_ascend.inplace_partial_rotary_mul(q.unsqueeze(1), cos, sin, ...)
return q, qr, qr_pertoken_scale, tail_overlap_output
```

**改后**：

```python
q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.qb_n_local_heads, self.head_dim))
...
torch.ops._C_ascend.inplace_partial_rotary_mul(q.unsqueeze(1), cos_qb, sin_qb, ...)  # 聚合域的 cos/sin
q = self._qb_return_heads(q, hidden_states.shape[0])                                 # ← 还 head
return q, qr, qr_pertoken_scale, tail_overlap_output
```

**关键不变量**：返回的 `qr / qr_pertoken_scale` **永远是本地 token 的原值**——它们随后被喂给 Indexer（C4A 层）；Indexer、compressor、KV 路径对 qb 全程零感知。多流编排（kv 分支在辅流上用本地 cos/sin）也完全不受影响，gather 发生在主流上。

### 6.4 调用点 ②：`_forward_prefill` 非多流分支

**改前**：

```python
else:
    qr = self.q_norm(q_a)
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
    qr_pertoken_scale = None
q = DeviceOperator.apply_dsa_q_rms(q, ...)
torch.ops._C_ascend.inplace_partial_rotary_mul(q.unsqueeze(1), cos, sin, ...)
```

**改后**：

```python
else:
    qr = self.q_norm(q_a)
    qr_qb, cos_qb, sin_qb = self._qb_gather_input(qr, cos, sin)
    q = self.wq_b(qr_qb).unflatten(-1, (self.qb_n_local_heads, self.head_dim))
    qr_pertoken_scale = None
q = DeviceOperator.apply_dsa_q_rms(q, ...)
torch.ops._C_ascend.inplace_partial_rotary_mul(q.unsqueeze(1), cos_qb, sin_qb, ...)
q = self._qb_return_heads(q, hidden_states.shape[0])
```

（w8a8 分支只改了 unflatten 的头数变量并补 `cos_qb, sin_qb = cos, sin` 别名；qb 启用时该分支不可达。）

### 6.5 调用点 ③：`_forward_decode` 非多流分支

与 ② 同构，另有一处**顺手清理**：

**改前**：

```python
qr = q = self.q_norm(q_a)          # qr 与 q 别名同一张量
...
q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
```

**改后**：

```python
qr = self.q_norm(q_a)              # 拆开别名：qr 明确是"留给 indexer 的本地值"
qr_qb, cos_qb, sin_qb = self._qb_gather_input(qr, cos, sin)
q = self.wq_b(qr_qb).unflatten(-1, (self.qb_n_local_heads, self.head_dim))
```

语义不变（原来 q 也立即被 wq_b 输出覆盖），但消除了"qr 和 q 是不是同一个张量"的歧义——引入 qb 后 `qr`（本地）与 `q`（聚合域计算、随后归还）生命周期彻底分叉，别名会误导读者。

### 6.6 阶段 6：profiling 分支（图捕获预热）

**改前**（`forward` 中 `attn_metadata is None` 的 dummy run 分支）：

```python
if oproj_tp_enable():
    o_proj_input = torch.zeros(...)
    self._forward_o_proj(o_proj_input, output)     # OTP 的预热先例
else:
    output.fill_(0)
return output
```

**改后**（追加两行）：

```python
if self.qb_exchange is not None:
    self._qb_profile_run(forward_context.num_tokens, hidden_states.dtype, hidden_states.device)
return output
```

**意义**：ACL graph 捕获前，必须让 AllGather / all-to-all 至少真实执行一次——惰性分配静态缓冲（地址从此固定）、让 HCCL 通信子完成初始化，之后的捕获与重放才稳定。零输入走完整链路（gather → wq_b → rms → rope → return），录进图的 op 序列与真实执行完全一致。

---

## 7. 一次 decode 的完整时序（把所有改动串起来）

设 dp=4、qb=4、bs=1、MTP 关、bf16 权重，卡 0 持有 token A：

```
[启动期]  阶段1 校验通过 → 阶段2 建 qbtp 组{0,1,2,3} → 阶段3 每卡 wq_b 只装 16 头权重
[首步]    阶段6 profiling：零输入跑通两次 collective，缓冲地址定死；随后 ACL graph 捕获
[每层]    wq_a(复制) → qr [1,1536]（卡0=A，卡1=B，卡2=C，卡3=D）
          q_norm → 【AllGather】 → qr_g [4,1536]（每卡都有 A,B,C,D）
          wq_b(16头分片) → [4,16,512]        ← 本层唯一的"省"发生在这里：读 25MB 而非 100MB
          q_rms + rope(cos_g/sin_g)          ← 在聚合域做，用各 token 自己位置的 cos/sin
          【all-to-all】→ 卡0 得 A 的 [1,64,512]（其余卡同理得 B/C/D）
          Indexer(本地 qr)、FA、o_proj、MoE   ← 全部原 DP 路径，零改动
[每层通信账单]  AllGather 3×[4,1536]级 + all-to-all [4,1,16,512]级 ≈ 数十 KB（对比省下的 75MB 权重读取）
```

## 8. 关闭态行为对照（回归安全证明)

| 代码点 | `qb_tensor_parallel_size=0`（默认）时 |
|---|---|
| linear_op 选择分支 | 条件不成立，wq_b 走原生 ColumnParallel（tp=1 即不切） |
| dsa init 检测 | `custom_op` 非 QB 类型 → `qb_exchange=None`，`qb_n_local_heads ≡ n_local_heads` |
| `_qb_gather_input / _qb_return_heads` | 恒等直通，返回原对象 |
| 三个调用点 | 计算序列与原代码逐 op 等价（含 `qr = q =` 别名拆分，语义不变） |
| profiling 分支 | 新增两行被 `is not None` 短路 |
| deepseek_v4 校验 | 合法存量配置全部通过（只在整除破坏/非法组合时报错） |

即：**不开启该配置时，本 PR 对现网行为的影响为零**（新增校验的报错路径除外，且那些路径原本就会在更晚、更难诊断的位置失败）。

## 9. 明确没做/不需要做的内容

| 项 | 结论 |
|---|---|
| 设计文档 M5（quantization/method_adapters.py 的 tp_rank 注入） | **实勘后取消**：该注入只针对 RowParallelLinear（o_proj/down_proj）；wq_b 是列切，量化 scale 由 `create_weights` 按 `output_size_per_partition` 自动生成、loader 按 `output_dim` 属性分片，qb 分片下天然正确 |
| 源仓的 `get_max_tokens` 运行时同步 | **整体废弃**：静态容量 E + 框架 DP metadata 已覆盖，无 host 同步、无 graph break 风险 |
| 源仓的采样广播 | **架构免疫**：vLLM v1 采样集中在 EngineCore 单点，TP/DP 各 rank 不独立持有采样结果 |
| W8A8-dynamic 激活量化 + qb | 首期显式报错拒绝（见 6.1），激活量化需"先聚后量化"的算子顺序改造，列后续 |
| prefill 大 chunk 场景 | 超出静态容量 E 即报错（与 OTP 同边界）；"仅 decode 桶启用"开关列后续 |
| 测试（设计文档 M10） | 需 NPU 实机环境，随联调提交：交换器数值单测、dp4+qb4 vs dp4 greedy 对齐、图重放稳定性、组合矩阵 |

## 10. 验证状态

- 本机：7 文件 AST 语法检查通过；行长 ≤120（ruff 配置）；482 行完整 diff 逐行人工复核；关闭态不变性论证（§8）。
- 待实机：见 §9 测试行与设计文档 P2/P3 阶段退出标准。
