# DeepSeek-V4 并行方案迁移：vllm-ascend 改动清单与设计

> **目标仓库基线**：vllm-ascend `releases/v0.26.0rc`
> **迁移来源**：cann-recipes-infer `pr680`（PR #3《DeepSeek V4 Attention TP/MOE TP》）及后续"大范围 Attention TP"演进（源仓 commit `122b424`）
> **文档性质**：改动清单 + 详细设计（评审稿）

---

## 1. 背景与目标

源仓为 DSV4 实现了三类并行能力，用于 batch_size=1 低时延解码（GEMV 权重带宽瓶颈）与 DP 部署热点优化：

1. **路① 大范围 Attention TP + MoE TP**：token 复制到全部 rank，wq_b 按 Q head 列切、FA 本地 head × 本卡完整 KV、o_proj 按 group 切分 + AllReduce；MoE 每卡全专家、FFN 中间维切分 + 层尾 AllReduce。
2. **路② DP 下局部 QB TP（qb_tp）**：保持 DP 语义，仅切 wq_b——组内 AllGather 聚 token、本地 head 分片计算、all-to-all 还 head 给属主卡。
3. **配套**：TP 组内采样广播、token padding 对齐、MX 量化 scale 随 token 搬运、norm+量化融合算子。

迁移目标：在 vllm-ascend 中以**最小新增面**落地上述能力。经现状勘察，路① 已由 vLLM 原生 TP 等价承载（且为源方案 v2 的更优形态），**真正需要开发的只有路② 的 qb TP**；其余为校验补齐、组合验证与性能对齐。

## 2. 现状结论：已具备、无需改动的部分

| 源能力 | v0.26.0rc 既有实现 | 结论 |
|---|---|---|
| wq_b 按 head 列切 | `models/deepseek_v4.py:773`（ColumnParallelLinear）、`:747` `n_local_heads = num_attention_heads // tp_size` | 已具备（标准 TP） |
| FA 本地 head + sinks 切片 | `models/deepseek_v4.py:761-762`（attn_sink 按本地头分片）；`attention/dsa_v1.py` AscendDSAImpl 以 `n_local_heads` 组织 FA | 已具备，即源方案 v2 形态 |
| o_proj 组切 + AllReduce | `models/deepseek_v4.py:754`（n_local_groups）、`:792-812`（wo_a 列切 / wo_b RowParallel 内部归约） | 已具备 |
| MoE TP 骨架 | vLLM FusedMoE tp 分片 + `models/deepseek_v4.py:435-438` `maybe_all_reduce_tensor_model_parallel`；通信方式 `MoECommType{ALLGATHER,MC2,ALLTOALL,FUSED_MC2}`（`ascend_forward_context.py:25`） | 已具备，需组合验证（§4.9） |
| DP 下 o_proj 局部 TP | `ascend_config.py:522` `oproj_tensor_parallel_size`；`attention/dsa_v1.py:1962-2028` `_forward_o_proj`（静态缓冲 + all_to_all + reduce_scatter，ACL graph 已验证） | 已具备，qb TP 直接复用其工程模式 |
| embed/lmhead 细粒度 TP | `ascend_config.py:523-524`；`ops/vocab_parallel_embedding.py:65-67,317-321` | 已具备 |
| DP token 对齐 | `worker/model_runner_v1.py:730-745` `_sync_metadata_across_dp`（每步产出 num/max_tokens_across_dp）；`utils.py:1158` `get_potential_max_tokens()`（静态交换容量） | 已具备，qb TP 复用（源方案的运行时 MAX 同步整体废弃） |
| MTP / DSpark 载体 | `models/deepseek_v4_mtp.py`、`deepseek_v4_dspark.py`、`spec_decode/dspark_proposer.py` | 已具备，需组合验证（§6） |
| 多流 | `ascend_config.py:176` `multistream_dsv4_dsa_overlap`（默认 True） | 已具备 |
| DSA CP | `attention/context_parallel/dsa_cp.py` | 保留，与 qb TP 互斥 |

## 3. 需要改动总览

| # | 改动项 | 文件 | 类型 | 阶段 |
|---|---|---|---|---|
| M1 | finegrained 配置新增 `qb_tensor_parallel_size` + 约束校验 | `vllm_ascend/ascend_config.py` | 修改 | P2 |
| M2 | 新增 qb 通信组（创建/访问/销毁） | `vllm_ascend/distributed/parallel_state.py` | 修改 | P2 |
| M3 | 新增 `qb_tp_enable()` 判定 | `vllm_ascend/utils.py` | 修改 | P2 |
| M4 | 新增绑定 qb 组的列切 Linear 形态 | `vllm_ascend/ops/linear_op.py` | 新增类 | P2 |
| M5 | 量化 tp_rank 注入扩展至 qb 维度 | `vllm_ascend/quantization/method_adapters.py` | 修改 | P2 |
| M6 | wq_b 构造按 qb 开关分流；attention/MoE 整除与互斥校验补齐 | `vllm_ascend/models/deepseek_v4.py` | 修改 | P1/P2 |
| M7 | q 路径插入 QB 交换；profiling 空跑覆盖 qb collective | `vllm_ascend/attention/dsa_v1.py` | 修改 | P2 |
| M8 | QB 交换器（聚 token / 还 head / 静态缓冲 / decode-only 开关） | `vllm_ascend/attention/qb_tp_exchange.py` | **新增文件** | P2 |
| M9 | MoE TP（EP 关闭）× 量化方法组合对拍与补丁 | `vllm_ascend/ops/fused_moe/*`、`vllm_ascend/quantization/methods/*` | 验证/修改 | P1 |
| M10 | 单元与 e2e 测试 | `tests/ut/`、`tests/e2e/pull_request/four_card/` | 新增 | P1-P3 |

## 4. 详细设计

### 4.1 M1 配置层（ascend_config.py）

- **位置**：`FinegrainedTPConfig`（:516-575）。
- **改动**：新增字段 `qb_tensor_parallel_size`（默认 0=关闭），读取自 `additional_config.finegrained_tp_config`。
- **校验规则**（构建期集中报错，运行路径不留兜底）：
  1. `tensor_parallel_size == 1`（与标准 TP 互斥：权重分片轴与 rank 网格正交，叠加时 head 分片与 token 分布无法对齐——与既有 oproj 约束 :529-541 同理）；
  2. `data_parallel_size % qb_tp == 0` 且 `num_attention_heads % qb_tp == 0`；
  3. 仅 graph mode（沿用 :548：静态交换缓冲依赖 ACL graph 捕获，profiling 不覆盖 eager 全模块）；
  4. 仅 PD 场景 D-node（沿用 :551 现行边界，放宽列为开放问题）；
  5. 与 DSA CP（`enable_dsa_cp()`）互斥；
  6. MoE 无需附加约束：qb TP 不改变 token 分布，MoE 继续走 DP+EP；源方案"路②禁 moe_tp"在目标仓因 MoE TP 绑定全局 tp 而天然满足。

### 4.2 M2 通信组（distributed/parallel_state.py）

- **位置**：`init_ascend_model_parallel`（:21，组创建 :102-134，getter :146-161，销毁路径同文件）。
- **改动**：新增 `_QB_TP` 组——按 DP rank 连续分块、组大小 `qb_tensor_parallel_size`，创建/获取/销毁与 `_OTP`/`_LMTP` 完全同构；导出 `get_qb_tp_group()`。
- **用途边界**：qb 组仅承载 AllGather（聚 token）与 all-to-all（还 head）两类集合通信，不与既有组共享缓冲。

### 4.3 M3 工具函数（utils.py）

- **位置**：:837-855（`lmhead_tp_enable`/`oproj_tp_enable`/`olora_tp_enable`/`mlp_tp_enable` 旁）。
- **改动**：新增 `qb_tp_enable()`，语义同构（`qb_tensor_parallel_size > 1`）。

### 4.4 M4 Linear 形态（ops/linear_op.py）

- **位置**：参照 :171-246 既有"绑定 OTP/MLP 组"的 Linear 写法。
- **改动**：新增绑定 `get_qb_tp_group()` 的列切 Linear 形态，供 wq_b 使用：
  - `output_size_per_partition = n_heads * head_dim // qb_tp`；
  - `tp_rank = get_qb_tp_group().rank_in_group`；
  - weight_loader 复用 vLLM ColumnParallel 的按输出维连续分片逻辑（**head 连续分片**，与源仓 checkpoint 切分语义一致，权重文件无需转换）。

### 4.5 M5 量化适配（quantization/method_adapters.py）

- **位置**：:146-148（既有 OTP/MLP 的 tp_rank 注入分支）。
- **改动**：新增 qb 维度分支，保证 wq_b 的权重/激活 scale 按同一 head 维切分；重点覆盖 w4a8_mxfp4 / w4a4_mxfp4 / w8a8_mxfp8 在 qb 分片下的 scale 布局（源仓教训：MX scale 需匹配 GMM 期望布局，配置与真实权重不符必须加载期报错）。

### 4.6 M6 模型层（models/deepseek_v4.py）

1. **wq_b 构造分流**（:773 附近）：`qb_tp_enable()` 时使用 M4 的 qb 列切形态；`enable_dsa_cp` 分支（ReplicatedLinear）优先级保持不变。
2. **标准 TP 校验补齐**（`DeepseekV4Attention.__init__`，:747/:754 现为隐式整除假设）：
   - `num_attention_heads % tp_size == 0`、`o_groups % tp_size == 0`；
   - `moe_intermediate_size % tp_size == 0`（MoE TP 时）；
   - `enable_eplb` 与 MoE TP（tp>1 且 EP 关闭）互斥（TP 形态专家负载天然均衡，冗余专家机制无意义且权重布局冲突；对应源仓 v2 的 force_eplb 拒绝逻辑）。

### 4.7 M7 attention 后端（attention/dsa_v1.py）

- **插入点**：`AscendDSAImpl` 的 q 路径——wq_a/q_norm 之后、wq_b 之前调用交换器"聚 token"；q_b_norm/rope 之后、FA 之前调用"还 head"。FA/Indexer/o_proj/MoE 路径**零改动**（交换器归还后布局与 DP 原状完全一致）。
- **profiling 空跑**（:2060-2068）：qb 开启时，与既有 OTP 分支同构——零输入走完整交换流程，确保 AllGather/all-to-all 被 ACL graph 录制。
- **rope 数据源**：本层 cos/sin（metadata 按 token 组织）需随 token 一同进入交换器聚合（见 M8）。

### 4.8 M8 QB 交换器（新增 attention/qb_tp_exchange.py）

数据流（qb_tp=4、E=静态交换容量、T_loc=本卡 token 数）：

```
qr [T_loc,1536]（含 per-token MX scale，若激活量化开启）
 ① 聚 token：pad 到 E → 组内 AllGather → [qb_tp·E, ·]
    同步聚合：per-token scale、本层 rope cos/sin（均为"按 token 行"数据）
 ② 本地 QBMM：wq_b(1/4 head 分片) → [qb_tp·E, n_heads/qb_tp, head_dim]
    权重读取 ÷4，FLOPs 与原状持平
 ③ q_b_norm + rope（使用①聚合的 cos/sin）
 ④ 还 head：all_to_all_single（静态缓冲）
    → recv [qb_tp, E, n_heads/qb_tp, hd] → [:, :T_loc].transpose(0,1)
    → reshape [T_loc, n_heads, hd]，裁掉 padding
```

工程要点（全面继承 `_forward_o_proj` :1962-2028 已验证模式）：

1. **静态缓冲**：send/gather/a2a 缓冲首次调用（profiling run）惰性分配、地址终身稳定；集合通信使用 raw `dist` 接口写入静态缓冲（GroupCoordinator 的 list 包装每次分配新内存，graph 重放会失步——dsa_v1.py:2016-2020 注释所述原因同样适用）。
2. **静态容量取代运行时同步**：交换一律 pad 到 `get_potential_max_tokens()`（utils.py:1158）。源方案的 `get_max_tokens`（含 `.cpu()` host 同步、is_decode 短路、按层缓存）四个机制**整体废弃**——DP token 元数据已由框架每步同步（model_runner_v1.py:730-745），E 覆盖所有图桶，形状恒定。
3. **head 顺序正确性**：列切为连续 head 分片（第 r 卡持 head `[r·H/qb,(r+1)·H/qb)`），all-to-all 后按组内 rank 序转置拼接即恢复原始 head 编号，与 FA/o_proj 的"连续 head 成组"约定一致。
4. **decode-only 阶段开关**：decode（GEMV、权重带宽主导）为收益区；prefill（计算主导）下聚合使单卡 GEMM 行数放大 qb_tp 倍且通信量随 T 线性增长，可能负收益。提供"仅 decode 桶启用、prefill 桶回退复制 wq_b"开关；回退时 wq_b 双份驻留的显存代价在评审中确认，默认值由 P4 实测定。
5. **投机解码边界**：首期仅主模型 attention 启用；`deepseek_v4_mtp.py`/`deepseek_v4_dspark.py` 的 draft attention 保持 wq_b 复制，draft 路径不得进入 qb 组 collective（所有 rank 一致地不调用即安全，测试以 collective 序列打点验证）。

### 4.9 M9 MoE TP 验证补齐（ops/fused_moe、quantization/methods）

- 组合：`--tensor-parallel-size N` 且**不加** `--enable-expert-parallel` → `MoECommType.ALLGATHER` + FusedMoE tp 分片（W13 列切/SwiGLU 本地/W2 行切）+ 层尾 all_reduce（deepseek_v4.py:435-438）。
- 工作：对 DSV4 支持的每种量化方法（w4a8_mxfp4 / w4a4_mxfp4 / w8a8_mxfp8 等）逐一对拍"专家权重分片 + scale 分片/重排 + grouped matmul 调用"三要素；缺失布局转换在对应 quant method 内补齐。
- 源仓 `init_routing_v2(quant_mode=3)` + `finalize_routing(skip1)` 的单 kernel 融合形态列为二期性能优化（`ops/fused_moe/prepare_finalize.py` 增加 Ascend 专用实现），不阻塞一期正确性。

### 4.10 M10 测试

- **UT**：qb 组构建与非法配置报错矩阵（tp>1×qb、CP×qb、eager×qb、非整除）；交换器数值单测（与"本地全头 wq_b"参考实现逐元素对齐）；静态缓冲地址稳定性；量化 scale 分片正确性。
- **e2e（four_card）**：新增 `test_deepseek_v4_qb_tp.py`（dp=4+qb=4 vs dp=4 greedy 逐 token 对齐）；扩展 `test_deepseek_v4.py`（现仅 tp=4+EP，:61-62）增加 tp=4+EP 关闭形态；MTP3 组合用例。
- **跨仓金标准**：同权重同量化同 prompt 下与 cann-recipes-infer `pr680` 输出 token 级对比。
- **性能**：A5 4 卡 bs=1、8K 输入/256 输出，TPOT/TTFT 对照源仓 `1bs_tp_performance` 基线；profiler 确认 qb collective 全部被图捕获、步进内无 host 同步点。

## 5. 无需迁移项及论证

| 源机制 | 不迁移原因 |
|---|---|
| 采样/MTP 决策组内广播（源 execution_engine.py） | vLLM v1 采样与接受判定集中于 EngineCore 单点，TP worker 不持有采样结果，下一步输入经调度产物统一下发，rank 间分叉在架构上不可能；DP attention 下各 rank 独立 EngineCore、请求集不相交 |
| `get_max_tokens` 运行时 MAX 同步 + host 同步规避 | 框架已有每步 DP 元数据同步与静态交换容量（§4.8 要点 2） |
| `dense_tp` 维度 | vLLM TP 统一切分 dense MLP，为源配置（dense_tp=1）的严格超集，bs=1 下只优不劣 |
| MC2 组条件注册 | 目标仓 MoE 通信方式由 MoECommType 按 EP/TP 形态自动选择，无对应问题 |

## 6. 仅验证项（预期零代码改动）

1. 标准 TP + 温度采样：多 rank 前向 logits 一致性打点（调试开关）。
2. 标准 TP + MTP（`num_speculative_tokens=3`）：接受率与 tp=1 对齐。
3. finegrained lmhead TP 的 all-to-all（vocab_parallel_embedding.py:317-321）与 qb TP 同开时 logits 拼装正确性。
4. DSpark + 标准 TP / + qb TP 的 collective 序列跨 rank 一致性。

## 7. 分阶段计划与验收标准

| 阶段 | 内容 | 退出标准 |
|---|---|---|
| P0 | tp=4（EP 关）与 dp=4（EP 开）跑通 DSV4+MTP3 | greedy 输出与 tp=1 逐 token 一致 |
| P1 | M6 校验、M9 MoE TP 量化对拍、e2e 形态扩展 | 各量化模式精度通过；非法组合构建期报错 |
| P2 | M1-M5、M7、M8（qb TP 全量） | dp=4+qb=4 精度一致；ACL graph 重放 ≥1000 步无 HCCL 失步 |
| P3 | 组合矩阵（qb+oproj+lmhead 同开；MTP/DSpark）；§6 验证项 | 全部组合用例通过 |
| P4 | 性能对标与 decode-only 开关默认值决策；融合算子（norm+MX 量化单 kernel）评估 | bs=1 TPOT 与源仓基线差距 <10%，否则 profile 归因闭环 |

## 8. 风险与开放问题

1. **PD D-node 边界**：finegrained 体系现行仅限 PD D 节点，qb TP 首期沿用；离线单实例放开需单独评估 profiling 预热路径。
2. **prefill 收益不确定**：decode-only 开关默认值依赖 P4 实测；prefill 回退导致 wq_b 双份驻留时需评估显存预算。
3. **AllGather 的 graph 捕获**：目标仓已验证 all-to-all/reduce-scatter 捕获，qb 引入的 AllGather 形状组合需在 P2 早期以最小用例先行验证。
4. **w4a4 系列 scale 布局**：TP/qb 分片下差异较大，若对拍发现方法级缺口，P1 工作量存在上浮风险。
5. **与源仓后续演进的对齐**：源仓 v2 之后仍有 separate MLA（MLA/LI 角色分卡）等演进，本方案不纳入范围，但 qb 组拓扑设计应避免与其未来迁移冲突。
