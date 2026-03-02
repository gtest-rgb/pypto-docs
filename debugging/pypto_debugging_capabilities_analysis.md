# PyPTO 维测能力分析 - 按算子开发工作流三阶段

本文档按照算子开发、功能调试、性能调试三个阶段，系统分析 PyPTO 框架所需的维测能力及现有实现。

## 编译流水线

```
Tensor Graph → Tile Graph → Block Graph → Execute Graph → kernel.cpp (CCE) → bisheng编译器 → kernel.o
```

---

## 一、算子开发阶段

算子开发阶段涵盖从 Python API 到生成可执行 kernel 的完整编译流程。

### 1.1 各子阶段所需维测能力

| 子阶段 | 核心任务 | 所需维测能力 |
|--------|----------|-------------|
| Tensor Graph | 构建高层张量计算图 | 图结构可视化、算子语义验证、Shape/Dtype检查 |
| Tile Graph | 将张量操作切分为硬件感知的Tile块 | Tile切分策略查看、内存布局分析、L0/L1缓冲区利用率预估 |
| Block Graph | 子图划分与合并优化 | 子图边界可视化、并行度分析、合并策略调试 |
| Execute Graph | 生成执行调度图 | 依赖链分析、执行顺序预览、内存分配预览 |
| CodeGen | 生成kernel.cpp (CCE代码) | CCE代码查看、指令级调试、寄存器分配验证 |

### 1.2 现有API与工具

#### 1.2.1 分阶段编译控制

```python
# CompStage枚举控制编译深度
from pypto import CompStage

pypto.set_host_options(compile_stage=CompStage.TENSOR_GRAPH)   # 停在Tensor Graph
pypto.set_host_options(compile_stage=CompStage.TILE_GRAPH)     # 停在Tile Graph
pypto.set_host_options(compile_stage=CompStage.EXECUTE_GRAPH)  # 停在Execute Graph
pypto.set_host_options(compile_stage=CompStage.CODEGEN_INSTRUCTION)  # 停在指令生成
pypto.set_host_options(compile_stage=CompStage.CODEGEN_BINARY)       # 编译到二进制
```

**CompStage 枚举定义：**

| 值 | 名称 | 说明 |
|----|------|------|
| 0 | ALL_COMPLETE | 完整编译并执行（默认） |
| 1 | TENSOR_GRAPH | 编译到 Tensor Graph 后停止 |
| 2 | TILE_GRAPH | 编译到 Tile Graph 后停止 |
| 3 | EXECUTE_GRAPH | 编译到 Execute Graph 后停止 |
| 4 | CODEGEN_INSTRUCTION | 编译到指令级后停止 |
| 5 | CODEGEN_BINARY | 编译到二进制后停止 |

#### 1.2.2 Pass级别调试

```python
# PassConfigKey枚举
from pypto import PassConfigKey

# 为特定Pass启用图导出
pypto.set_pass_config("PVC2_OOO", "ExpandFunction", PassConfigKey.KEY_DUMP_GRAPH, True)
pypto.set_pass_config("PVC2_OOO", "ExpandFunction", PassConfigKey.KEY_PRINT_GRAPH, True)

# 全局启用
pypto.set_pass_default_config(PassConfigKey.KEY_DUMP_GRAPH, True)
pypto.set_pass_default_config(PassConfigKey.KEY_HEALTH_CHECK, True)
```

**PassConfigKey 枚举：**

| Key | 说明 |
|-----|------|
| KEY_DUMP_GRAPH | 导出图IR到JSON文件 |
| KEY_PRINT_GRAPH | 打印图到文本文件 |
| KEY_PRE_CHECK | Pass前验证 |
| KEY_POST_CHECK | Pass后验证 |
| KEY_HEALTH_CHECK | 生成健康检查报告 |
| KEY_DISABLE_PASS | 跳过该Pass |

#### 1.2.3 Tiling配置

```python
# CubeTile配置矩阵乘法切分
from pypto import CubeTile

cube_tile = CubeTile(
    m=[16, 128],      # M维度: [L0, L1]
    k=[32, 256],      # K维度: [L0, L1]
    n=[16, 64],       # N维度: [L0, L1]
    enable_multi_data_load=True,   # 启用L1大包搬运
    enable_split_k=False           # 启用K轴多核切分
)

# 应用配置
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], True, False)
pypto.set_vec_tile_shapes(64, 512)
```

**CubeTile 参数说明：**

| 参数 | 说明 |
|------|------|
| m | M维度的Tile大小 [L0, L1]，L0为最内层，L1为中间层 |
| k | K维度的Tile大小 [L0, L1, L2]，L2可选用于大K切分 |
| n | N维度的Tile大小 [L0, L1] |
| enable_multi_data_load | 是否启用多数据加载优化 |
| enable_split_k | 是否在GM(Global Memory)累加结果 |

### 1.3 输出产物

| 产物 | 位置 | 说明 |
|------|------|------|
| `Begin_TensorGraph.json` | `output/output_*/` | Tensor Graph阶段开始 |
| `End_TensorGraph.json` | `output/output_*/` | Tensor Graph阶段结束 |
| `End_TileGraph.json` | `output/output_*/` | Tile Graph阶段结束 |
| `End_BlockGraph.json` | `output/output_*/` | Block Graph阶段结束 |
| `*.cce` | `kernel_aicpu/` | 生成的CCE源码 |
| `*.o` | `kernel_aicpu/` | 编译后的二进制 |

### 1.4 能力差距分析

| 缺失能力 | 影响 | 优先级 |
|----------|------|--------|
| 无交互式图可视化工具 | 难以直观理解图结构 | 高 |
| 无Tile切分冲突检测 | 可能产生非法配置 | 高 |
| 无内存使用预估 | 难以提前判断内存是否足够 | 中 |
| 无CCE代码注释/调试工具 | 难以理解生成的指令 | 中 |
| 无断点式编译调试 | 难以定位具体Pass问题 | 低 |

---

## 二、功能调试阶段

功能调试阶段聚焦于验证 kernel 执行的正确性。

### 2.1 核心任务

- 验证kernel执行的正确性
- 定位数值精度问题
- 对比CPU/NPU结果差异

### 2.2 现有API与工具

#### 2.2.1 精度验证配置

```python
# 启用Pass验证
pypto.set_verify_options(
    enable_pass_verify=True,           # 总开关
    pass_verify_save_tensor=True,      # 保存中间Tensor
    pass_verify_save_tensor_dir="./debug_output",
    pass_verify_pass_filter=["Pass1"], # 只验证特定Pass
    pass_verify_error_tol=[1e-3, 1e-3] # [rtol, atol]
)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| enable_pass_verify | bool | False | 所有pass_verify选项的总开关 |
| pass_verify_save_tensor | bool | False | 是否保存仿真计算数据到磁盘 |
| pass_verify_save_tensor_dir | str | `{RUNNING_DIR}/output/output_{TS}` | 检测结果和数据保存路径 |
| pass_verify_pass_filter | List[str] | [] | 需要验证的Pass名称列表，为空表示验证所有Pass |
| pass_verify_error_tol | List[float] | [1e-3, 1e-3] | 精度对比容差 [rtol, atol] |

#### 2.2.2 Golden数据对比

```python
# 设置Golden数据
golden_output = torch.add(input_x, input_y)
pypto.set_verify_golden_data(goldens=[None, None, golden_output])

# 执行算子时自动对比
result = my_operator(pypto.Tensor(input_x), pypto.Tensor(input_y))
```

**set_verify_golden_data 参数：**

| 参数 | 说明 |
|------|------|
| in_out_tensors | 算子执行时的实际输入输出Tensor |
| goldens | Golden输出数据列表，长度需与算子输入输出参数列表匹配，用None跳过特定位置 |

#### 2.2.3 中间值打印与保存

```python
# 条件打印
pypto.pass_verify_print(tensor_a, " step=", 10)

# 循环中条件打印
for idx in pypto.loop(10):
    pypto.pass_verify_print("idx=", idx, "value=", tensor[idx], cond=(idx < 5))

# 动态命名保存
for idx in pypto.loop(10):
    pypto.pass_verify_save(tensor_out, "tensor_out_$idx", idx=idx)

# 带条件的保存
for idx in pypto.loop(N):
    pypto.pass_verify_save(
        tensor_out,
        "checkpoint_$idx",
        idx=idx,
        cond=(idx % 100 == 0)  # 每100次迭代保存一次
    )
```

**pass_verify_print 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| *values | Mixed | - | 混合列表，Tensor打印紧凑格式，int打印标量值 |
| cond | int/SymbolicScalar | 1 | 打印条件，为0时不打印 |

**pass_verify_save 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| tensor | Tensor | - | 要保存的Tensor |
| fname | str/SymbolicScalar | - | 文件名模板，支持$NAME占位符 |
| cond | int/SymbolicScalar | 1 | 保存条件，为0时忽略 |
| **kwargs | Mixed | - | 占位符到标量值的映射 |

#### 2.2.4 CPU/NPU仿真模式

```python
from pypto import RunMode

# CPU仿真模式（无需NPU）
@pypto.jit(runtime_options={"run_mode": RunMode.SIM})
def my_operator(x):
    return pypto.add(x, x)

# NPU执行模式
@pypto.jit(runtime_options={"run_mode": RunMode.NPU})
def my_operator(x):
    return pypto.add(x, x)
```

**RunMode 枚举：**

| 值 | 名称 | 说明 |
|----|------|------|
| 0 | NPU | 在NPU硬件上执行 |
| 1 | SIM | 在CPU上仿真执行 |

### 2.3 工具脚本

| 工具 | 位置 | 功能 |
|------|------|------|
| `parse_dump_tensors.py` | `tools/verifier/` | 解析和对比dump的Tensor与Golden |
| `process_dump_tensor.py` | `tools/dump_tensor/` | 处理dump的二进制Tensor文件 |

### 2.4 推荐调试流程

```
Level 0: 小规模数据验证 (8-16元素)
├── CPU仿真模式运行
├── 启用pass_verify_print查看中间值
└── 验证基础功能正确性
    ↓ 通过
Level 1: 中等规模验证 (1K元素)
├── 设置Golden数据对比
├── 检查精度误差
└── 验证典型场景
    ↓ 通过
Level 2: 边界情况验证
├── 极值、零值测试
├── pass_verify_save保存问题Tensor
└── 验证边界条件
    ↓ 通过
Level 3: NPU上板验证
├── 切换到NPU模式
├── 对比CPU/NPU结果
└── 验证实际硬件执行
```

### 2.5 能力差距分析

| 缺失能力 | 影响 | 优先级 |
|----------|------|--------|
| 无断点调试能力 | 难以逐步定位问题 | 高 |
| CPU仿真精度有限 | 可能与NPU行为不一致 | 高 |
| 无混合调试模式 | 无法部分CPU/部分NPU调试 | 中 |
| 无自动Golden生成 | 需手动编写参考实现 | 中 |
| 无错误溯源机制 | 难以将错误追溯到源操作 | 低 |

---

## 三、性能调试阶段

性能调试阶段聚焦于优化 kernel 执行时间，满足性能指标要求。

### 3.1 核心任务

- 分析kernel执行时间
- 识别性能瓶颈（气泡、内存带宽等）
- 优化Tiling和Pass配置

### 3.2 现有API与工具

#### 3.2.1 启用性能数据采集

```python
# 启用运行时调试（生成泳道图数据）
pypto.set_debug_options(runtime_debug_mode=1)
```

#### 3.2.2 泳道图生成

```bash
# 自动生成泳道图
./tools/gen_swimlane.sh -f python3

# 手动生成
python3 tools/profiling/draw_swim_lane.py \
    output/output_*/tilefwk_L1_prof_data.json \
    output/output_*/dyn_topo.txt \
    output/output_*/program.json \
    --label_type=1 --time_convert_denominator=50
```

#### 3.2.3 Pass性能配置

```python
# 子图划分控制
pypto.set_pass_options(
    pg_upper_bound=100,          # 子图大小上限
    pg_lower_bound=10,           # 子图大小下限
    pg_parallel_lower_bound=4,   # 最小并行度
)

# AIV合并策略
pypto.set_pass_options(
    vec_nbuffer_mode=1,          # 0:禁用 1:自动 2:手动
    vec_nbuffer_setting={-1: 4}, # 手动设置合并数量
)

# AIC合并策略
pypto.set_pass_options(
    cube_l1_reuse_mode=1,
    cube_l1_reuse_setting={-1: 4},
    cube_nbuffer_mode=1,
)
```

**Pass配置参数说明：**

| 参数 | 说明 |
|------|------|
| pg_skip_partition | 是否跳过子图划分 |
| pg_upper_bound | 子图大小上限（合并图参数） |
| pg_lower_bound | 子图大小下限（合并图参数） |
| pg_parallel_lower_bound | 同结构子图的最小并行度 |
| mg_vec_parallel_lb | AIV同结构子图的最小并行度 |
| vec_nbuffer_mode | AIV同结构子图合并策略 |
| cube_l1_reuse_mode | GM数据重复搬运的子图合并策略 |
| cube_nbuffer_mode | AIC同结构子图合并策略 |

#### 3.2.4 运行时调度配置

```python
pypto.set_runtime_options(
    device_sched_mode=0,  # 0:默认 1:L2亲和性 2:公平调度
)
```

### 3.3 输出产物与分析

#### 3.3.1 泳道图文件

| 文件 | 说明 | 查看方式 |
|------|------|----------|
| `merged_swimlane.json` | Chrome Trace格式 | https://ui.perfetto.dev/ |
| `tilefwk_prof_data.png` | 可视化时间线 | 图片查看器 |

#### 3.3.2 气泡分析报告 (bubble_analysis.log)

```
[AIV_48] Execute task num:1
    Core Total Work Time: 4.86us    # 实际计算时间
    Total Wait Time: 0.0us          # 总等待时间
    Wait Schedule Time: 0.0us       # 调度等待（真正的"气泡"）
    Wait Predecessor Time: 0.0us    # 依赖等待
```

**指标解读：**

| 指标 | 含义 | 优化方向 |
|------|------|----------|
| Core Total Work Time | 实际计算时间 | 优化算法复杂度 |
| Wait Schedule Time | 调度气泡 | 优化任务调度策略 |
| Wait Predecessor Time | 依赖等待 | 优化数据流依赖 |
| Total Wait Time | 总空闲时间 | 综合优化调度和依赖 |

#### 3.3.3 PMU性能计数器

```bash
# 生成PMU数据CSV
python3 tools/profiling/tilefwk_pmu_to_csv.py \
    --pmu_data_path output/output_*/aicpu.data \
    --arch dav_2201
```

**PMU事件类型：**

| Event | 类型 | 指标示例 |
|-------|------|----------|
| 1 | 计算单元利用率 | cube_fp16_exec, vec_fp32_exec |
| 2 | 繁忙周期 | vec_busy_cycles, cube_busy_cycles, icache_miss |
| 4 | 内存请求 | ub_read_req, l1_read_req, l2_read_req, main_read_req |
| 5 | L0缓存请求 | l0a_read_req, l0b_read_req, l0c_read_req |
| 6 | 阻塞周期 | bankgroup_stall, bank_stall, vec_resc_conflict |
| 7 | 带宽指标 | ub_read_bw, l2_write_bw, main_mem_write_bw |
| 8 | 缓存命中率 | write_cache_hit, read_cache_hit |

**支持的架构：**
- `dav_2201` (默认)
- `dav_3510`

### 3.4 Tiling自动探索工具

```bash
# 自动探索最优Tiling配置
python3 tools/scripts/tiling_tool.py config.json
```

**功能：**
- 基于硬件约束生成候选Tiling配置
- 评分算法考虑L0缓存利用率
- 支持自动探索最优K轴切分

**硬件约束：**
```python
L1_SIZE = 524288    # 512KB
L0A_SIZE = 65536    # 64KB
L0B_SIZE = 65536    # 64KB
L0C_SIZE = 131072   # 128KB
```

### 3.5 推荐性能调优流程

```
1. 启用性能数据采集
   pypto.set_debug_options(runtime_debug_mode=1)

2. 执行算子并生成泳道图
   ./tools/gen_swimlane.sh -f python3

3. 分析气泡报告
   ├── 查看Wait Schedule Time（调度气泡）
   └── 查看Wait Predecessor Time（依赖等待）

4. 可视化分析（Perfetto）
   ├── 识别空闲核心
   └── 分析任务依赖链

5. 优化策略
   ├── Tiling调整: 使用tiling_tool.py探索
   ├── Pass配置: 调整nbuffer_mode/l1_reuse
   └── 调度策略: 调整device_sched_mode

6. 深度分析（可选）
   └── PMU数据分析内存带宽和缓存命中率
```

### 3.6 性能评级标准

| 星级 | 性能水平 | 说明 |
|------|----------|------|
| ⭐ | 基础可用 | 功能正确，性能未优化 |
| ⭐⭐ | 初步优化 | 基本无气泡，但仍有优化空间 |
| ⭐⭐⭐ | 良好 | 计算与搬运重叠较好 |
| ⭐⭐⭐⭐ | 优秀 | 接近理论性能峰值 |
| ⭐⭐⭐⭐⭐ | 极致 | 达到硬件理论极限 |

### 3.7 能力差距分析

| 缺失能力 | 影响 | 优先级 |
|----------|------|--------|
| 无自动性能回归检测 | 难以发现性能退化 | 高 |
| 无Roofline模型集成 | 难以判断优化空间 | 高 |
| Tiling工具仅支持Matmul | 其他算子需手动调优 | 中 |
| 无PMU数据可视化 | 需手动分析CSV | 中 |
| 无实时性能预警 | 无法及时发现性能问题 | 低 |

---

## 四、维测能力总览表

### 按阶段汇总

| 阶段 | API/工具 | 核心能力 | 代码位置 |
|------|----------|----------|----------|
| **算子开发** | `CompStage`, `set_host_options` | 分阶段编译控制 | `python/pypto/config.py` |
| | `PassConfigKey`, `set_pass_config` | Pass级别图导出 | `python/pypto/pass_config.py` |
| | `CubeTile`, `set_cube_tile_shapes` | Tiling配置 | `python/pypto/config.py` |
| | `set_debug_options(compile_debug_mode)` | 编译时调试 | `python/pypto/config.py` |
| **功能调试** | `set_verify_options` | 精度验证配置 | `python/pypto/config.py` |
| | `set_verify_golden_data` | Golden数据对比 | `python/pypto/runtime.py` |
| | `pass_verify_print`, `pass_verify_save` | 中间值打印/保存 | `python/pypto/op/verify.py` |
| | `RunMode.SIM/NPU` | CPU/NPU模式切换 | `python/pypto/runtime.py` |
| | `parse_dump_tensors.py` | Tensor解析对比 | `tools/verifier/` |
| **性能调试** | `set_debug_options(runtime_debug_mode)` | 性能数据采集 | `python/pypto/config.py` |
| | `draw_swim_lane.py`, `gen_swimlane.sh` | 泳道图生成 | `tools/profiling/` |
| | `bubble_analysis.log` | 气泡分析报告 | 输出产物 |
| | `tilefwk_pmu_to_csv.py` | PMU计数器分析 | `tools/profiling/` |
| | `tiling_tool.py` | Tiling自动探索 | `tools/scripts/` |
| | `set_pass_options`, `set_runtime_options` | 性能配置调优 | `python/pypto/config.py` |

---

## 五、关键文件路径

| 类别 | 文件 | 说明 |
|------|------|------|
| 配置API | `python/pypto/config.py` | CompStage, set_*_options |
| 验证工具 | `python/pypto/op/verify.py` | pass_verify_print/save |
| Pass配置 | `python/pypto/pass_config.py` | PassConfigKey枚举 |
| 运行时 | `python/pypto/runtime.py` | JIT, set_verify_golden_data |
| 泳道图 | `tools/profiling/draw_swim_lane.py` | Chrome Trace生成 |
| PMU分析 | `tools/profiling/tilefwk_pmu_to_csv.py` | PMU计数器解析 |
| Tiling | `tools/scripts/tiling_tool.py` | 自动Tiling探索 |
| Golden对比 | `tools/verifier/parse_dump_tensors.py` | Tensor对比工具 |

---

## 六、后续改进建议

### 高优先级

1. **交互式图可视化工具**: 提供可视化导航IR图的能力，帮助理解图结构
2. **Tile切分冲突检测**: 自动检测非法Tiling配置，避免运行时错误
3. **断点调试能力**: 支持逐步定位问题的调试方式
4. **自动性能回归检测**: 建立性能基线，自动发现性能退化

### 中优先级

5. **内存使用预估**: 在编译阶段预估内存需求，提前判断是否足够
6. **CPU仿真精度提升**: 提高CPU仿真与NPU行为的一致性
7. **Roofline模型集成**: 帮助判断优化空间和理论性能上限
8. **PMU数据可视化**: 图表化展示性能计数器数据

### 低优先级

9. **CCE代码注释工具**: 为生成的CCE代码添加注释，便于理解
10. **混合调试模式**: 支持部分CPU/部分NPU的调试方式
11. **自动Golden生成**: 为常见模式自动生成参考实现
12. **实时性能预警**: 在开发过程中及时发现性能问题

---

## 参考资料

- [PyPTO 官方文档](https://pypto.gitcode.com/)
- [PyPTO API文档 - Verify选项](../api/config/pypto-set_verify_options.md)
- [PyPTO API文档 - Debug选项](../api/config/pypto-set_debug_options.md)
- [PyPTO 编程指南](../tutorials/index.md)
