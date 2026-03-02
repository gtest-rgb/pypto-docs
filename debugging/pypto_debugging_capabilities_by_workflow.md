# PyPTO 维测能力分析 - 按算子开发工作流三阶段

## 编译流水线

```
Tensor Graph -> Tile Graph -> Block Graph -> Execute Graph -> kernel.cpp (CCE) -> bisheng编译器 -> kernel.o
```

---

## 一、算子开发阶段

### 1.1 各子阶段所需维测能力

| 子阶段 | 核心任务 | 所需维测能力 |
|--------|----------|-------------|
| Tensor Graph | 构建高层张量计算图 | 图结构可视化、算子语义验证、Shape/Dtype检查 |
| Tile Graph | 将张量操作切分为硬件感知的Tile块 | Tile切分策略查看、内存布局分析、L0/L1缓冲区利用率预估 |
| Block Graph | 子图划分与合并优化 | 子图边界可视化、并行度分析、合并策略调试 |
| Execute Graph | 生成执行调度图 | 依赖链分析、执行顺序预览、内存分配预览 |
| CodeGen | 生成kernel.cpp (CCE代码) | CCE代码查看、指令级调试、寄存器分配验证 |

### 1.2 现有API与工具

#### 分阶段编译控制
```python
# CompStage枚举控制编译深度
from pypto import CompStage

pypto.set_host_options(compile_stage=CompStage.TENSOR_GRAPH)   # 停在Tensor Graph
pypto.set_host_options(compile_stage=CompStage.TILE_GRAPH)     # 停在Tile Graph
pypto.set_host_options(compile_stage=CompStage.EXECUTE_GRAPH)  # 停在Execute Graph
pypto.set_host_options(compile_stage=CompStage.CODEGEN_INSTRUCTION)  # 停在指令生成
pypto.set_host_options(compile_stage=CompStage.CODEGEN_BINARY)       # 编译到二进制
```

#### Pass级别调试
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

#### Tiling配置
```python
# CubeTile配置矩阵乘法切分
from pypto import CubeTile

cube_tile = CubeTile(
    m=[16, 128],      # M维度: [L0, L1]
    k=[32, 256],      # K维度: [L0, L1]
    n=[16, 64],       # N维度: [L0, L1]
    enable_multi_data_load=True,
    enable_split_k=False
)

# 应用配置
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], True, False)
pypto.set_vec_tile_shapes(64, 512)
```

### 1.3 输出产物

| 产物 | 位置 | 说明 |
|------|------|------|
| `Begin_TensorGraph.json` | `output/output_*/` | Tensor Graph阶段开始 |
| `End_TensorGraph.json` | `output/output_*/` | Tensor Graph阶段结束 |
| `End_TileGraph.json` | `output/output_*/` | Tile Graph阶段结束 |
| `End_BlockGraph.json` | `output/output_*/` | Block Graph阶段结束 |
| `*.cce` | `kernel_aicpu/` | 生成的CCE源码 |
| `*.o` | `kernel_aicpu/` | 编译后的二进制 |

### 1.4 能力差距

| 缺失能力 | 影响 |
|----------|------|
| 无交互式图可视化工具 | 难以直观理解图结构 |
| 无Tile切分冲突检测 | 可能产生非法配置 |
| 无内存使用预估 | 难以提前判断内存是否足够 |
| 无CCE代码注释/调试工具 | 难以理解生成的指令 |

---

## 二、功能调试阶段

### 2.1 核心任务

- 验证kernel执行的正确性
- 定位数值精度问题
- 对比CPU/NPU结果差异

### 2.2 现有API与工具

#### 精度验证配置
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

#### Golden数据对比
```python
# 设置Golden数据
golden_output = torch.add(input_x, input_y)
pypto.set_verify_golden_data(goldens=[None, None, golden_output])

# 执行算子时自动对比
result = my_operator(pypto.Tensor(input_x), pypto.Tensor(input_y))
```

#### 中间值打印与保存
```python
# 条件打印
pypto.pass_verify_print(tensor_a, " step=", 10)

# 循环中条件打印
for idx in pypto.loop(10):
    pypto.pass_verify_print("idx=", idx, "value=", tensor[idx], cond=(idx < 5))

# 动态命名保存
for idx in pypto.loop(10):
    pypto.pass_verify_save(tensor_out, "tensor_out_$idx", idx=idx)
```

#### CPU/NPU仿真模式
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

### 2.3 工具脚本

| 工具 | 位置 | 功能 |
|------|------|------|
| `parse_dump_tensors.py` | `tools/verifier/` | 解析和对比dump的Tensor与Golden |
| `process_dump_tensor.py` | `tools/dump_tensor/` | 处理dump的二进制Tensor文件 |

### 2.4 推荐调试流程

```
1. 小规模数据验证 (8-16元素)
   ├── CPU仿真模式运行
   └── 启用pass_verify_print查看中间值

2. 中等规模验证 (1K元素)
   ├── 设置Golden数据对比
   └── 检查精度误差

3. 边界情况验证
   ├── 极值、零值测试
   └── pass_verify_save保存问题Tensor

4. NPU上板验证
   ├── 切换到NPU模式
   └── 对比CPU/NPU结果
```

### 2.5 能力差距

| 缺失能力 | 影响 |
|----------|------|
| 无断点调试能力 | 难以逐步定位问题 |
| CPU仿真精度有限 | 可能与NPU行为不一致 |
| 无混合调试模式 | 无法部分CPU/部分NPU调试 |
| 无自动Golden生成 | 需手动编写参考实现 |

---

## 三、性能调试阶段

### 3.1 核心任务

- 分析kernel执行时间
- 识别性能瓶颈（气泡、内存带宽等）
- 优化Tiling和Pass配置

### 3.2 现有API与工具

#### 启用性能数据采集
```python
# 启用运行时调试（生成泳道图数据）
pypto.set_debug_options(runtime_debug_mode=1)
```

#### 泳道图生成
```bash
# 自动生成泳道图
./tools/gen_swimlane.sh -f python3

# 手动生成
python3 tools/profiling/draw_swim_lane.py \
    output/output_*/tilefwk_L1_prof_data.json \
    output/output_*/dyn_topo.txt \
    output/output_*/program.json
```

#### Pass性能配置
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

#### 运行时调度配置
```python
pypto.set_runtime_options(
    device_sched_mode=0,  # 0:默认 1:L2亲和性 2:公平调度
)
```

### 3.3 输出产物与分析

#### 泳道图文件
| 文件 | 说明 | 查看方式 |
|------|------|----------|
| `merged_swimlane.json` | Chrome Trace格式 | https://ui.perfetto.dev/ |
| `tilefwk_prof_data.png` | 可视化时间线 | 图片查看器 |

#### 气泡分析报告 (bubble_analysis.log)
```
[AIV_48] Execute task num:1
    Core Total Work Time: 4.86us    # 实际计算时间
    Total Wait Time: 0.0us          # 总等待时间
    Wait Schedule Time: 0.0us       # 调度等待（真正的"气泡"）
    Wait Predecessor Time: 0.0us    # 依赖等待
```

#### PMU性能计数器
```bash
# 生成PMU数据CSV
python3 tools/profiling/tilefwk_pmu_to_csv.py \
    --pmu_data_path output/output_*/aicpu.data \
    --arch dav_2201
```

**PMU事件类型：**
- Event 1: 计算单元利用率 (cube_fp16_exec, vec_fp32_exec)
- Event 2: 繁忙周期 (vec_busy_cycles, cube_busy_cycles)
- Event 4: 内存请求 (ub_read_req, l1_read_req, l2_read_req)
- Event 7: 带宽指标 (ub_read_bw, l2_write_bw)
- Event 8: 缓存命中率 (write_cache_hit, read_cache_hit)

### 3.4 Tiling自动探索工具

```bash
# 自动探索最优Tiling配置
python3 tools/scripts/tiling_tool.py config.json
```

**功能：**
- 基于硬件约束生成候选Tiling配置
- 评分算法考虑L0缓存利用率
- 支持自动探索最优K轴切分

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

### 3.6 能力差距

| 缺失能力 | 影响 |
|----------|------|
| 无自动性能回归检测 | 难以发现性能退化 |
| 无Roofline模型集成 | 难以判断优化空间 |
| Tiling工具仅支持Matmul | 其他算子需手动调优 |
| 无PMU数据可视化 | 需手动分析CSV |

---

## 四、维测能力综合对比表

### 4.1 算子开发阶段

| 所需维测能力 | 已有工具/API | 状态 | 缺失能力 | 优先级 |
|-------------|-------------|------|----------|--------|
| **分阶段编译控制** | `CompStage`, `set_host_options()` | ✅ 已有 | - | - |
| **Pass级别图导出** | `PassConfigKey`, `set_pass_config()` | ✅ 已有 | - | - |
| **Tiling配置** | `CubeTile`, `set_cube_tile_shapes()` | ✅ 已有 | - | - |
| **编译时调试** | `set_debug_options(compile_debug_mode)` | ✅ 已有 | - | - |
| **图结构可视化** | `*.json` 导出 + Toolkit | ⚠️ 部分 | 交互式图可视化工具 | P0 |
| **Tile切分冲突检测** | - | ❌ 无 | 自动检测非法配置 | P0 |
| **内存使用预估** | - | ❌ 无 | 编译阶段内存预估 | P1 |
| **CCE代码注释** | `*.cce` 源码导出 | ⚠️ 部分 | CCE代码调试工具 | P2 |
| **断点式编译调试** | `CompStage` 分阶段 | ⚠️ 部分 | 交互式断点调试 | P2 |

### 4.2 功能调试阶段

| 所需维测能力 | 已有工具/API | 状态 | 缺失能力 | 优先级 |
|-------------|-------------|------|----------|--------|
| **精度验证配置** | `set_verify_options()` | ✅ 已有 | - | - |
| **Golden数据对比** | `set_verify_golden_data()` | ✅ 已有 | - | - |
| **中间值打印** | `pass_verify_print()` | ✅ 已有 | - | - |
| **中间值保存** | `pass_verify_save()` | ✅ 已有 | - | - |
| **CPU仿真模式** | `RunMode.SIM` | ✅ 已有 | - | - |
| **NPU执行模式** | `RunMode.NPU` | ✅ 已有 | - | - |
| **Tensor解析对比** | `parse_dump_tensors.py` | ✅ 已有 | - | - |
| **断点调试** | - | ❌ 无 | 逐步定位问题能力 | P0 |
| **CPU仿真精度** | `RunMode.SIM` | ⚠️ 有限 | 提升与NPU一致性 | P0 |
| **自动Golden生成** | - | ❌ 无 | 常见模式自动生成参考实现 | P1 |
| **混合调试模式** | - | ❌ 无 | 部分CPU/部分NPU调试 | P2 |
| **错误溯源** | - | ❌ 无 | 错误追溯到源操作 | P2 |

### 4.3 性能调试阶段

| 所需维测能力 | 已有工具/API | 状态 | 缺失能力 | 优先级 |
|-------------|-------------|------|----------|--------|
| **性能数据采集** | `set_debug_options(runtime_debug_mode)` | ✅ 已有 | - | - |
| **泳道图生成** | `draw_swim_lane.py`, `gen_swimlane.sh` | ✅ 已有 | - | - |
| **气泡分析报告** | `bubble_analysis.log` | ✅ 已有 | - | - |
| **PMU计数器分析** | `tilefwk_pmu_to_csv.py` | ✅ 已有 | - | - |
| **Tiling自动探索** | `tiling_tool.py` (仅Matmul) | ⚠️ 部分 | 通用Tiling探索 | P1 |
| **Pass性能配置** | `set_pass_options()` | ✅ 已有 | - | - |
| **运行时调度配置** | `set_runtime_options()` | ✅ 已有 | - | - |
| **性能回归检测** | - | ❌ 无 | 自动对比基线性能 | P0 |
| **Roofline模型** | - | ❌ 无 | 判断优化空间 | P0 |
| **PMU数据可视化** | CSV输出 | ⚠️ 部分 | 图表化展示 | P1 |
| **实时性能预警** | - | ❌ 无 | 开发中及时发现问题 | P2 |

### 4.4 状态图例

| 符号 | 含义 |
|------|------|
| ✅ 已有 | 功能完整可用 |
| ⚠️ 部分 | 功能存在但不完善 |
| ❌ 无 | 功能缺失 |

### 4.5 优先级定义

| 优先级 | 定义 | 说明 |
|--------|------|------|
| **P0** | 高优先级 | 严重影响开发效率，需优先解决 |
| **P1** | 中优先级 | 影响部分场景，应逐步完善 |
| **P2** | 低优先级 | 锦上添花，可长期规划 |

---

## 五、缺失能力汇总（按优先级排序）

### P0 - 高优先级（严重影响开发效率）

| 阶段 | 缺失能力 | 影响 | 建议实现方式 |
|------|----------|------|-------------|
| 算子开发 | 交互式图可视化工具 | 难以直观理解IR图结构 | Web-based图探索器 |
| 算子开发 | Tile切分冲突检测 | 可能产生非法配置导致运行时错误 | 编译前静态检查 |
| 功能调试 | 断点调试能力 | 难以逐步定位问题 | 集成调试器支持 |
| 功能调试 | CPU仿真精度提升 | 可能与NPU行为不一致 | 完善仿真层实现 |
| 性能调试 | 性能回归检测 | 难以发现性能退化 | CI集成性能基线对比 |
| 性能调试 | Roofline模型集成 | 难以判断优化空间 | 集成性能分析模型 |

### P1 - 中优先级（影响部分场景）

| 阶段 | 缺失能力 | 影响 | 建议实现方式 |
|------|----------|------|-------------|
| 算子开发 | 内存使用预估 | 难以提前判断内存是否足够 | 编译阶段内存分析 |
| 功能调试 | 自动Golden生成 | 需手动编写参考实现 | 常见模式模板库 |
| 性能调试 | 通用Tiling探索 | 仅支持Matmul | 扩展tiling_tool.py |
| 性能调试 | PMU数据可视化 | 需手动分析CSV | 图表化Dashboard |

### P2 - 低优先级（锦上添花）

| 阶段 | 缺失能力 | 影响 | 建议实现方式 |
|------|----------|------|-------------|
| 算子开发 | CCE代码调试工具 | 难以理解生成的指令 | 代码注释/调试器 |
| 算子开发 | 断点式编译调试 | 难以定位具体Pass问题 | 交互式编译控制 |
| 功能调试 | 混合调试模式 | 无法部分CPU/部分NPU调试 | 分段执行控制 |
| 功能调试 | 错误溯源机制 | 难以将错误追溯到源操作 | 源码映射表 |
| 性能调试 | 实时性能预警 | 无法及时发现性能问题 | IDE集成预警 |

---

## 六、关键文件路径

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

## 七、API 使用示例

详细的 API 使用示例请参考 [API 使用示例文档](api_examples.md)，包含：

- **算子开发阶段 API**: `set_host_options()`, `CompStage`, `set_pass_config()`, `PassConfigKey`, `set_cube_tile_shapes()`, `set_vec_tile_shapes()`, `CubeTile`
- **功能调试阶段 API**: `set_verify_options()`, `set_verify_golden_data()`, `pass_verify_print()`, `pass_verify_save()`, `RunMode`, `pypto.jit`, `pypto.loop()`
- **性能调试阶段 API**: `set_debug_options()`, `set_runtime_options()`, `set_pass_options()`
- **命令行工具**: `draw_swim_lane.py`, `gen_swimlane.sh`, `tilefwk_pmu_to_csv.py`, `tiling_tool.py`, `parse_dump_tensors.py`
- **综合示例**: 完整的算子开发调试流程、性能调优示例、分阶段编译调试

---

## 八、后续改进建议

1. **统一调试面板**: 整合所有工具的Web界面
2. **交互式图探索器**: 可视化导航IR图
3. **自动Golden生成**: 为常见模式自动生成参考实现
4. **性能回归检测**: 自动对比基线性能
5. **PMU数据可视化**: 图表化展示性能计数器
