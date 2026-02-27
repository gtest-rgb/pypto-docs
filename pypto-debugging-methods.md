# PyPTO 维测手段详细总结

基于对 PyPTO 文档 (https://pypto.gitcode.com/) 及相关技术资料的深入研究，以下是 PyPTO 框架提供的所有维测手段的详细总结。

## 概述

PyPTO（发音: pai p-t-o，Python Portable Tensor Operator）是华为 CANN 推出的一款面向 AI 加速器的高效编程框架，采用 PTO（Parallel Tensor/Tile Operation）编程范式。

### 核心架构
```
Python 前端 → JIT 编译层 → pto-isa 后端 → AI Core 执行
```

### 两种运行模式
1. **完全使用 PyPTO 开发程序并运行** - 适用于独立开发和调试
2. **将 PyPTO 程序接入 GE** - 与其他 Ascend C 算子共同运行

### 关键特性
- **功能仿真** - 无需昇腾卡即可进行功能验证
- **性能仿真** - 在 CPU 环境预估 NPU 性能
- **分层抽象** - Tensor → Tile → Block → 虚拟指令集

---

## 一、功能调试

### 1.1 调试配置 API
| API | 功能描述 | 代码位置 |
|-----|---------|----------|
| `pypto.set_debug_options()` | 设置调试选项 | `python/pypto/config.py:309-338` |
| `pypto.get_debug_options()` | 获取当前调试配置 | `python/pypto/config.py:309-338` |
| `pypto.jit()` | JIT 编译配置，支持调试模式 | `python/pypto/runtime.py` |
| `pypto.options()` | 上下文管理器/装饰器配置 | `python/pypto/config.py` |
| `pypto.set_print_options()` | Tensor 打印选项配置 | `python/pypto/config.py` |

### 1.2 编译阶段控制 (CompStage)

PyPTO 支持分阶段编译调试，通过 `CompStage` 枚举控制编译深度：

```python
from pypto import CompStage

class CompStage(enum.Enum):
    ALL_COMPLETE = 0        # 完整编译（默认）
    TENSOR_GRAPH = 1        # 编译到 Tensor Graph
    TILE_GRAPH = 2          # 编译到 Tile Graph
    EXECUTE_GRAPH = 3       # 编译到 Execute Graph
    CODEGEN_INSTRUCTION = 4 # 编译到指令级
    CODEGEN_BINARY = 5      # 编译到二进制
```

**使用方法**：
```python
# 通过 set_host_options 设置编译阶段
pypto.set_host_options(compile_stage=CompStage.TENSOR_GRAPH)
```

**代码位置**: `python/pypto/config.py:21-27`

### 1.3 options() 上下文管理器

`pypto.options()` 支持两种使用模式：

#### 装饰器模式
```python
@pypto.options(pass_options={"cube_l1_reuse_setting": {-1: 4}})
def my_operator(x, y):
    return pypto.matmul(x, y)
```

#### 上下文管理器模式
```python
with pypto.options(name="test", cube_tile_shapes=[...]):
    result = my_operator(input_a, input_b)
```

### 1.4 set_print_options() Tensor 打印配置

```python
pypto.set_print_options(
    precision=4,        # 浮点数精度
    threshold=1000,     # 元素数量阈值
    edgeitems=3,        # 边缘显示元素数
    linewidth=80        # 行宽
)
```

### 1.5 主要调试功能

#### CPU 孪生调试（Twin Debug）
- **功能**: 在 CPU 环境中模拟 NPU 执行，快速定位代码逻辑问题
- **优势**: 无需 NPU 硬件即可进行功能验证
- **适用场景**: NPU 上板运行前初步定位算子问题

#### 通用调试手段
- **打印调试**: 支持在算子代码中添加打印语句
- **GDB 调试**: 支持使用 GDB 进行代码调试
- **JIT 编译调试**: 支持 JIT 编译过程中的调试

### 1.6 控制流调试工具

#### 控制流图（Control Flow Graph）
| 功能 | 描述 |
|------|------|
| 查看控制流图 | 可视化程序的控制流结构 |
| 搜索控制流图节点 | 快速定位特定代码块 |
| 跳转到代码行 | 从图形界面直接跳转到源代码 |

#### 控制流 API
| API | 功能描述 |
|-----|---------|
| `pypto.cond()` | 条件分支控制 |
| `pypto.loop()` | 循环控制 |
| `pypto.loop_unroll()` | 循环展开优化 |
| `pypto.function()` | 函数定义 |
| `pypto.is_loop_begin()` | 判断是否循环开始 |
| `pypto.is_loop_end()` | 判断是否循环结束 |

#### Dynamic Loop + SymbolicScalar
- **Dynamic Loop**: 支持动态长度循环
- **SymbolicScalar**: 符号化标量，支持动态计算
  - `is_concrete()` - 是否为具体值
  - `is_expression()` - 是否为表达式
  - `is_symbol()` - 是否为符号
  - `as_variable()` - 转换为变量

---

## 二、精度调试

### 2.1 精度验证 API
| API | 功能描述 | 代码位置 |
|-----|---------|----------|
| `pypto.set_verify_options()` | 设置精度验证选项 | `python/pypto/config.py:260-306` |
| `pypto.get_verify_options()` | 获取精度验证配置 | `python/pypto/config.py:260-306` |
| `pypto.pass_verify_print()` | 打印验证结果 | `python/pypto/op/verify.py:22-74` |
| `pypto.pass_verify_save()` | 保存验证数据到文件 | `python/pypto/op/verify.py:111-156` |
| `pypto.set_verify_golden_data()` | 设置金标准数据用于精度对比 | `python/pypto/op/verify.py` |

### 2.2 verify_options 完整参数说明

```python
pypto.set_verify_options(
    enable_pass_verify=True,    # 启用 Pass 验证
    verify_dump_path="./dump",  # Tensor 落盘路径
    compare_tolerance=1e-5,     # 比较容差
    dump_format="npy",          # 落盘格式 (npy/bin)
    enable_atomic_compare=True  # 启用原子比较
)
```

### 2.3 pass_verify_print() 详细用法

**代码位置**: `python/pypto/op/verify.py:22-74`

#### 基本用法
```python
# 打印 Tensor 信息
pypto.pass_verify_print(tensor_a)

# 打印多个值（支持 Tensor 和标量）
pypto.pass_verify_print(tensor_a, " step=", 10)

# 打印格式化字符串
pypto.pass_verify_print("Result:", tensor_out, "shape=", tensor_out.shape)
```

#### 条件打印
```python
# 带条件的打印（仅当条件为 True 时打印）
for idx in pypto.loop(10):
    pypto.pass_verify_print("idx=", idx, cond=(idx > 0))

# 打印特定索引的数据
for i in pypto.loop(N):
    pypto.pass_verify_print("value[", i, "]=", tensor[i], cond=(i < 5))
```

#### 多参数组合打印
```python
# 组合 Tensor、标量、字符串
pypto.pass_verify_print(
    "Iteration:", idx,
    "Input:", input_tensor,
    "Output:", output_tensor,
    "Error:", error_val
)
```

### 2.4 pass_verify_save() 动态文件名

**代码位置**: `python/pypto/op/verify.py:111-156`

#### 基本保存
```python
# 保存 Tensor 到文件
pypto.pass_verify_save(tensor_out, "tensor_output")
```

#### 动态命名（$placeholder 语法）
```python
# 使用 $placeholder 进行动态命名
for idx in pypto.loop(10):
    pypto.pass_verify_save(tensor_out, "tensor_out_$idx", idx=idx)

# 多个占位符
for i, j in pypto.loop(10, 10):
    pypto.pass_verify_save(
        partial_result,
        "result_$i_$j",
        i=i, j=j
    )

# 带条件的保存
for idx in pypto.loop(N):
    pypto.pass_verify_save(
        tensor_out,
        "checkpoint_$idx",
        idx=idx,
        cond=(idx % 100 == 0)  # 每100次迭代保存一次
    )
```

### 2.5 精度调试手段

#### Tensor 落盘（Dump）
- **功能**: 保存中间计算结果用于精度分析
- **配置**: 通过 `pypto.set_verify_options()` 设置落盘路径
- **用途**: 对比 CPU/NPU 执行结果，定位精度偏差
- **代码位置**: `python/pypto/op/verify.py:111-156`

#### 金标准对比（Golden Data）
```python
# 设置金标准数据示例
pypto.set_verify_golden_data(golden_tensor)
```
- **功能**: 与预期结果进行数值对比
- **支持**: 多种数据类型（FP32、FP16、BF16、INT8 等）

#### CPU/NPU 结果对比
- **CPU 仿真**: 在 CPU 环境执行计算
- **NPU 实际执行**: 在昇腾 NPU 上执行
- **对比分析**: 比较两者结果差异，定位精度问题

### 2.6 自动微分与梯度调试

#### 混合式 AD 系统
PyPTO 支持混合式自动微分：
- **符号微分** - 基于计算图的符号微分，用于图优化
- **动态反向传播** - 运行时动态反向传播，用于调试与灵活性

### 2.7 调试案例
- **ffn_shared_expert_quant 算子 NPU 上板调试案例**: 演示 NPU 实际运行时的精度调试流程

---

## 三、性能调优

### 3.1 性能分析 API

#### 运行时配置
| API | 功能描述 | 代码位置 |
|-----|---------|----------|
| `pypto.set_runtime_options()` | 设置运行时选项 | `python/pypto/config.py:195-257` |
| `pypto.get_runtime_options()` | 获取运行时配置 | `python/pypto/config.py:195-257` |

#### 运行时调试选项
```python
pypto.set_runtime_options(
    runtime_debug_mode=1,        # 启用运行时调试
    enable_profiling=True,       # 启用性能分析
    trace_level=2                # 追踪级别
)
```

#### 代码生成配置
| API | 功能描述 |
|-----|---------|
| `pypto.set_codegen_options()` | 设置代码生成选项 |
| `pypto.get_codegen_options()` | 获取代码生成配置 |

#### 编译 Pass 配置
| API | 功能描述 |
|-----|---------|
| `pypto.set_pass_options()` | 设置编译 Pass 选项 |
| `pypto.get_pass_options()` | 获取编译 Pass 配置 |
| `pypto.set_pass_config()` | 设置 Pass 配置 |
| `pypto.get_pass_config()` | 获取 Pass 配置 |
| `pypto.get_pass_configs()` | 获取所有 Pass 配置 |
| `pypto.set_pass_default_config()` | 设置默认 Pass 配置 |
| `pypto.get_pass_default_config()` | 获取默认 Pass 配置 |

### 3.2 性能数据文件说明

启用性能调试后，会在输出目录生成以下文件：

| 文件名 | 说明 |
|--------|------|
| `merged_swimlane.json` | Chrome Trace 格式泳道图，可用 chrome://tracing 打开 |
| `machine_runtime_operator_trace.json` | AICPU 控制流追踪数据 |
| `bubble_analysis.log` | 气泡分析报告，包含等待时间统计 |
| `performance_report.json` | 综合性能报告 |

### 3.3 气泡分析报告解读 (bubble_analysis.log)

#### 关键指标说明
```
[AIV_48] Execute task num:1
    Core Total Work Time: 4.86us    # 核心实际计算时间
    Total Wait Time: 0.0us          # 总等待时间
    Wait Schedule Time: 0.0us       # 调度等待时间（真正的"气泡"）
    Wait Predecessor Time: 0.0us    # 等待前驱任务完成的时间
```

#### 指标解读
| 指标 | 含义 | 优化方向 |
|------|------|----------|
| Core Total Work Time | 实际计算时间 | 优化算法复杂度 |
| Wait Schedule Time | 调度气泡 | 优化任务调度策略 |
| Wait Predecessor Time | 依赖等待 | 优化数据流依赖 |
| Total Wait Time | 总空闲时间 | 综合优化调度和依赖 |

#### 启用气泡分析
```python
pypto.set_debug_options(
    runtime_debug_mode=1  # 启用运行时调试以生成气泡分析
)
```

### 3.4 泳道图生成

#### 生成脚本
```bash
# 使用工具脚本生成泳道图
./tools/gen_swimlane.sh output/output_*

# 泳道图绘制 Python 脚本
python tools/profiling/draw_swim_lane.py --input output/ --output swimlane.html
```

**工具位置**:
- `tools/gen_swimlane.sh` - 泳道图生成脚本
- `tools/profiling/draw_swim_lane.py` - 泳道图绘制脚本

### 3.5 性能评级标准（5星体系）

| 星级 | 性能水平 | 说明 |
|------|----------|------|
| ⭐ | 基础可用 | 功能正确，性能未优化 |
| ⭐⭐ | 初步优化 | 基本无气泡，但仍有优化空间 |
| ⭐⭐⭐ | 良好 | 计算与搬运重叠较好 |
| ⭐⭐⭐⭐ | 优秀 | 接近理论性能峰值 |
| ⭐⭐⭐⭐⭐ | 极致 | 达到硬件理论极限 |

### 3.6 CubeTile 类详解

**代码位置**: `python/pypto/config.py:552-606`

CubeTile 用于配置矩阵乘法（Matmul）算子的 Tiling 策略：

```python
class CubeTile:
    def __init__(self,
                 m: List[int],           # M 维度的 Tile 大小
                 k: List[int],           # K 维度的 Tile 大小
                 n: List[int],           # N 维度的 Tile 大小
                 enable_multi_data_load: bool = False,  # 多数据加载
                 enable_split_k: bool = False):         # GM 累加
        """
        参数说明：
        m: [L0, L1] - M 维度的两级 Tile 大小
           L0: 最内层 Tile（Cube 单次计算）
           L1: 中间层 Tile（L1 缓冲区大小）
        k: [L0, L1, L2] - K 维度的三级 Tile 大小
           L2 可选，用于大 K 维度切分
        n: [L0, L1] - N 维度的两级 Tile 大小
        enable_multi_data_load: 是否启用多数据加载优化
        enable_split_k: 是否在 GM (Global Memory) 累加结果
        """
```

#### 使用示例
```python
from pypto import CubeTile

# 创建 CubeTile 配置
cube_tile = CubeTile(
    m=[16, 128],      # M: L0=16, L1=128
    k=[32, 256],      # K: L0=32, L1=256
    n=[16, 64],       # N: L0=16, L1=64
    enable_multi_data_load=True,
    enable_split_k=False
)

# 应用到算子
@pypto.options(cube_tile_shapes=[cube_tile])
def matmul_op(a, b):
    return pypto.matmul(a, b)
```

### 3.7 Tiling 配置 API

#### Tile 形状配置
| API | 功能描述 |
|-----|---------|
| `pypto.set_vec_tile_shapes()` | 设置向量化 Tile 形状 |
| `pypto.get_vec_tile_shapes()` | 获取向量化 Tile 配置 |
| `pypto.set_cube_tile_shapes()` | 设置 Cube Tile 形状 |
| `pypto.get_cube_tile_shapes()` | 获取 Cube Tile 配置 |
| `pypto.set_matrix_size()` | 设置矩阵大小 |

#### TileShape 控制策略
- **控制算子的 tile 切分策略**
- **优化带宽利用率**
- **调整块大小以提升性能**

### 3.8 性能分析工具

#### 计算图可视化
| 功能 | 描述 |
|------|------|
| 查看计算图 | 可视化 Tensor 计算依赖关系 |
| 查看健康报告 | 诊断计算图中的潜在问题 |
| 搜索计算图节点 | 快速定位特定算子 |
| 控制图层布局展示 | 调整图形显示方式 |
| 锁定计算链路 | 追踪特定数据流路径 |
| 对比计算图差异 | 比较不同版本的计算图 |
| 自定义节点颜色及节点信息 | 个性化显示 |
| 跳转到代码行 | 从图形界面直接跳转到源代码 |
| 局部搜索渲染计算图 | 聚焦特定区域分析 |

#### 泳道图（Swimlane/Timeline）
| 功能 | 描述 |
|------|------|
| 测量节点间的时间间隔 | 精确测量执行时间 |
| 按时间范围查看泳道图 | 聚焦特定时间段 |
| 设置时间观测线 | 标记关键时间点 |
| 查看性能报告 | 获取性能统计数据 |
| 设置着色模式 | 区分不同类型的操作 |
| 配置泳道图系统参数 | 自定义显示参数 |
| 泳道图跳转到计算图 | 联动分析 |

#### 泳道图诊断要点
- **识别"气泡"（Bubbles）**: 发现性能瓶颈和空闲时间
- **流水线效率评估**: 观测时间线分布
- **计算/搬运重叠分析**: 分析计算时间与数据搬运时间的重叠度

#### 三栏联动视图
同时查看：
- 控制流图
- 计算图
- 泳道图

### 3.9 性能优化关键技术

#### 流水线效率评估
- 观测时间线分布
- 分析计算时间与搬运时间的重叠度
- 优化数据流调度

#### 吞吐量瓶颈定位
- 识别性能瓶颈算子
- 分析内存带宽利用率
- 优化计算密度

#### Tiling 块大小优化
- 调整 Tile 大小以优化带宽利用率
- 平衡计算与内存访问
- 实测案例：通过 Tile 策略调整可获得 **27% 性能提升**

#### 向量化指令使用
- 检查是否正确使用向量化指令
- 优化向量化计算效率

### 3.10 性能优化案例

| 案例 | 描述 |
|------|------|
| **QuantIndexerProlog 算子性能优化案例** | 演示完整的性能调优流程 |
| **Matmul 高性能编程指导** | 矩阵乘法算子的性能优化指南 |
| **DeepSeek-V3.2-Exp Lightning Indexer** | 大模型推理算子优化实践 |

### 3.11 IR 构建与图表示机制

PyPTO 的中间表示（IR）支持：
- **依赖分析** - 分析算子间的数据依赖
- **内存规划** - 优化内存使用
- **硬件映射** - 将高层操作映射到硬件指令

---

## 四、辅助工具

### 4.1 PyPTO Toolkit VSCode 插件

#### 安装方式
1. 下载 .vsix 插件文件
2. 打开 VSCode → 扩展选项卡 → "…" → "从 VSIX 安装…"
3. 选择已下载的 .vsix 文件完成安装

#### 主要功能
- **计算图可视化查看**
- **泳道图可视化查看**
- **代码与图形联动**

### 4.2 数据准备

#### 调试数据生成
- 从 PyPTO 程序运行结果生成调试数据
- 支持多种数据格式：
  - 控制流图数据
  - 计算图数据
  - 泳道图数据

### 4.3 其他功能
- 配置管理
- 数据导入/导出
- 视图定制

---

## 五、PyPTO 开发工作流

### 5.1 完整开发流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PyPTO 算子开发工作流                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  阶段一：环境准备                                                        │
│  ├── 1. 检查 CANN 包安装                                                 │
│  │       echo ${PATH} | grep cann-8.5.0                                 │
│  ├── 2. 获取 pto-isa 源码                                                │
│  │       export PTO_TILE_LIB_CODE_PATH=./pto_isa/pto-isa/               │
│  ├── 3. 设置 NPU 设备                                                    │
│  │       export TILE_FWK_DEVICE_ID=0                                    │
│  └── 4. 验证环境                                                         │
│          npu-smi info                                                   │
│           ↓                                                             │
│  阶段二：需求分析                                                         │
│  ├── 1. 确定算子名称和数学公式                                            │
│  ├── 2. 确定输入/输出规格                                                │
│  ├── 3. 确定数据类型支持                                                 │
│  └── 4. 确定精度要求                                                     │
│           ↓                                                             │
│  阶段三：Plan 模式（复杂算子必需）                                         │
│  ├── 1. 公式 → API 映射设计                                              │
│  ├── 2. 数据流设计                                                       │
│  └── 3. 开发步骤分解                                                     │
│           ↓                                                             │
│  阶段四：开发实现                                                         │
│  ├── 1. 创建目录 custom/my_operator/                                     │
│  ├── 2. 编写 golden 函数                                                 │
│  ├── 3. 编写测试用例                                                     │
│  └── 4. 使用 @pypto.frontend.jit 实现算子                                │
│           ↓                                                             │
│  阶段五：构建安装                                                         │
│  ├── python3 build_ci.py -f python3 --disable_auto_execute              │
│  └── 或 pip install -e .                                                │
│           ↓                                                             │
│  阶段六：功能验证（渐进式）                                                │
│  ├── Level 0: 8-16 元素 → 基础功能验证                                    │
│  ├── Level 1: 1K 元素   → 典型场景验证                                    │
│  ├── Level 2: 极值/零值 → 边界情况验证                                    │
│  └── Level 3: 大数据量  → 性能验证                                        │
│           ↓                                                             │
│  阶段七：性能调优（可选）                                                  │
│  ├── 1. 启用 debug_options={"runtime_debug_mode": 1}                    │
│  ├── 2. 重新编译运行                                                     │
│  ├── 3. 分析泳道图和气泡报告                                              │
│  └── 4. 优化 Tiling 策略                                                 │
│           ↓                                                             │
│  阶段八：文档编写                                                         │
│  └── 编写 README.md（算子概述、编译运行、测试结果、已知限制）               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 环境变量说明

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `TILE_FWK_DEVICE_ID` | NPU 设备 ID | `export TILE_FWK_DEVICE_ID=0` |
| `PTO_TILE_LIB_CODE_PATH` | pto-isa 源码路径 | `export PTO_TILE_LIB_CODE_PATH=./pto_isa/pto-isa/` |
| `ASCEND_HOME_PATH` | CANN 安装路径 | 自动设置 |

### 5.3 关键命令速查

| 操作 | 命令 |
|------|------|
| 编译 whl 包 | `python3 build_ci.py -f python3 --disable_auto_execute` |
| 可编辑安装 | `pip install -e .` |
| 运行单元测试 | `python3 build_ci.py -f python3 -u` |
| 运行系统测试 | `python3 build_ci.py -f python3 -s` |
| 运行全部测试 | `python3 build_ci.py -f python3 -u -s` |
| 运行算子（NPU） | `python3 custom/my_operator/my_operator.py --run-mode npu` |
| 运行算子（CPU） | `python3 custom/my_operator/my_operator.py --run-mode cpu` |
| 生成泳道图 | `./tools/gen_swimlane.sh output/output_*` |

### 5.4 测试框架说明

#### pytest 配置
**文件**: `pytest.ini`
```ini
[pytest]
python_files = test_*.py glm_*.py deepseekv32_*.py qwen3_next_*.py
testpaths = models python/tests/ut python/tests/st
```

#### 测试命令
```bash
# 运行单元测试
python -m pytest python/tests/ut -v -n auto

# 运行系统测试（指定设备）
python -m pytest python/tests/st -v --device 0 1 2 3

# 运行特定测试文件
python -m pytest python/tests/ut/test_my_operator.py -v
```

#### 测试目录结构
```
python/tests/
├── ut/                    # 单元测试（Unit Tests）
│   └── test_*.py
└── st/                    # 系统测试（System Tests）
    └── test_*.py
```

### 5.5 Golden 数据框架

**位置**: `cmake/scripts/golden_ctrl.py`, `cmake/scripts/golden_register.py`

```python
from cmake.scripts.golden_register import GoldenRegister

# 注册 Golden 生成函数
@GoldenRegister.reg_golden_func(case_names=["test_case_1", "test_case_2"], version=0)
def generate_golden(input_tensor):
    """生成 Golden 数据用于精度对比"""
    # 计算预期结果
    expected_output = compute_expected(input_tensor)
    return expected_output
```

### 5.6 核心开发规范

1. **目录规范**: 所有自定义算子必须放在 `custom/` 目录下
2. **文件规范**: 实现代码与测试代码分开存放
3. **命名规范**: 文件名使用下划线命名法，如 `my_operator.py`
4. **调试规范**: 使用渐进式调试（Level 0 → Level 3）
5. **优先级规范**: 有 NPU 卡时必须使用 `run_mode=npu` 进行验证
6. **注释规范**: 关键算法需添加注释说明

### 5.7 相关 Skills

| Skill | 说明 | 位置 |
|-------|------|------|
| pypto-operator-develop-workflow | 开发工作流指导 | `.opencode/skills/pypto-operator-develop-workflow/SKILL.md` |
| pypto-operator-perf-autotune | 性能自动调优 | `.opencode/skills/pypto-operator-perf-autotune/SKILL.md` |

---

## 六、调试调优工作流

### 6.1 Man-In-The-Loop 工作流

PyPTO 采用人工参与的优化流程，在开箱即用与性能极致之间提供平滑过渡：

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   快速可用       │ → │   灵活调优       │ → │   深度优化       │
│ (默认配置运行)    │    │ (工具链诊断)     │    │ (极致性能)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

1. **快速可用**: 使用默认配置快速获得可运行实现
2. **灵活调优**: 通过工具链识别问题、理解性能瓶颈
3. **深度优化**: 进行深度定制，追求极致性能

### 6.2 调试流程建议

```
┌──────────────────────────────────────────────────────────────┐
│                    PyPTO 调试调优流程                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 功能正确性验证                                            │
│     ├── CPU 孪生调试                                          │
│     ├── 控制流图分析                                          │
│     └── 打印/GDB 调试                                         │
│            ↓                                                 │
│  2. 精度验证                                                  │
│     ├── 金标准数据对比                                        │
│     ├── Tensor 落盘分析                                       │
│     └── CPU/NPU 结果对比                                      │
│            ↓                                                 │
│  3. 性能分析                                                  │
│     ├── 计算图健康检查                                        │
│     ├── 泳道图分析                                            │
│     └── 瓶颈识别                                              │
│            ↓                                                 │
│  4. 性能优化                                                  │
│     ├── Tiling 策略调整                                       │
│     ├── Pass 配置优化                                         │
│     └── 向量化指令优化                                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.3 分层抽象设计

| 开发者类型 | 使用层次 | 关注点 |
|-----------|---------|--------|
| **算法开发者** | Tensor 层次 | 快速实现功能 |
| **性能专家** | Tile 层次 | 深度优化性能 |
| **系统开发者** | Block 层次 | 框架集成 |

---

## 七、相关资源

### 7.1 关键文件路径索引

#### Python API 核心文件
| 文件 | 说明 |
|------|------|
| `python/pypto/__init__.py` | 入口点，导出公共 API |
| `python/pypto/config.py` | 配置系统（CompStage、CubeTile、options 等） |
| `python/pypto/tensor.py` | Tensor 类定义 |
| `python/pypto/runtime.py` | JIT 运行时 |
| `python/pypto/_controller.py` | 流程控制（loop、cond 等） |
| `python/pypto/op/verify.py` | 验证 API（pass_verify_print/save） |

#### C++ 框架文件
| 文件 | 说明 |
|------|------|
| `framework/include/tilefwk/tilefwk.h` | 框架主入口 |
| `framework/src/passes/` | 编译 Pass 实现 |
| `framework/src/codegen/` | 代码生成模块 |

#### 工具脚本
| 文件 | 说明 |
|------|------|
| `tools/gen_swimlane.sh` | 泳道图生成脚本 |
| `tools/profiling/draw_swim_lane.py` | 泳道图绘制脚本 |
| `build_ci.py` | CI 构建脚本 |
| `cmake/scripts/golden_ctrl.py` | Golden 控制脚本 |
| `cmake/scripts/golden_register.py` | Golden 注册脚本 |

#### 配置文件
| 文件 | 说明 |
|------|------|
| `pytest.ini` | pytest 测试配置 |
| `setup.py` | Python 包配置 |

### 7.2 官方资源
| 资源 | 链接 |
|------|------|
| **PyPTO 官方文档** | https://pypto.gitcode.com/ |
| **PyPTO 官方仓库** | https://atomgit.com/cann/pypto |
| **CANN 组织** | https://atomgit.com/cann |
| **华为 Ascend 官方文档** | https://www.hiascend.com/document |
| **华为支持网站** | https://support.huawei.com |

### 7.3 技术博客与文章
| 资源 | 链接 |
|------|------|
| **CANN 推出新型面向 AI 加速器的高性能编程框架——PyPTO** | https://bbs.huaweicloud.com/blogs/471435 |
| **PyPTO 库：用 Python 优雅驾驭 CANN 指令集的"魔法棒"** | https://m.blog.csdn.net/to_mountain/article/details/157944762 |
| **PyPTO：面向AI加速器的高性能编程框架全面解析与实践指南** | https://m.blog.csdn.net/2601_95191255/article/details/157909984 |
| **深度计算编程范式：PyPTO 架构下的张量分块与并行调度优化** | https://m.blog.csdn.net/qq_51601665/article/details/157811291 |
| **PyPTO：一场"人"与"编译器"的白盒契约** | https://m.blog.csdn.net/WTYuong/article/details/156273169 |
| **CANN PyPTO 的自动微分支持与梯度图生成机制** | https://m.blog.csdn.net/2301_80026901/article/details/157843507 |
| **CANN PyPTO 编程范式的 IR 构建与图表示机制** | https://blog.csdn.net/2301_80026901/article/details/157843309 |
| **PyPTO发布了，终于可以说话啦** | https://zhuanlan.zhihu.com/p/1960020143351525796 |

### 7.4 示例代码
| 资源 | 链接 |
|------|------|
| **cann-recipes-infer 仓库** | LLM 与多模态模型推理优化样例 |
| **DeepSeek-V3.2-Exp Inference on NPU** | DeepSeek 模型 NPU 推理环境搭建指导 |

---

## 八、总结

### 维测工具全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PyPTO 维测工具全景图                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   功能调试       │  │   精度调试       │  │   性能调优       │          │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤          │
│  │ • CPU孪生调试    │  │ • 金标准对比     │  │ • 计算图分析     │          │
│  │ • 控制流图       │  │ • Tensor落盘     │  │ • 泳道图分析     │          │
│  │ • GDB调试        │  │ • CPU/NPU对比    │  │ • Tiling配置     │          │
│  │ • 打印调试       │  │ • 自动微分       │  │ • Pass优化       │          │
│  │ • JIT调试        │  │ • 梯度调试       │  │ • 向量化优化     │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│           ↓                   ↓                   ↓                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      可视化工具                                   │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  • PyPTO Toolkit VSCode 插件                                      │    │
│  │  • 三栏联动视图（控制流图 + 计算图 + 泳道图）                         │    │
│  │  • 健康报告                                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 工具分类总结

| 类别 | 主要工具/方法 | 核心API |
|------|-------------|---------|
| **功能调试** | CPU孪生调试、控制流图、GDB调试、打印调试 | `set_debug_options()`, `get_debug_options()` |
| **精度调试** | 金标准对比、Tensor落盘、CPU/NPU对比验证 | `set_verify_golden_data()`, `pass_verify_print()` |
| **性能调优** | 计算图、泳道图、Tiling配置、Pass优化 | `set_vec_tile_shapes()`, `set_pass_options()` |
| **可视化工具** | PyPTO Toolkit VSCode插件、三栏联动视图 | - |

### 关键优势

1. **完整的工具链**: 从功能验证到性能优化的全流程支持
2. **分层抽象**: 满足不同层次开发者的需求
3. **可视化诊断**: 直观的图形化分析工具
4. **CPU/NPU 一致性**: 支持 CPU 仿真，降低硬件依赖
5. **Man-In-The-Loop**: 人工参与的优化流程，平衡易用性与性能
