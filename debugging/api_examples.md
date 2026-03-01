# PyPTO 维测 API 使用示例

本文档提供 PyPTO 维测 API 和工具的完整使用示例，按算子开发工作流的三个阶段组织：算子开发、功能调试、性能调试。

---

## 一、算子开发阶段 API

### 1.1 set_host_options() - 编译阶段控制

控制编译流水线的停止阶段，用于分阶段调试编译问题。

```python
import pypto

# 停在 Tensor Graph 阶段（仅构建高层张量计算图）
pypto.set_host_options(compile_stage=pypto.CompStage.TENSOR_GRAPH)

# 停在 Tile Graph 阶段（完成 Tile 切分）
pypto.set_host_options(compile_stage=pypto.CompStage.TILE_GRAPH)

# 停在 Execute Graph 阶段（完成执行调度图生成）
pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH)

# 停在指令生成阶段（生成 CCE 代码）
pypto.set_host_options(compile_stage=pypto.CompStage.CODEGEN_INSTRUCTION)

# 编译到二进制（完整编译）
pypto.set_host_options(compile_stage=pypto.CompStage.CODEGEN_BINARY)

# 完整编译并执行（默认）
pypto.set_host_options(compile_stage=pypto.CompStage.ALL_COMPLETE)
```

**输出产物位置**：
| 阶段 | 产物文件 | 位置 |
|------|----------|------|
| Tensor Graph | `Begin_TensorGraph.json`, `End_TensorGraph.json` | `output/output_*/` |
| Tile Graph | `End_TileGraph.json` | `output/output_*/` |
| Block Graph | `End_BlockGraph.json` | `output/output_*/` |
| CodeGen | `*.cce` | `kernel_aicpu/` |
| Binary | `*.o` | `kernel_aicpu/` |

---

### 1.2 CompStage - 编译阶段枚举

`CompStage` 枚举定义了编译流水线的各个阶段。

```python
from pypto import CompStage

# 枚举值说明
CompStage.ALL_COMPLETE       = 0  # 完整编译并执行
CompStage.TENSOR_GRAPH       = 1  # 停在 Tensor Graph
CompStage.TILE_GRAPH         = 2  # 停在 Tile Graph
CompStage.EXECUTE_GRAPH      = 3  # 停在 Execute Graph
CompStage.CODEGEN_INSTRUCTION = 4  # 停在指令生成
CompStage.CODEGEN_BINARY     = 5  # 编译到二进制

# 典型使用场景：定位编译问题
def debug_compile_stage():
    # 逐步深入，定位问题出现的阶段
    for stage in [CompStage.TENSOR_GRAPH, CompStage.TILE_GRAPH,
                  CompStage.EXECUTE_GRAPH, CompStage.CODEGEN_INSTRUCTION]:
        pypto.set_host_options(compile_stage=stage)
        try:
            # 尝试编译
            result = my_kernel(input_tensor)
            print(f"Stage {stage} passed")
        except Exception as e:
            print(f"Stage {stage} failed: {e}")
            break
```

---

### 1.3 set_pass_config() - Pass 级别配置

为特定 Pass 配置调试选项，用于精细控制编译过程。

```python
from pypto import PassConfigKey

# 为特定策略的特定 Pass 启用图导出
pypto.set_pass_config(
    strategy="PVC2_OOO",          # 策略名称
    identifier="ExpandFunction",   # Pass 名称
    key=PassConfigKey.KEY_DUMP_GRAPH,  # 配置键
    value=True                     # 启用图导出
)

# 启用图打印（终端输出）
pypto.set_pass_config(
    "PVC2_OOO",
    "ExpandFunction",
    PassConfigKey.KEY_DUMP_GRAPH,
    True
)

# 查询当前配置
current_value = pypto.get_pass_config(
    "PVC2_OOO",
    "ExpandFunction",
    PassConfigKey.KEY_DUMP_GRAPH,
    default_value=False
)
print(f"Current config: {current_value}")

# 获取 Pass 的完整配置
configs = pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
print(f"dumpGraph: {configs.dumpGraph}")
print(f"printGraph: {configs.printGraph}")
print(f"healthCheck: {configs.healthCheck}")
```

---

### 1.4 set_pass_default_config() - 全局 Pass 配置

设置所有 Pass 的默认配置。

```python
from pypto import PassConfigKey

# 全局启用图导出（所有 Pass 都会导出 IR 图）
pypto.set_pass_default_config(PassConfigKey.KEY_DUMP_GRAPH, True)

# 全局启用健康检查
pypto.set_pass_default_config(PassConfigKey.KEY_HEALTH_CHECK, True)

# 查询默认配置
is_enabled = pypto.get_pass_default_config(PassConfigKey.KEY_DUMP_GRAPH, False)
print(f"Dump graph enabled: {is_enabled}")
```

**注意事项**：
- 全局配置会影响所有 Pass，可能产生大量输出
- 建议在定位具体问题时使用 `set_pass_config()` 针对特定 Pass 配置

---

### 1.5 set_cube_tile_shapes() - Cube Tile 配置

配置矩阵乘法（Cube 计算）的 Tile 切分参数。

```python
import pypto

# 基本用法：设置 M, K, N 三个维度的 Tile 大小
# 参数格式：[m, k, n] 其中每个维度为 [L0大小, L1大小]
pypto.set_cube_tile_shapes(
    m=[32, 32],    # M 维度: [L0=32, L1=32]
    k=[64, 64],    # K 维度: [L0=64, L1=64]
    n=[64, 64],    # N 维度: [L0=64, L1=64]
)

# 启用多数据加载（提高 L1 到 L0 的数据传输效率）
pypto.set_cube_tile_shapes(
    m=[32, 128],
    k=[64, 256],
    n=[128, 256],
    enable_multi_data_load=True
)

# 启用 Split-K（K 轴切分，结果在 GM 累加）
pypto.set_cube_tile_shapes(
    m=[16, 64],
    k=[32, 512],
    n=[64, 128],
    enable_multi_data_load=True,
    enable_split_k=True
)

# 在 JIT 函数中使用
@pypto.jit
def matmul_kernel(a: pypto.Tensor, b: pypto.Tensor):
    pypto.set_cube_tile_shapes([32, 32], [64, 256], [128, 128])
    return pypto.matmul(a, b, a.dtype)

# 获取当前配置
m, k, n, multi_load, split_k = pypto.get_cube_tile_shapes()
print(f"M: {m}, K: {k}, N: {n}")
print(f"Multi-data load: {multi_load}, Split-K: {split_k}")
```

**Tile 配置原则**：
- L0 Tile 受硬件缓存大小限制（L0A: 64KB, L0B: 64KB, L0C: 128KB）
- L1 Tile 决定数据复用粒度
- 较大的 Tile 通常有更好的性能，但会增加内存占用

---

### 1.6 set_vec_tile_shapes() - Vector Tile 配置

配置向量计算（Vector 计算）的 Tile 切分参数。

```python
import pypto

# 基本用法：设置每个维度的 Tile 大小
# 参数个数必须与张量维度数一致
pypto.set_vec_tile_shapes(64, 64)        # 2D 张量
pypto.set_vec_tile_shapes(1, 1, 8, 8)    # 4D 张量

# 在 JIT 函数中使用
@pypto.jit
def add_kernel(a: pypto.Tensor, b: pypto.Tensor):
    # 设置 4D 张量的 Tile 形状
    pypto.set_vec_tile_shapes(1, 1, 8, 8)
    return pypto.add(a, b)

# 获取当前配置
shapes = pypto.get_vec_tile_shapes()
print(f"Vec tile shapes: {shapes}")  # 输出: [1, 1, 8, 8]

# 不同 Tile 配置对性能的影响
@pypto.jit
def compare_tile_performance(a: pypto.Tensor, b: pypto.Tensor):
    # 小 Tile：更多循环迭代，但每次计算量小
    pypto.set_vec_tile_shapes(1, 2, 4, 128)
    out1 = pypto.add(a, b)

    # 大 Tile：更少循环迭代，每次计算量大
    pypto.set_vec_tile_shapes(2, 4, 8, 256)
    out2 = pypto.add(a, b)

    return out1, out2
```

**配置约束**：
- Tile 维度数必须与张量维度数一致
- 每个维度的 Tile 大小建议为 2 的幂次
- 有效维度数范围为 [1, 4]

---

### 1.7 CubeTile - 矩阵乘法 Tile 配置类

使用类的方式配置 Cube Tile，支持更结构化的参数管理。

```python
from pypto import CubeTile

# 创建 CubeTile 配置对象
cube_tile = CubeTile(
    m=[16, 128],      # M 维度: [L0, L1]
    k=[32, 256],      # K 维度: [L0, L1]（可以是 2 或 3 个元素）
    n=[16, 64],       # N 维度: [L0, L1]
    enable_multi_data_load=True,   # 启用多数据加载
    enable_split_k=False           # 禁用 Split-K
)

# 在 options 中使用
@pypto.options(cube_tile_shapes=cube_tile)
def matmul_kernel(a: pypto.Tensor, b: pypto.Tensor):
    return pypto.matmul(a, b, a.dtype)

# 作为上下文管理器使用
with pypto.options(cube_tile_shapes=cube_tile):
    result = pypto.matmul(input_a, input_b)

# 使用列表形式（自动转换为 CubeTile）
with pypto.options(cube_tile_shapes=[[16, 16], [256, 512, 128], [128, 128], True, False]):
    result = pypto.matmul(input_a, input_b)

# 获取底层实现
impl = cube_tile.impl()
print(f"M: {impl.m}, K: {impl.k}, N: {impl.n}")
```

---

## 二、功能调试阶段 API

### 2.1 set_verify_options() - 精度验证配置

配置精度验证选项，用于自动对比计算结果与 Golden 数据。

```python
import pypto

# 基本配置：启用精度验证
pypto.set_verify_options(
    enable_pass_verify=True,           # 总开关：启用 Pass 验证
    pass_verify_error_tol=[1e-3, 1e-3] # [rtol, atol] 相对/绝对容差
)

# 完整配置：保存中间 Tensor
pypto.set_verify_options(
    enable_pass_verify=True,
    pass_verify_save_tensor=True,      # 保存中间 Tensor 到文件
    pass_verify_save_tensor_dir="./debug_output",  # 保存目录
    pass_verify_pass_filter=["Pass1", "Pass2"],  # 只验证特定 Pass
    pass_verify_error_tol=[1e-3, 1e-3] # 容差设置
)

# 使用不同精度要求
# FP32 精度
pypto.set_verify_options(
    enable_pass_verify=True,
    pass_verify_error_tol=[1e-5, 1e-5]
)

# FP16 精度
pypto.set_verify_options(
    enable_pass_verify=True,
    pass_verify_error_tol=[1e-3, 1e-3]
)

# 获取当前配置
options = pypto.get_verify_options()
print(f"Verify enabled: {options.get('enable_pass_verify')}")
print(f"Error tolerance: {options.get('pass_verify_error_tol')}")
```

---

### 2.2 set_verify_golden_data() - Golden 数据设置

设置基准数据用于自动对比验证。

```python
import torch
import pypto

# 准备输入数据
input_x = torch.randn(1024, dtype=torch.float32)
input_y = torch.randn(1024, dtype=torch.float32)

# 使用 PyTorch 计算 Golden 结果
golden_output = torch.add(input_x, input_y)

# 设置 Golden 数据
# goldens 列表对应输出张量，None 表示不需要验证的位置
pypto.set_verify_golden_data(goldens=[golden_output])

# 多输出的情况
golden_out1 = torch.add(input_x, input_y)
golden_out2 = torch.mul(input_x, input_y)
pypto.set_verify_golden_data(goldens=[golden_out1, golden_out2])

# 部分验证：只验证第二个输出
pypto.set_verify_golden_data(goldens=[None, golden_out2])

# 在 JIT 函数中使用
@pypto.jit
def my_operator(x: pypto.Tensor, y: pypto.Tensor) -> pypto.Tensor:
    return pypto.add(x, y)

# 执行时自动对比
pypto.set_verify_options(enable_pass_verify=True, pass_verify_error_tol=[1e-3, 1e-3])
pypto.set_verify_golden_data(goldens=[golden_output])
result = my_operator(pypto.Tensor(input_x), pypto.Tensor(input_y))
# 如果结果与 Golden 差异超过容差，会抛出异常
```

---

### 2.3 pass_verify_print() - 中间值打印

在算子执行过程中条件打印中间 Tensor 值，用于调试。

```python
import pypto

# 基本用法：打印 Tensor
pypto.pass_verify_print(tensor_a, " label=", "value")

# 带条件的打印
for idx in pypto.loop(10):
    # 只在前 5 次迭代打印
    pypto.pass_verify_print("idx=", idx, " value=", tensor[idx], cond=(idx < 5))

# 打印多个值
pypto.pass_verify_print(
    "step=", step_idx,
    " input=", input_tensor,
    " output=", output_tensor
)

# 在循环中使用
@pypto.jit
def debug_kernel(x: pypto.Tensor):
    result = x
    for i in pypto.loop(10):
        result = pypto.add(result, x)
        # 每次迭代都打印中间结果
        pypto.pass_verify_print("iteration=", i, " result=", result)
    return result

# 使用 SymbolicScalar 作为条件
for idx in pypto.loop(100):
    pypto.pass_verify_print(
        "idx=", idx,
        " tensor=", tensor[idx],
        cond=(idx == 0) | (idx == 99)  # 只打印首尾
    )
```

**注意事项**：
- 此 API 仅用于调试，不影响计算结果
- 打印发生在 Pass 验证阶段
- `cond` 参数为 0 时不打印，非 0 时打印

---

### 2.4 pass_verify_save() - 中间值保存

将中间 Tensor 保存到文件，用于后续分析。

```python
import pypto

# 基本用法：保存 Tensor 到文件
pypto.pass_verify_save(tensor, "output_tensor")

# 使用动态文件名
for idx in pypto.loop(10):
    pypto.pass_verify_save(
        tensor_out,
        "tensor_out_$idx",  # $idx 会被替换为实际值
        idx=idx
    )
    # 生成文件：tensor_out_0, tensor_out_1, ...

# 带条件保存
for idx in pypto.loop(100):
    # 只保存每 10 次的结果
    pypto.pass_verify_save(
        tensor,
        "checkpoint_$idx",
        idx=idx,
        cond=(idx % 10 == 0)
    )

# 多变量文件名
for batch in pypto.loop(8):
    for head in pypto.loop(12):
        pypto.pass_verify_save(
            attention_output,
            "attn_b${batch}_h${head}",
            batch=batch,
            head=head
        )

# 在复杂算子中使用
@pypto.jit
def layer_norm_debug(x: pypto.Tensor, weight: pypto.Tensor, bias: pypto.Tensor):
    # 计算均值
    mean = pypto.reduce_mean(x, axis=-1)
    pypto.pass_verify_save(mean, "step1_mean")

    # 计算方差
    var = pypto.reduce_mean(pypto.mul(x - mean, x - mean), axis=-1)
    pypto.pass_verify_save(var, "step2_variance")

    # 归一化
    x_norm = (x - mean) / pypto.sqrt(var + 1e-5)
    pypto.pass_verify_save(x_norm, "step3_normalized")

    # 仿射变换
    output = x_norm * weight + bias
    pypto.pass_verify_save(output, "step4_output")

    return output
```

**文件保存位置**：
- 由 `set_verify_options(pass_verify_save_tensor_dir="...")` 指定
- 默认保存在 `output/output_*/` 目录下

---

### 2.5 RunMode - 执行模式枚举

定义算子的执行模式：NPU 实际执行或 CPU 仿真执行。

```python
from pypto import RunMode

# 枚举值
RunMode.NPU = 0  # 在 NPU 上执行（需要硬件环境）
RunMode.SIM = 1  # CPU 仿真执行（无需 NPU）

# 在 JIT 装饰器中指定
@pypto.jit(runtime_options={"run_mode": RunMode.NPU})
def npu_kernel(x: pypto.Tensor):
    return pypto.add(x, x)

@pypto.jit(runtime_options={"run_mode": RunMode.SIM})
def sim_kernel(x: pypto.Tensor):
    return pypto.add(x, x)

# 动态选择执行模式
import os

def get_run_mode():
    """根据环境自动选择执行模式"""
    if os.environ.get("ASCEND_HOME_PATH"):
        return RunMode.NPU
    return RunMode.SIM

@pypto.jit(runtime_options={"run_mode": get_run_mode()})
def auto_kernel(x: pypto.Tensor):
    return pypto.add(x, x)

# 使用 frontend.jit
@pypto.frontend.jit(runtime_options={"run_mode": RunMode.SIM})
def frontend_sim_kernel(x: pypto.Tensor):
    return pypto.add(x, x)
```

**使用场景**：
- `RunMode.SIM`：开发环境无 NPU 时进行功能验证
- `RunMode.NPU`：生产环境进行实际计算

---

### 2.6 pypto.jit - JIT 编译装饰器

将 Python 函数编译为可在 NPU 上执行的高性能算子。

```python
import pypto

# 基本用法
@pypto.jit
def simple_add(x: pypto.Tensor, y: pypto.Tensor):
    return pypto.add(x, y)

# 带运行时配置
@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def sim_add(x: pypto.Tensor, y: pypto.Tensor):
    return pypto.add(x, y)

# 带验证配置
@pypto.jit(
    verify_options={"enable_pass_verify": True, "pass_verify_error_tol": [1e-3, 1e-3]},
    debug_options={"runtime_debug_mode": 1}
)
def verified_add(x: pypto.Tensor, y: pypto.Tensor):
    return pypto.add(x, y)

# 带 Tiling 配置
@pypto.jit
def matmul_with_tiling(a: pypto.Tensor, b: pypto.Tensor):
    pypto.set_cube_tile_shapes([32, 32], [64, 256], [128, 128])
    return pypto.matmul(a, b, a.dtype)

# 带 Pass 配置
@pypto.jit(pass_options={"pg_upper_bound": 100})
def optimized_kernel(x: pypto.Tensor):
    return pypto.relu(x)

# 完整配置示例
@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
    host_options={"compile_stage": pypto.CompStage.ALL_COMPLETE},
    pass_options={"vec_nbuffer_mode": 1},
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={"enable_pass_verify": True},
    debug_options={"runtime_debug_mode": 1}
)
def full_config_kernel(x: pypto.Tensor):
    return pypto.add(x, x)

# 使用 frontend.jit（支持动态形状）
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def dynamic_kernel(x: pypto.Tensor):
    return pypto.add(x, x)
```

---

### 2.7 pypto.loop() - 循环控制

在 JIT 编译的算子中创建可符号化的循环结构。

```python
import pypto

# 基本用法：range(stop)
for idx in pypto.loop(10):
    tensor[idx] = pypto.add(input[idx], bias)

# 完整用法：range(start, stop, step)
for idx in pypto.loop(0, 100, 2):
    result[idx] = pypto.mul(input[idx], scale)

# 带命名的循环（便于调试）
for idx in pypto.loop(10, name="LOOP_L0_bIdx", idx_name="bIdx"):
    result[idx] = pypto.add(input[idx], bias)

# 嵌套循环
for batch in pypto.loop(8, name="batch_loop"):
    for head in pypto.loop(12, name="head_loop"):
        attention[batch, head] = compute_attention(q[batch, head], k[batch, head])

# 使用循环索引进行条件判断
for idx in pypto.loop(100):
    if pypto.cond(pypto.is_loop_begin(idx)):
        # 第一次迭代的特殊处理
        result[idx] = pypto.add(input[idx], init_value)
    elif pypto.cond(pypto.is_loop_end(idx)):
        # 最后一次迭代的特殊处理
        result[idx] = pypto.add(input[idx], final_value)
    else:
        result[idx] = input[idx]

# 循环展开
for idx, unroll_factor in pypto.loop_unroll(0, 64, 1, unroll_list=[4, 2, 1]):
    # unroll_factor 会根据 unroll_list 变化
    # 优先使用 4 展开，然后 2，最后 1
    result[idx] = compute(input[idx])

# 在 JIT 函数中使用
@pypto.jit
def batch_process(x: pypto.Tensor):
    batch_size = 8
    result = pypto.tensor(x.shape, x.dtype)

    for b in pypto.loop(batch_size, name="batch"):
        chunk = x[b * 64 : (b + 1) * 64, :]
        result[b * 64 : (b + 1) * 64, :] = pypto.add(chunk, chunk)

    return result
```

---

## 三、性能调试阶段 API

### 3.1 set_debug_options() - 调试选项配置

配置编译时和运行时的调试选项。

```python
import pypto

# 启用运行时调试（生成泳道图数据）
pypto.set_debug_options(runtime_debug_mode=1)

# 启用编译时调试
pypto.set_debug_options(compile_debug_mode=1)

# 同时启用
pypto.set_debug_options(
    compile_debug_mode=1,
    runtime_debug_mode=1
)

# 在 JIT 函数中使用
@pypto.jit(debug_options={"runtime_debug_mode": 1})
def profiled_kernel(x: pypto.Tensor):
    return pypto.add(x, x)

# 使用 options 上下文管理器
with pypto.options(debug_options={"runtime_debug_mode": 1}):
    result = my_kernel(input_tensor)

# 获取当前配置
options = pypto.get_debug_options()
print(f"Runtime debug: {options.get('runtime_debug_mode')}")
print(f"Compile debug: {options.get('compile_debug_mode')}")
```

**输出产物**：
- `runtime_debug_mode=1` 会生成：
  - `tilefwk_L1_prof_data.json` - 性能数据
  - `dyn_topo.txt` - 任务拓扑
  - `program.json` - 程序信息

---

### 3.2 set_runtime_options() - 运行时配置

配置运行时调度和内存管理选项。

```python
import pypto

# 调度模式配置
pypto.set_runtime_options(
    device_sched_mode=0  # 0: 默认调度, 1: L2 亲和性, 2: 公平调度
)

# 内存配置
pypto.set_runtime_options(
    stitch_function_inner_memory=1024,     # 内部内存池大小
    stitch_function_outcast_memory=2048,   # 外部内存池大小
    stitch_cfgcache_size=10000             # 控制流缓存大小
)

# 任务调度配置
pypto.set_runtime_options(
    stitch_function_num_initial=10,   # 首次提交任务数
    stitch_function_num_step=5,       # 循环处理任务数
    stitch_function_size=100          # 每循环最大计算量
)

# 流调度
pypto.set_runtime_options(
    triple_stream_sched=True  # 启用三流调度
)

# 动态形状优化
pypto.set_runtime_options(
    valid_shape_optimize=1  # 动态 validShape 优化
)

# 在 JIT 中使用
@pypto.jit(runtime_options={
    "device_sched_mode": 1,
    "stitch_cfgcache_size": 100000000
})
def optimized_kernel(x: pypto.Tensor):
    return pypto.add(x, x)

# 获取当前配置
options = pypto.get_runtime_options()
print(f"Schedule mode: {options.get('device_sched_mode')}")
```

---

### 3.3 set_pass_options() - Pass 性能配置

配置 Pass 阶段的性能优化选项。

```python
import pypto

# 子图划分控制
pypto.set_pass_options(
    pg_skip_partition=False,        # 是否跳过子图划分
    pg_upper_bound=100,             # 子图大小上限
    pg_lower_bound=10,              # 子图大小下限
    pg_parallel_lower_bound=4       # 最小并行度
)

# AIV (Vector) 合并策略
pypto.set_pass_options(
    vec_nbuffer_mode=1,             # 0: 禁用, 1: 自动, 2: 手动
    vec_nbuffer_setting={-1: 4}     # 手动设置：-1 表示所有子图合并 4 个
)

# AIC (Cube) 合并策略
pypto.set_pass_options(
    cube_l1_reuse_mode=1,           # L1 复用模式
    cube_l1_reuse_setting={-1: 4},  # L1 复用设置
    cube_nbuffer_mode=1,            # Cube nbuffer 模式
    cube_nbuffer_setting={-1: 4}    # Cube nbuffer 设置
)

# 其他优化
pypto.set_pass_options(
    mg_copyin_upper_bound=50,  # 合并图拷入上限
    sg_set_scope=1             # 子图作用域设置
)

# 在 JIT 中使用
@pypto.jit(pass_options={
    "pg_upper_bound": 100,
    "vec_nbuffer_mode": 1,
    "cube_l1_reuse_mode": 1
})
def optimized_matmul(a: pypto.Tensor, b: pypto.Tensor):
    return pypto.matmul(a, b, a.dtype)

# 获取当前配置
options = pypto.get_pass_options()
print(f"PG upper bound: {options.get('pg_upper_bound')}")
```

---

## 四、命令行工具

### 4.1 draw_swim_lane.py - 泳道图生成

将性能数据转换为 Chrome Trace 格式，用于可视化分析。

```bash
# 基本用法
python3 tools/profiling/draw_swim_lane.py \
    output/output_1/tilefwk_L1_prof_data.json \
    output/output_1/dyn_topo.txt \
    output/output_1/program.json

# 带参数的用法
python3 tools/profiling/draw_swim_lane.py \
    output/output_1/tilefwk_L1_prof_data.json \
    output/output_1/dyn_topo.txt \
    output/output_1/program.json \
    --label_type=1 \
    --time_convert_denominator=50

# 生成可执行拓扑 JSON
python3 tools/profiling/draw_swim_lane.py \
    tilefwk_L1_prof_data.json \
    dyn_topo.txt \
    program.json \
    --gen_exe_topo_json

# 参数说明
# --label_type: 标签类型 (0: 子图ID, 1: 语义标签, 2: 混合)
# --time_convert_denominator: 时间转换分母
# --gen_exe_topo_json: 生成可执行拓扑 JSON
```

**输出文件**：
- `merged_swimlane.json` - Chrome Trace 格式，可在 https://ui.perfetto.dev/ 查看
- `bubble_analysis.log` - 气泡分析报告
- `execute.json` - 可执行拓扑（使用 --gen_exe_topo_json）

---

### 4.2 gen_swimlane.sh - 批量泳道图生成

批量处理多个输出目录，自动生成泳道图。

```bash
# 处理 Python 前端输出（默认处理最新的 1 个）
./tools/gen_swimlane.sh -f python3

# 处理 C++ 前端输出
./tools/gen_swimlane.sh -f cpp

# 指定自定义输出目录
./tools/gen_swimlane.sh -p /path/to/output

# 处理多个输出目录（从最新开始处理 5 个）
./tools/gen_swimlane.sh -f python3 -n 5

# 从指定编号开始处理
./tools/gen_swimlane.sh -f python3 -s 10 -n 3

# 显示帮助
./tools/gen_swimlane.sh -h

# 参数说明
# -f, --front: 前端类型 (python3, py, python, cpp)
# -p, --path: 自定义输出目录路径
# -n, --number: 处理的目录数量
# -s, --start: 起始编号
# -h, --help: 显示帮助
```

**目录命名约定**：
- 目录名应以 `_编号` 结尾（如 `output_1`, `output_2`）

---

### 4.3 tilefwk_pmu_to_csv.py - PMU 数据分析

解析 PMU 性能计数器数据，生成 CSV 报告。

```bash
# 基本用法
python3 tools/profiling/tilefwk_pmu_to_csv.py \
    --pmu_data_path output/output_1/ \
    --arch dav_2201

# 指定 PMU 事件类型
python3 tools/profiling/tilefwk_pmu_to_csv.py \
    -p output/output_1/ \
    --arch dav_2201 \
    --pmuEvent 2

# 指定输出目录
python3 tools/profiling/tilefwk_pmu_to_csv.py \
    -p output/output_1/ \
    --arch dav_2201 \
    --output ./pmu_reports

# 参数说明
# -p, --path: aicpu.data 文件所在目录
# --arch: 架构类型 (dav_2201, dav_3510)
# --pmuEvent: PMU 事件类型 (1, 2, 4, 5, 6, 7, 8)
# --output: 输出目录

# PMU 事件类型 (dav_2201)
# Event 1: 计算单元利用率 (cube_fp16_exec, vec_fp32_exec)
# Event 2: 繁忙周期 (vec_busy_cycles, cube_busy_cycles)
# Event 4: 内存请求 (ub_read_req, l1_read_req, l2_read_req)
# Event 5: L0 缓存请求 (l0a_read_req, l0b_read_req, l0c_read_req)
# Event 6: 停顿周期 (bankgroup_stall_cycles, bank_stall_cycles)
# Event 7: 带宽指标 (ub_read_bw, l2_write_bw)
# Event 8: 缓存命中率 (write_cache_hit, read_cache_hit)
```

**输出文件**：
- `tilefwk_prof_pmu.csv` - PMU 数据 CSV 报告

---

### 4.4 tiling_tool.py - Tiling 自动探索

自动探索最优的 Tiling 配置（目前主要支持 Matmul）。

```bash
# 基本用法
python3 tools/scripts/tiling_tool.py config.json

# 显示覆盖信息
python3 tools/scripts/tiling_tool.py config.json --coverage

# 配置文件示例 (config.json)
{
    "test_name": "test_matmul",
    "device_number": 0,
    "build_folder": "./",
    "results_folder": "./tiling_results",
    "prof_try_cnt": 3,
    "max_cnt": 100,
    "warn_up_cnt": 10,
    "save_best_k": 5,
    "files": [
        {
            "path/to/operator.py": [
                {
                    "line": 42,
                    "string": "pypto.set_cube_tile_shapes([{m}, {M}], [{k}, {K}], [{n}, {N}], {mdl}, {sk})",
                    "shape": "Matmul_fp32_128_512_256"
                }
            ]
        }
    ]
}
```

**功能说明**：
- 基于硬件约束生成候选 Tiling 配置
- 使用评分算法评估配置（考虑 L0 缓存利用率）
- 自动运行测试并收集性能数据
- 保存最优的 K 个配置

---

### 4.5 parse_dump_tensors.py - Tensor 解析对比

解析 dump 的 Tensor 文件并与 Golden 数据对比。

```bash
# 基本用法
python3 tools/verifier/parse_dump_tensors.py \
    --dump_tensor_path output/dump_tensor/device_0 \
    --verify_path output/output_1/

# 参数说明
# --dump_tensor_path: dump_tensor 目录路径
# --verify_path: verify_result.csv 所在目录

# 输出文件
# tensor_info.csv - 包含所有 Tensor 信息和对比结果
```

**输出内容**：
- Tensor 基本信息（shape, dtype, magic）
- 执行时间信息
- 与 Golden 数据的对比结果（`cmp_res` 字段）

---

## 五、综合示例

### 5.1 完整的算子开发调试流程

```python
import os
import torch
import pypto

# 1. 设置环境
os.environ['TILE_FWK_DEVICE_ID'] = '0'

# 2. 准备数据
input_x = torch.randn(1024, dtype=torch.float32)
input_y = torch.randn(1024, dtype=torch.float32)
golden_output = torch.add(input_x, input_y)

# 3. 配置调试选项
pypto.set_verify_options(
    enable_pass_verify=True,
    pass_verify_save_tensor=True,
    pass_verify_save_tensor_dir="./debug_output",
    pass_verify_error_tol=[1e-3, 1e-3]
)

# 4. 设置 Golden 数据
pypto.set_verify_golden_data(goldens=[golden_output])

# 5. 定义算子（带调试信息）
@pypto.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    debug_options={"runtime_debug_mode": 1}
)
def debug_add(x: pypto.Tensor, y: pypto.Tensor) -> pypto.Tensor:
    result = pypto.add(x, y)
    # 打印中间结果
    pypto.pass_verify_print("result=", result)
    # 保存到文件
    pypto.pass_verify_save(result, "add_result")
    return result

# 6. 执行算子
try:
    output = debug_add(pypto.Tensor(input_x), pypto.Tensor(input_y))
    print("Verification passed!")
except Exception as e:
    print(f"Verification failed: {e}")

# 7. 生成泳道图（在命令行执行）
# ./tools/gen_swimlane.sh -f python3
```

### 5.2 性能调优示例

```python
import pypto
import torch
import time

# 配置性能调试
pypto.set_debug_options(runtime_debug_mode=1)

# 定义带 Tiling 配置的算子
@pypto.jit(pass_options={"vec_nbuffer_mode": 1, "cube_l1_reuse_mode": 1})
def optimized_matmul(a: pypto.Tensor, b: pypto.Tensor) -> pypto.Tensor:
    # 配置 Cube Tile
    pypto.set_cube_tile_shapes(
        m=[64, 128],
        k=[128, 256],
        n=[64, 128],
        enable_multi_data_load=True
    )
    return pypto.matmul(a, b, a.dtype)

# 准备数据
a = torch.randn(512, 512, dtype=torch.float16, device='npu:0')
b = torch.randn(512, 512, dtype=torch.float16, device='npu:0')

# 预热
for _ in range(3):
    _ = optimized_matmul(a, b)

# 性能测试
start = time.perf_counter()
for _ in range(10):
    output = optimized_matmul(a, b)
elapsed = time.perf_counter() - start
print(f"Average time: {elapsed / 10 * 1000:.2f} ms")

# 分析性能数据
# python3 tools/profiling/draw_swim_lane.py output/output_1/tilefwk_L1_prof_data.json ...
# python3 tools/profiling/tilefwk_pmu_to_csv.py -p output/output_1/ --arch dav_2201
```

### 5.3 分阶段编译调试

```python
import pypto

# 逐步调试编译问题
stages = [
    (pypto.CompStage.TENSOR_GRAPH, "Tensor Graph"),
    (pypto.CompStage.TILE_GRAPH, "Tile Graph"),
    (pypto.CompStage.EXECUTE_GRAPH, "Execute Graph"),
    (pypto.CompStage.CODEGEN_INSTRUCTION, "Codegen Instruction"),
    (pypto.CompStage.CODEGEN_BINARY, "Codegen Binary"),
]

for stage, name in stages:
    print(f"\n=== Testing {name} stage ===")
    pypto.set_host_options(compile_stage=stage)

    try:
        @pypto.jit
        def test_kernel(x: pypto.Tensor):
            return pypto.add(x, x)

        # 这里只编译不执行
        print(f"{name}: OK")
    except Exception as e:
        print(f"{name}: FAILED - {e}")
        break
```

---

## 六、常见问题与解决方案

### Q1: 如何在无 NPU 环境下开发调试？

使用 CPU 仿真模式：

```python
@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def my_kernel(x: pypto.Tensor):
    return pypto.add(x, x)
```

### Q2: 如何定位精度问题？

1. 启用验证并保存中间 Tensor
2. 使用 `pass_verify_print()` 打印关键位置
3. 使用 `pass_verify_save()` 保存问题 Tensor
4. 对比 CPU 和 NPU 结果

### Q3: 如何分析性能瓶颈？

1. 启用 `runtime_debug_mode=1`
2. 使用 `gen_swimlane.sh` 生成泳道图
3. 查看 `bubble_analysis.log` 中的气泡分析
4. 使用 `tilefwk_pmu_to_csv.py` 分析 PMU 数据

### Q4: 如何选择最优 Tiling 配置？

1. 使用 `tiling_tool.py` 自动探索
2. 对比不同配置的泳道图和 PMU 数据
3. 关注 L0 缓存利用率和气泡时间
