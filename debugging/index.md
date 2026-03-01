# 调试与维测

本章节提供 PyPTO 调试和维测相关的文档。

```{toctree}
:maxdepth: 1
:titlesonly:

debugging_system_review
pypto_debugging_capabilities_analysis
pypto_debugging_capabilities_by_workflow
api_examples
```

## 调试能力概览

PyPTO 提供了多层次的调试和维测能力：

### 功能调试

| 能力 | API/工具 | 说明 |
|------|----------|------|
| CPU 孪生调试 | `--run_mode sim` | 无需 NPU 进行功能验证 |
| 控制流图可视化 | Toolkit | 帮助理解程序结构 |
| 中间值打印 | `pass_verify_print()` | 条件打印中间结果 |
| 中间值落盘 | `pass_verify_save()` | 保存中间 Tensor 到文件 |
| 分阶段编译 | `CompStage` | 定位编译阶段问题 |

### 精度调试

| 能力 | API/工具 | 说明 |
|------|----------|------|
| Golden 对比 | `set_verify_options()` | 自动对比验证 |
| Tensor 落盘 | `pass_verify_save_tensor_dir` | 保存中间结果 |
| 自定义容差 | `pass_verify_error_tol` | 支持不同精度要求 |
| Golden 数据设置 | `set_verify_golden_data()` | 设置基准数据 |

### 性能调优

| 能力 | API/工具 | 说明 |
|------|----------|------|
| 泳道图 | `merged_swimlane.json` | 可视化时间线 |
| 气泡分析 | `bubble_analysis.log` | 识别空闲时间 |
| Tiling 配置 | `CubeTile`, `TileShape` | 灵活的切分策略 |
| 性能报告 | `performance_report.json` | 综合性能数据 |

## 快速开始

### 1. 启用精度调试

```python
import pypto

# 配置精度调试选项
pypto.set_verify_options(
    enable_pass_verify=True,
    pass_verify_save_tensor=True,
    pass_verify_save_tensor_dir="./debug_output",
    pass_verify_error_tol=[1e-3, 1e-3]
)

# 在算子中使用打印和保存
@pypto.jit
def my_kernel(x: pypto.Tensor, y: pypto.Tensor):
    z = pypto.add(x, y)
    pypto.pass_verify_print(z)  # 打印中间结果
    pypto.pass_verify_save(z, "z_output")  # 保存到文件
    return z
```

### 2. 启用调试模式

```python
# 开启编译阶段调试
pypto.set_debug_options(compile_debug_mode=1)

# 开启运行时调试（泳道图）
pypto.set_debug_options(runtime_debug_mode=1)
```

### 3. 使用 Golden 数据对比

```python
import torch

# 准备 golden 数据
golden_output = torch.add(input_x, input_y)

# 设置 golden 数据
pypto.set_verify_golden_data(goldens=[golden_output])

# 执行算子，自动对比
result = my_kernel(pypto.Tensor(input_x), pypto.Tensor(input_y))
```

## 渐进式调试方法

推荐使用 Level 0~N 多级用例构建：

```
Level 0: 8-16 元素  ──▶ 基础功能验证
    ↓ 通过
Level 1: 1K 元素     ──▶ 典型场景验证
    ↓ 通过
Level 2: 极值/零值   ──▶ 边界情况验证
    ↓ 通过
Level 3: 大数据量    ──▶ 性能验证
```

## 相关文档

- [维测体系审视与改进建议](debugging_system_review.md): 从算子开发者角度审视 PyPTO 维测方法的完备性
- [PyPTO维测能力分析（按API分类）](pypto_debugging_capabilities_analysis.md): 按 API 分类系统分析维测能力
- [PyPTO维测能力分析（按工作流三阶段）](pypto_debugging_capabilities_by_workflow.md): 按算子开发、功能调试、性能调试三阶段系统分析维测能力
- [API 使用示例](api_examples.md): 所有维测 API 和工具的完整使用示例
- [API 文档 - Verify 选项](../api/config/pypto-set_verify_options.md)
- [API 文档 - Debug 选项](../api/config/pypto-set_debug_options.md)
- [API 文档 - pass_verify_print](../api/others/pypto-pass_verify_print.md)
- [API 文档 - pass_verify_save](../api/others/pypto-pass_verify_save.md)
