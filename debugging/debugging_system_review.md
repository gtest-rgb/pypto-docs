# PyPTO 维测体系审视与改进建议

## 目标

从算子编程人员角度审视 PyPTO 维测方法的完备性和易用性，对比 CUDA/Triton/PyTorch，提出改进建议。

---

## 一、算子开发者工作流与维测需求

### 典型开发工作流

```
需求分析 → 方案设计 → 实现编码 → 功能验证 → 精度验证 → 性能调优 → 部署上线
                ↓           ↓           ↓           ↓           ↓
             语法检查     单步调试     结果对比     误差定位     瓶颈分析
```

### 开发者核心痛点

1. **语法/编译错误**：JIT 编译时错误难以定位到源码位置
2. **逻辑错误**：计算结果与预期不符，需要逐行调试
3. **精度问题**：NPU 与 CPU 结果存在差异，难以定位误差来源
4. **性能瓶颈**：不知道哪部分代码拖慢了整体性能
5. **运行时错误**：NPU 上板失败，错误信息不够清晰

---

## 二、PyPTO 现有维测能力评估

### 已具备的能力

| 类别 | 能力 | API/工具 | 评价 |
|------|------|----------|------|
| **功能调试** | CPU 孪生调试 | `--run_mode sim` | 核心能力，无需 NPU |
| | 控制流图可视化 | Toolkit | 帮助理解程序结构 |
| | 中间值打印 | `pass_verify_print()` | 条件打印，灵活 |
| | 中间值落盘 | `pass_verify_save()` | 支持动态文件名 |
| | 分阶段编译 | `CompStage` | 帮助定位编译阶段问题 |
| **精度调试** | Golden 对比 | `set_verify_options()` | 自动对比验证 |
| | Tensor 落盘 | `pass_verify_save_tensor_dir` | 保存中间结果 |
| | 自定义容差 | `pass_verify_error_tol` | 支持不同精度要求 |
| | Golden 数据设置 | `set_verify_golden_data()` | 设置基准数据 |
| **性能调优** | 泳道图 | `merged_swimlane.json` | 可视化时间线 |
| | 气泡分析 | `bubble_analysis.log` | 识别空闲时间 |
| | Tiling 配置 | `CubeTile`, `TileShape` | 灵活的切分策略 |
| | 性能报告 | `performance_report.json` | 综合性能数据 |

### 缺失或不足的能力

| 类别 | 缺失能力 | 影响 | 紧急度 |
|------|----------|------|--------|
| **交互式调试** | 断点/单步执行 | 无法逐行检查代码逻辑 | 高 |
| | 变量检查器 | 无法实时查看 Tensor 值 | 高 |
| | 调用栈追踪 | 错误时无法看到调用链 | 中 |
| **错误诊断** | 精确的错误位置 | JIT 错误难定位到源码行 | 高 |
| | 详细的错误信息 | 错误提示不够具体 | 中 |
| | 错误恢复机制 | 一处错误导致整个程序终止 | 低 |
| **数值调试** | NaN/Inf 自动检测 | 需手动检查数值异常 | 高 |
| | 梯度异常检测 | 无类似 `set_detect_anomaly` | 中 |
| | 数值范围警告 | 无越界/溢出警告 | 中 |
| **性能调试** | 热点代码定位 | 泳道图与源码关联不够直观 | 中 |
| | 内存分析 | 无内存使用分析工具 | 低 |
| **开发体验** | IDE 集成调试 | 无断点、变量查看等 | 高 |
| | 单元测试框架 | 需要更完善的测试辅助 | 中 |

---

## 三、与 CUDA/Triton/PyTorch 对比

### 对比矩阵

| 维测能力 | PyPTO | CUDA | Triton | PyTorch |
|----------|-------|------|--------|---------|
| **断点调试** | 无 | cuda-gdb | Python pdb | pdb/ipdb |
| **单步执行** | 无 | cuda-gdb | 仅 Python 层 | pdb |
| **打印调试** | pass_verify_print | printf | device_print | print |
| **中间值保存** | pass_verify_save | 手动 | 手动 | .numpy() |
| **内存检查** | 无 | cuda-memcheck | 无 | 自动管理 |
| **性能分析** | 泳道图 | Nsight | torch.profiler | torch.profiler |
| **NaN/Inf 检测** | 无 | 手动 | 手动 | isnan/isinf |
| **梯度调试** | 有限 | 无 | 无 | set_detect_anomaly |
| **错误定位** | JIT 限制 | 行号精确 | Python 层 | Python 层 |
| **IDE 集成** | 仅可视化 | Nsight | VS Code | 完整支持 |
| **CPU 仿真** | 完整 | 无 | 解释器模式 | CPU 运行 |
| **Golden 对比** | 内置 API | 手动 | 手动 | 手动 |

### 关键差距分析

#### 1. 交互式调试能力（最大差距）

- **CUDA**: cuda-gdb 提供完整的断点、单步、变量检查
- **PyTorch**: pdb 可以在任何位置暂停，检查变量
- **PyPTO**: JIT 编译模式导致无法交互调试

#### 2. 错误诊断精度

- **PyTorch**: 错误直接指向 Python 源码行
- **CUDA**: cuda-gdb 显示精确的 CUDA C++ 源码位置
- **PyPTO**: JIT 错误信息经过多层转换，难以定位

#### 3. 数值异常检测

- **PyTorch**: `torch.autograd.set_detect_anomaly(True)` 自动定位 NaN/Inf
- **PyPTO**: 需要手动使用 `pass_verify_print` 检查

---

## 四、改进建议

### 优先级 P0（紧急，影响核心开发效率）

#### 1. 增强错误定位能力

```python
# 建议增加的 API
pypto.set_debug_options(
    show_source_location=True,    # 错误时显示源码位置
    show_stack_trace=True,        # 显示调用栈
    error_context_lines=5         # 错误上下文行数
)
```

**实现思路**：

- 在 AST 解析阶段记录源码位置信息
- 错误时回溯到原始 Python 代码位置
- 参考 PyTorch 的错误堆栈格式

#### 2. 添加数值异常自动检测

```python
# 建议增加的 API
pypto.set_verify_options(
    detect_nan=True,              # 自动检测 NaN
    detect_inf=True,              # 自动检测 Inf
    detect_overflow=True,         # 检测数值溢出
    stop_on_error=True            # 发现异常时停止
)
```

**实现思路**：

- 在每个算子执行后自动检查输出
- 类似 `torch.autograd.set_detect_anomaly`
- 检测到异常时报告具体算子和位置

#### 3. 提供"慢模式"解释执行

```python
# 建议增加的模式
pypto.set_host_options(
    execution_mode="interpret"    # 逐行解释执行
)
```

**实现思路**：

- 不进行 JIT 编译，逐个算子执行
- 每个算子执行后可以检查中间结果
- 类似 Triton 的解释器模式

### 优先级 P1（重要，提升调试效率）

#### 4. 增强中间值检查

```python
# 建议增加的 API
pypto.watch(tensor, name="x")           # 添加监视变量
pypto.checkpoint("stage_1")             # 设置检查点
pypto.diff(tensor, golden, name="op")   # 自动对比差异
```

#### 5. 改进错误信息格式

```
# 当前格式（不够清晰）
RuntimeError: Tensor shape mismatch

# 建议格式
RuntimeError: Tensor shape mismatch
  File "my_op.py", line 42, in my_operator
    result = pypto.matmul(a, b)
  Expected shape: [1024, 512]
  Actual shape: [1024, 256]
  Input 'a' shape: [1024, 256]
  Input 'b' shape: [256, 256]
```

#### 6. 添加性能热点定位

```python
# 建议增加的 API
with pypto.profile() as prof:
    result = my_operator(x)

prof.print_hotspots(top_k=10)      # 打印热点算子
prof.suggest_optimizations()        # 优化建议
```

### 优先级 P2（增强，提升开发体验）

#### 7. IDE 集成增强

- VS Code 断点支持（通过 debugpy 集成）
- 变量查看面板
- 实时 Tensor 可视化

#### 8. 单元测试辅助

```python
# 建议增加的测试装饰器
@pypto.test_case(
    shapes=[[1024, 1024]],
    dtypes=[pypto.DT_FLOAT, pypto.DT_BF16],
    rtol=1e-3,
    golden_fn=torch_matmul
)
def test_matmul(a, b):
    return pypto.matmul(a, b)
```

#### 9. 调试日志分级

```python
pypto.set_debug_options(
    log_level="DEBUG",             # DEBUG/INFO/WARNING/ERROR
    log_file="debug.log"
)
```

---

## 五、其他改进角度

### 1. 开发者体验（DX）角度

- **Quick Start 模板**: 提供一键调试脚手架
- **交互式教程**: Jupyter Notebook 形式的调试教程
- **错误代码文档**: 每种错误类型的解决方案

### 2. 社区生态角度

- **调试案例库**: 收集典型调试场景和解决方案
- **Stack Overflow 标签**: 建立社区支持
- **GitHub Issue 模板**: 标准化问题报告

### 3. 工具链整合角度

- **与 PyTorch 互操作**: 支持在 PyTorch 中调用 PyPTO 算子并调试
- **性能对比工具**: 与 PyTorch/CUDA 实现自动对比
- **CI/CD 集成**: 自动化测试和回归检测

### 4. 文档角度

- **调试决策树**: 根据问题类型推荐调试方法
- **API 速查卡**: 单页调试 API 参考
- **视频教程**: 可视化调试流程演示

---

## 六、实施路线图

### 第一阶段（1-2 周）

- [ ] 增强错误信息，包含源码位置
- [ ] 添加 NaN/Inf 自动检测
- [ ] 改进错误信息格式

### 第二阶段（2-4 周）

- [ ] 实现"慢模式"解释执行
- [ ] 添加变量监视和检查点
- [ ] 性能热点定位 API

### 第三阶段（1-2 月）

- [ ] IDE 集成（VS Code 扩展增强）
- [ ] 单元测试框架增强
- [ ] 调试日志分级

### 第四阶段（持续）

- [ ] 调试案例库建设
- [ ] 文档和教程完善
- [ ] 社区反馈收集

---

## 七、总结

### 完备性评估

- **功能调试**: 70% 完备（缺交互式调试）
- **精度调试**: 80% 完备（缺自动异常检测）
- **性能调优**: 85% 完备（缺热点定位）

### 易用性评估

- **API 设计**: 良好，命名清晰
- **错误信息**: 需改进，不够精确
- **文档完整性**: 良好
- **IDE 集成**: 需增强

### 核心建议

1. **最高优先级**: 增强错误定位和 NaN/Inf 自动检测
2. **中期目标**: 实现慢模式解释执行
3. **长期目标**: 完善工具链和社区生态

---

## 八、现有维测能力详细说明

### 8.1 精度调试 Verify 特性

PyPTO 提供了完整的精度调试 Verify 特性，通过以下 API 进行配置：

#### set_verify_options

```python
pypto.set_verify_options(
    enable_pass_verify=True,           # 总体使能开关
    pass_verify_save_tensor=True,      # 将模拟计算数据存盘
    pass_verify_save_tensor_dir="/path/to/save",  # 检测结果及数据的保存路径
    pass_verify_pass_filter=["Pass1", "Pass2"],   # 待自检的 Pass 名称列表
    pass_verify_error_tol=[1e-3, 1e-3]            # rtol 和 atol
)
```

#### pass_verify_print

在精度调试 Verify 特性使能时，打印指定 Tensor 计算的结果：

```python
# 基本打印
pypto.pass_verify_print(tensor)

# 条件打印
for idx in pypto.loop(10):
    t = pypto.add(a, b)
    pypto.pass_verify_print(t, cond=(idx == 5))  # 仅在 idx==5 时打印
```

#### pass_verify_save

保存中间 Tensor 到文件：

```python
# 保存到 t2.data 和 t2.csv
pypto.pass_verify_save(t2, 't2-fileprefix')

# 动态文件名 + 条件保存
pypto.pass_verify_save(t3, "t3_debug_loop_$idx", cond=(idx == 5), idx=5)
```

#### set_verify_golden_data

设置基准数据进行对比：

```python
# 仅设置 golden 输出
pypto.set_verify_golden_data(goldens=[None, None, golden_out0])

# 设置输入和 golden 输出
pypto.set_verify_golden_data([real_in0, real_in1, real_out0], [None, None, golden_out0])
```

### 8.2 调试选项配置

#### set_debug_options

```python
# 开启编译阶段调试模式
pypto.set_debug_options(compile_debug_mode=1)

# 开启执行阶段调试模式（泳道图）
pypto.set_debug_options(runtime_debug_mode=1)
```

### 8.3 分阶段编译控制

#### set_host_options

通过 CompStage 控制编译执行的阶段：

```python
pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH)
```

可用的阶段：
- `ALL_COMPLETE`: 正常编译与运行
- `TENSOR_GRAPH`: 生成最终张量图后停止
- `TILE_GRAPH`: 生成最终分片图后终止
- `EXECUTE_GRAPH`: 生成最终执行图后终止
- `CODEGEN_INSTRUCTION`: 生成指令代码后终止
- `CODEGEN_BINARY`: 生成代码二进制后终止

### 8.4 PyPTO Toolkit 可视化工具

PyPTO Toolkit 是一款全流程辅助工具，提供以下可视化能力：

#### 控制流图

- 将基于 PyPTO 框架表达的计算逻辑可视化
- 提供计算逻辑表达代码、控制流图及计算图的映射关系视图

#### 计算图

- 基于 Tensor 和 Operation 的基础图结构
- 提供原始计算图到可执行图编译过程的全流程可视化展示

#### 泳道图

- 提供核间、核内流水和统计信息的可视化展示
- 支持任务的依赖关系分析和多种维度的统计报告信息展示

#### 性能报告

- Task Time: 执行总时间
- AICore Time: 所有 AI Core 泳道中的任务总耗时
- AICore 利用率: AI Core Time / (时间轴总时长 * AICore 泳道的个数)

#### 三栏联动视图

- 支持算子代码-计算图-泳道图之间的实时联动
- 降低使用 PyPTO 框架调试调优的门槛

---

## 九、渐进式调试方法

### Level 0~N 多级用例构建

```
Level 0: 8-16 元素  ──▶ 基础功能验证
    ↓ 通过
Level 1: 1K 元素     ──▶ 典型场景验证
    ↓ 通过
Level 2: 极值/零值   ──▶ 边界情况验证
    ↓ 通过
Level 3: 大数据量    ──▶ 性能验证
```

### 分段调试步骤（复杂公式）

1. 识别公式中的关键中间步骤
2. 在每个步骤后插入 `pypto.assemble`，并将 tensor 的结果作为输出
3. 运行并比对每个中间输出值与预期
4. 定位误差来源的具体步骤
5. 修复该步骤的问题

### 错误处理原则

- **编译错误**: 定位错误行号，参考文档检查语法，对比官方示例
- **运行时错误**: 使用增加中间输出的方式定位问题，采用渐进式调试方法
- **精度错误**: 从最小用例开始，分段验证中间结果，检查数据类型
- **环境配置错误**: 检查 `TILE_FWK_DEVICE_ID` 是否已设置，使用 `npu-smi info` 确认 NPU 设备号
