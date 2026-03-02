# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a **documentation repository** for PyPTO (Python Portable Tensor Operator), a high-performance programming framework for AI accelerators (Huawei Ascend NPU). It contains:

- `pypto/` - Clone of the main PyPTO repository with source code and comprehensive documentation
- `debugging/` - Additional debugging and verification documentation
- `pypto-debugging-methods.md` - Comprehensive summary of PyPTO debugging methods

**For PyPTO development guidance**, see `pypto/CLAUDE.md` which contains detailed build commands, testing instructions, and development workflows.

## Key Documentation Files

| File | Purpose |
|------|---------|
| `pypto/CLAUDE.md` | Main development guidance for PyPTO |
| `pypto/AGENTS.md` | OpenCode agent guidance |
| `pypto/README.md` | Project overview in Chinese |
| `pypto-debugging-methods.md` | Comprehensive debugging methods summary |
| `debugging/index.md` | Debugging documentation index |

## Build Commands Summary

All build commands should be run from the `pypto/` directory:

```bash
# Build Python whl package
cd pypto && python3 build_ci.py -f python3 --disable_auto_execute

# Install the package
pip install build_out/pypto-*.whl --no-deps
```

## Testing

```bash
# Run unit tests
cd pypto && pytest python/tests/ut -v -n auto

# Run system tests (requires NPU)
cd pypto && pytest python/tests/st -v --device 0

# Run examples in simulation mode (no NPU required)
cd pypto && python3 examples/validate_examples.py -t examples --run_mode sim -w 16
```

## Environment Variables

```bash
export TILE_FWK_DEVICE_ID=0                                    # NPU device ID
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto_isa/pto-isa/  # PTO-ISA path
```

## Debugging Quick Reference

```python
import pypto

# Enable verification with golden comparison
pypto.set_verify_options(enable_pass_verify=True, pass_verify_error_tol=[1e-3, 1e-3])

# Enable debug mode for swimlane/bubble analysis
pypto.set_debug_options(runtime_debug_mode=1)

# Print intermediate values
pypto.pass_verify_print(tensor, "label=")

# Save intermediate tensors
pypto.pass_verify_save(tensor, "output_$idx", idx=loop_var)
```

## Documentation Build

```bash
cd pypto/docs && pip install -r requirements.txt && make html
python3 -m http.server 8000 -d pypto/docs/_build/html
```
