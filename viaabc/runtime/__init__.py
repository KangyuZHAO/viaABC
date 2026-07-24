from viaabc.runtime.base import Runtime
from viaabc.runtime.gpu import GPURuntime
from viaabc.runtime.local import LocalRuntime

CPURuntime = LocalRuntime

__all__ = ["Runtime", "LocalRuntime", "CPURuntime", "GPURuntime"]
