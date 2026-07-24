from viaabc.api import infer, infer_from_config
from viaabc.config import ABCConfig, ExperimentSpec
from viaabc.encoders import CallableEncoder, IdentityEncoder, TorchCheckpointEncoder, TorchEncoder
from viaabc.priors import JointPrior, LogNormalPrior, UniformPrior
from viaabc.result import ABCResult, ViaABCResult
from viaabc.runtime import Runtime
from viaabc.system import ABCSystem
from viaabc.experiment.registry import registry

registry.register("encoder.identity", IdentityEncoder)
registry.register("encoder.callable", CallableEncoder)
registry.register("encoder.torch", TorchEncoder)
registry.register("encoder.torch_checkpoint", TorchCheckpointEncoder)
registry.register("prior.joint", JointPrior)
registry.register("prior.uniform", UniformPrior)
registry.register("prior.lognormal", LogNormalPrior)
registry.register("runtime.auto", Runtime.auto)
registry.register("runtime.cpu", Runtime.cpu)
registry.register("runtime.gpu", Runtime.gpu)

__all__ = [
    "infer",
    "infer_from_config",
    "ABCConfig",
    "ExperimentSpec",
    "ABCSystem",
    "Runtime",
    "ViaABCResult",
    "ABCResult",
    "IdentityEncoder",
    "CallableEncoder",
    "TorchEncoder",
    "TorchCheckpointEncoder",
    "UniformPrior",
    "LogNormalPrior",
    "JointPrior",
]
