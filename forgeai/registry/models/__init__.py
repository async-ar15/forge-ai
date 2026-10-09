"""Owns: SQLAlchemy ORM modules for ForgeAI registry tables (Section 4).

Does not own: Alembic revision bodies, read replicas, or logical decoding consumers.
"""

from __future__ import annotations

from forgeai.registry.models.artifact import Artifact
from forgeai.registry.models.base import Base
from forgeai.registry.models.checkpoint import Checkpoint
from forgeai.registry.models.dataset import Dataset
from forgeai.registry.models.deployment import Deployment
from forgeai.registry.models.eval_run import EvalRun
from forgeai.registry.models.ml_model import RegistryModel
from forgeai.registry.models.model_version import ModelVersion
from forgeai.registry.models.quant_profile import QuantProfile
from forgeai.registry.models.routing_decision import RoutingDecision
from forgeai.registry.models.tenant import Tenant
from forgeai.registry.models.tenant_policy import TenantPolicy
from forgeai.registry.models.training_run import TrainingRun

__all__ = [
    "Artifact",
    "Base",
    "Checkpoint",
    "Dataset",
    "Deployment",
    "EvalRun",
    "ModelVersion",
    "QuantProfile",
    "RegistryModel",
    "RoutingDecision",
    "Tenant",
    "TenantPolicy",
    "TrainingRun",
]
