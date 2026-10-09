"""Model Registry package.

Imports are intentionally lazy here so lightweight modules (constants, schema
helpers) do not require database runtime dependencies during import.
"""

__all__ = [
    "Artifact",
    "Base",
    "Dataset",
    "Deployment",
    "EvalRun",
    "ModelVersion",
    "ObjectStore",
    "QuantProfile",
    "RegistryModel",
    "RoutingDecision",
    "Tenant",
    "TenantPolicy",
    "TrainingRun",
    "create_registry_engine",
]


def __getattr__(name: str) -> object:
    if name == "create_registry_engine":
        from forgeai.registry.database import create_registry_engine

        return create_registry_engine
    if name == "ObjectStore":
        from forgeai.registry.storage import ObjectStore

        return ObjectStore
    from forgeai.registry import models

    if hasattr(models, name):
        return getattr(models, name)
    raise AttributeError(name)
