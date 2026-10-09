"""Training service package for async orchestration and checkpoint promotion."""

__all__ = [
    "CheckpointManager",
    "CheckpointRecord",
    "LoRATrainer",
    "ModelPromoter",
    "TrainingJobConfig",
    "TrainingJobHandle",
    "TrainingJobStatus",
    "TrainingOrchestrator",
    "TrainingState",
]


def __getattr__(name: str) -> object:
    if name in {"CheckpointManager", "CheckpointRecord"}:
        from forgeai.training.checkpoint import CheckpointManager, CheckpointRecord

        return {
            "CheckpointManager": CheckpointManager,
            "CheckpointRecord": CheckpointRecord,
        }[name]
    if name == "LoRATrainer":
        from forgeai.training.lora_trainer import LoRATrainer

        return LoRATrainer
    if name == "ModelPromoter":
        from forgeai.training.promoter import ModelPromoter

        return ModelPromoter
    if name in {
        "TrainingJobConfig",
        "TrainingJobHandle",
        "TrainingJobStatus",
        "TrainingOrchestrator",
        "TrainingState",
    }:
        from forgeai.training.orchestrator import (
            TrainingJobConfig,
            TrainingJobHandle,
            TrainingJobStatus,
            TrainingOrchestrator,
            TrainingState,
        )

        return {
            "TrainingJobConfig": TrainingJobConfig,
            "TrainingJobHandle": TrainingJobHandle,
            "TrainingJobStatus": TrainingJobStatus,
            "TrainingOrchestrator": TrainingOrchestrator,
            "TrainingState": TrainingState,
        }[name]
    raise AttributeError(name)
