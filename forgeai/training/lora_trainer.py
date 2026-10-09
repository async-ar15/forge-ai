"""LoRA/QLoRA trainer executed inside Ray workers."""

from __future__ import annotations

import io
import json
import logging
import random
from typing import Any, cast

import numpy as np

from forgeai.training.constants import GRADIENT_NORM_WARNING_THRESHOLD
from forgeai.training.exceptions import DatasetSchemaError, NaNLossError
from forgeai.training.orchestrator import TrainingJobConfig

_LOG = logging.getLogger(__name__)
_DATASET_REQUIRED_KEYS = ("instruction", "input", "output")


class ForgeTrainingCallback:
    """Logs step metrics and aborts on unsafe loss values."""

    __slots__ = ()

    def on_log(
        self,
        args: object,
        state: object,
        control: object,
        logs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        logs = logs or {}
        step = int(getattr(state, "global_step", 0))
        epoch = float(getattr(state, "epoch", 0.0) or 0.0)
        loss = _to_float(logs.get("loss", 0.0))
        grad = _to_float(logs.get("grad_norm", 0.0))
        lr = _to_float(logs.get("learning_rate", 0.0))
        if np.isnan(loss):
            raise NaNLossError(step=step, epoch=epoch, loss_value=loss)
        if grad > GRADIENT_NORM_WARNING_THRESHOLD:
            _LOG.warning(
                "gradient explosion detected step=%d gradient_norm=%f", step, grad
            )
        _LOG.info(
            "training_step step=%d epoch=%.4f loss=%.6f "
            "gradient_norm=%.6f learning_rate=%.8f",
            step,
            epoch,
            loss,
            grad,
            lr,
        )


class LoRATrainer:
    """Deterministic LoRA/QLoRA fine-tuning executor."""

    __slots__ = ("_boto3", "_torch", "_transformers")

    def __init__(
        self,
        *,
        boto3_module: object,
        torch_module: object,
        transformers_module: object,
    ) -> None:
        self._boto3 = boto3_module
        self._torch = torch_module
        self._transformers = transformers_module

    def train(self, config: TrainingJobConfig) -> dict[str, object]:
        _set_all_seeds(self._torch, self._transformers, config.seed)
        rows = _load_jsonl_dataset(self._boto3, config.dataset_path)
        _validate_dataset_schema(rows, config.dataset_path)
        _log_dataset_distribution(rows)
        model, tokenizer = _load_base_model(self._transformers, self._torch, config)
        peft_model = _apply_lora(self._transformers, model, config)
        args = _training_arguments(self._transformers, config)
        callback = ForgeTrainingCallback()
        trainer = _build_trainer(
            self._transformers, peft_model, tokenizer, rows, args, callback
        )
        cast(Any, trainer).train()
        return {"ok": True}


def _set_all_seeds(
    torch_module: object,
    transformers_module: object,
    seed: int,
) -> None:
    """Set every seed before any data/model loading to enforce determinism."""

    random.seed(seed)
    np.random.seed(seed)
    torch_any = cast(Any, torch_module)
    transformers_any = cast(Any, transformers_module)
    torch_any.manual_seed(seed)
    torch_any.cuda.manual_seed_all(seed)
    transformers_any.set_seed(seed)
    torch_any.backends.cudnn.deterministic = True
    torch_any.backends.cudnn.benchmark = False


def _load_jsonl_dataset(boto3_module: object, s3_uri: str) -> list[dict[str, object]]:
    bucket, key = _parse_s3_uri(s3_uri)
    client = cast(Any, boto3_module).client("s3")
    obj = client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    text = body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else str(body)
    return [json.loads(line) for line in io.StringIO(text) if line.strip()]


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"dataset_path must be s3://... got {uri}")
    no = uri[5:]
    bucket, key = no.split("/", 1)
    return bucket, key


def _validate_dataset_schema(rows: list[dict[str, object]], path: str) -> None:
    for idx, row in enumerate(rows[:10]):
        missing = tuple(k for k in _DATASET_REQUIRED_KEYS if k not in row)
        if missing:
            raise DatasetSchemaError(
                dataset_path=path, line_index=idx, missing_keys=missing
            )


def _log_dataset_distribution(rows: list[dict[str, object]]) -> None:
    lengths = sorted(
        len(
            (
                str(r.get("instruction", ""))
                + str(r.get("input", ""))
                + str(r.get("output", ""))
            ).split()
        )
        for r in rows
    )
    if not lengths:
        _LOG.info("dataset_stats size=0 p50=0 p95=0 p99=0")
        return
    p50 = lengths[int(0.5 * (len(lengths) - 1))]
    p95 = lengths[int(0.95 * (len(lengths) - 1))]
    p99 = lengths[int(0.99 * (len(lengths) - 1))]
    _LOG.info("dataset_stats size=%d p50=%d p95=%d p99=%d", len(lengths), p50, p95, p99)


def _load_base_model(
    transformers_module: object,
    torch_module: object,
    config: TrainingJobConfig,
) -> tuple[object, object]:
    kwargs: dict[str, object] = {}
    transformers_any = cast(Any, transformers_module)
    torch_any = cast(Any, torch_module)
    if config.use_qlora:
        kwargs["quantization_config"] = _qlora_bits_config(
            transformers_any, torch_any, config.quantization_bits
        )
    model = transformers_any.AutoModelForCausalLM.from_pretrained(
        str(config.base_model_id), **kwargs
    )
    tokenizer = transformers_any.AutoTokenizer.from_pretrained(
        str(config.base_model_id)
    )
    if config.use_qlora:
        peft = __import__("peft")
        model = peft.prepare_model_for_kbit_training(model)
        _LOG.info(
            "QLoRA reduces GPU memory by ~75%% vs full fp16 "
            "fine-tuning at ~1-3%% accuracy cost."
        )
    return model, tokenizer


def _qlora_bits_config(
    transformers_module: object,
    torch_module: object,
    bits: int | None,
) -> object:
    if bits not in (4, 8):
        raise ValueError("quantization_bits must be 4 or 8 when use_qlora=True")
    transformers_any = cast(Any, transformers_module)
    torch_any = cast(Any, torch_module)
    return transformers_any.BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_compute_dtype=torch_any.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _apply_lora(
    transformers_module: object,
    model: object,
    config: TrainingJobConfig,
) -> object:
    peft = __import__("peft")
    lora_cfg = peft.LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
    )
    peft_model = peft.get_peft_model(model, lora_cfg)
    trainable, total = peft_model.get_nb_trainable_parameters()
    ratio = float(trainable / max(total, 1))
    _LOG.info("lora_params trainable=%d total=%d ratio=%.6f", trainable, total, ratio)
    if ratio >= 0.05:
        _LOG.warning("lora_trainable_ratio_high ratio=%.6f", ratio)
    return peft_model


def _training_arguments(
    transformers_module: object,
    config: TrainingJobConfig,
) -> object:
    return cast(Any, transformers_module).TrainingArguments(
        output_dir=config.output_path,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        max_steps=-1,
        logging_steps=1,
    )


def _build_trainer(
    transformers_module: object,
    model: object,
    tokenizer: object,
    rows: list[dict[str, object]],
    args: object,
    callback: ForgeTrainingCallback,
) -> object:
    transformers_any = cast(Any, transformers_module)
    collator = transformers_any.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = transformers_any.Trainer(
        model=model,
        args=args,
        train_dataset=rows,
        data_collator=collator,
        callbacks=[callback],
    )
    return trainer


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise TypeError(f"cannot coerce {type(value)!r} to float")


__all__ = ["ForgeTrainingCallback", "LoRATrainer"]
