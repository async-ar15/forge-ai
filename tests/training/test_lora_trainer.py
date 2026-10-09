from __future__ import annotations

import json
import types
import uuid
from unittest.mock import MagicMock

import pytest
from forgeai.training.exceptions import DatasetSchemaError, NaNLossError
from forgeai.training.lora_trainer import ForgeTrainingCallback, LoRATrainer
from forgeai.training.orchestrator import TrainingJobConfig


def _cfg() -> TrainingJobConfig:
    return TrainingJobConfig(
        job_id=uuid.uuid4(),
        base_model_id=uuid.uuid4(),
        dataset_path="s3://bucket/data.jsonl",
        output_path="s3://bucket/out",
        seed=7,
    )


def _torch() -> object:
    return types.SimpleNamespace(
        manual_seed=MagicMock(),
        cuda=types.SimpleNamespace(manual_seed_all=MagicMock()),
        backends=types.SimpleNamespace(
            cudnn=types.SimpleNamespace(deterministic=False, benchmark=True)
        ),
        float16=object(),
    )


def _transformers() -> object:
    trainer = MagicMock()
    trainer.train = MagicMock()
    trainer_instance = trainer.return_value
    trainer_instance.train = MagicMock()
    trainer_instance.state = types.SimpleNamespace(log_history=[{"loss": 0.1}])
    return types.SimpleNamespace(
        set_seed=MagicMock(),
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=MagicMock(return_value=MagicMock())
        ),
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=MagicMock(return_value=MagicMock())
        ),
        BitsAndBytesConfig=MagicMock(return_value=MagicMock()),
        TrainingArguments=MagicMock(return_value=MagicMock()),
        DataCollatorForSeq2Seq=MagicMock(return_value=MagicMock()),
        Trainer=trainer,
    )


def _boto3(payload: str) -> object:
    client = MagicMock()
    client.get_object.return_value = {
        "Body": MagicMock(read=MagicMock(return_value=payload.encode("utf-8")))
    }
    return types.SimpleNamespace(client=MagicMock(return_value=client))


def test_all_seeds_set_before_data_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    tmod = _torch()
    tfmod = _transformers()
    bmod = _boto3('{"instruction":"i","input":"x","output":"y"}\n')
    tmod.manual_seed.side_effect = lambda _x: order.append("seed_torch")
    tfmod.set_seed.side_effect = lambda _x: order.append("seed_transformers")
    bmod.client.return_value.get_object.side_effect = lambda **_k: (
        order.append("load_data"),
        {
            "Body": MagicMock(
                read=MagicMock(
                    return_value=b'{"instruction":"i","input":"x","output":"y"}\n'
                )
            )
        },
    )[1]
    monkeypatch.setitem(
        __import__("sys").modules,
        "peft",
        types.SimpleNamespace(
            LoraConfig=MagicMock(),
            get_peft_model=MagicMock(
                return_value=MagicMock(
                    get_nb_trainable_parameters=MagicMock(return_value=(10, 1000))
                )
            ),
            prepare_model_for_kbit_training=MagicMock(side_effect=lambda m: m),
        ),
    )
    LoRATrainer(boto3_module=bmod, torch_module=tmod, transformers_module=tfmod).train(
        _cfg()
    )
    assert order.index("seed_torch") < order.index("load_data")
    assert order.index("seed_transformers") < order.index("load_data")


def test_nan_loss_raises() -> None:
    cb = ForgeTrainingCallback()
    state = types.SimpleNamespace(global_step=3, epoch=1.0)
    with pytest.raises(NaNLossError):
        cb.on_log(
            None,
            state,
            None,
            logs={"loss": float("nan"), "grad_norm": 0.1, "learning_rate": 0.1},
        )


def test_gradient_norm_warning_fires(caplog: pytest.LogCaptureFixture) -> None:
    cb = ForgeTrainingCallback()
    state = types.SimpleNamespace(global_step=3, epoch=1.0)
    with caplog.at_level("WARNING"):
        cb.on_log(
            None,
            state,
            None,
            logs={"loss": 0.1, "grad_norm": 11.0, "learning_rate": 0.1},
        )
    assert "gradient explosion detected" in caplog.text


def test_qlora_path_loads_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _cfg()
    cfg = TrainingJobConfig(
        job_id=base.job_id,
        base_model_id=base.base_model_id,
        dataset_path=base.dataset_path,
        output_path=base.output_path,
        seed=base.seed,
        use_qlora=True,
        quantization_bits=4,
    )
    tmod = _torch()
    tfmod = _transformers()
    bmod = _boto3('{"instruction":"i","input":"x","output":"y"}\n')
    monkeypatch.setitem(
        __import__("sys").modules,
        "peft",
        types.SimpleNamespace(
            LoraConfig=MagicMock(),
            get_peft_model=MagicMock(
                return_value=MagicMock(
                    get_nb_trainable_parameters=MagicMock(return_value=(10, 1000))
                )
            ),
            prepare_model_for_kbit_training=MagicMock(side_effect=lambda m: m),
        ),
    )
    LoRATrainer(boto3_module=bmod, torch_module=tmod, transformers_module=tfmod).train(
        cfg
    )
    tfmod.BitsAndBytesConfig.assert_called_once()


def test_dataset_schema_validation_rejects_missing_keys() -> None:
    bad = json.dumps({"instruction": "i", "input": "x"})
    trainer = LoRATrainer(
        boto3_module=_boto3(bad + "\n"),
        torch_module=_torch(),
        transformers_module=_transformers(),
    )
    with pytest.raises(DatasetSchemaError):
        trainer.train(_cfg())
