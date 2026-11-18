"""Unit tests for training configuration, reproducibility, and device setup."""

from pathlib import Path
import tempfile
import torch

from src.training.config import TrainingConfig
from src.training.reproducibility import get_device, get_system_info, set_seed


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.seed == 42
    assert config.batch_size == 32
    assert config.epochs == 100
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.hidden_dim == 128
    assert config.num_layers == 3
    assert config.dropout == 0.1
    assert config.early_stopping_patience == 10
    assert config.num_tasks == 12


def test_training_config_serialization():
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "test_config.json"
        config = TrainingConfig(
            seed=123,
            batch_size=64,
            learning_rate=5e-4,
            hidden_dim=256,
        )
        config.save_json(json_path)
        assert json_path.exists()

        loaded = TrainingConfig.load_json(json_path)
        assert loaded.seed == 123
        assert loaded.batch_size == 64
        assert loaded.learning_rate == 5e-4
        assert loaded.hidden_dim == 256
        assert loaded.to_dict() == config.to_dict()


def test_training_config_model_kwargs():
    config = TrainingConfig(hidden_dim=64, num_layers=2, dropout=0.2, pooling="sum")
    kwargs = config.to_model_kwargs()
    assert kwargs["node_feature_dim"] == 19
    assert kwargs["edge_feature_dim"] == 18
    assert kwargs["hidden_dim"] == 64
    assert kwargs["num_layers"] == 2
    assert kwargs["pooling"] == "sum"
    assert kwargs["dropout"] == 0.2


def test_set_seed_determinism():
    set_seed(42)
    a1 = torch.randn(5, 5)
    set_seed(42)
    a2 = torch.randn(5, 5)
    assert torch.allclose(a1, a2)


def test_get_device():
    cpu_dev = get_device("cpu")
    assert cpu_dev.type == "cpu"

    auto_dev = get_device("auto")
    assert isinstance(auto_dev, torch.device)


def test_get_system_info():
    info = get_system_info()
    assert "torch_version" in info
    assert "python_version" in info
    assert "cuda_available" in info
