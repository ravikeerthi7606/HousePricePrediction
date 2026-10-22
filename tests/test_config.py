
from src.HousePrice.config_loader import load_config


def test_config_has_required_sections():
    config = load_config()

    assert "project" in config
    assert "data" in config
    assert "training" in config
    assert "model" in config


def test_target_column_is_configured():
    config = load_config()

    assert config["data"]["target"] == "SalePrice"
