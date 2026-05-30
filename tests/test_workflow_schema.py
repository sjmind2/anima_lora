import pytest
from pathlib import Path
from workflow.config import load_schema


class TestSchemaLoading:
    def test_load_preprocess_schema(self):
        schema = load_schema("preprocess")
        assert schema["type"] == "preprocess"
        groups = schema["groups"]
        group_names = [g["name"] for g in groups]
        assert "data_source" in group_names
        assert "bucket" in group_names

    def test_load_train_common_schema(self):
        schema = load_schema("train_common")
        assert schema["type"] == "train_common"
        all_fields = []
        for g in schema["groups"]:
            all_fields.extend(g["fields"])
        keys = [f["key"] for f in all_fields]
        assert "learning_rate" in keys
        assert "max_train_epochs" in keys
        assert "optimizer_type" in keys

    def test_load_train_lokr_schema(self):
        schema = load_schema("train_lokr")
        assert schema["method"] == "lokr"
        all_fields = []
        for g in schema["groups"]:
            all_fields.extend(g["fields"])
        keys = [f["key"] for f in all_fields]
        assert "lokr_factor" in keys
        assert "decompose_both" in keys
        assert "scale_weight_norms" in keys

    def test_load_infrastructure_schema(self):
        schema = load_schema("infrastructure")
        assert schema["type"] == "infrastructure"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent_method")
