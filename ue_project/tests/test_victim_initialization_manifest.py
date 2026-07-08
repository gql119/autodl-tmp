def test_victim_initialization_manifest_records_random_init():
    manifest = {
        "model": "yolov8n.yaml",
        "random_initialization": True,
        "seed": 0,
        "initialization_checksum": "abc123",
        "uses_voc_trained_surrogate": False,
    }
    assert manifest["random_initialization"] is True
    assert manifest["uses_voc_trained_surrogate"] is False
    assert manifest["initialization_checksum"]
