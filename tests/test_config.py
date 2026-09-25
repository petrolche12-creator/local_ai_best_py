from localai.config import Settings, load_settings, resolve_paths, save_settings
from localai.prompts import DEFAULT_PROMPT


def test_settings_roundtrip(tmp_path):
    paths = resolve_paths(Settings(), root=tmp_path)
    settings = Settings(mode="smart", temperature=0.4, system_prompt="свой", n_gpu_layers="ALL")
    save_settings(paths, settings)
    loaded = load_settings(paths)
    assert loaded.mode == "smart"
    assert loaded.temperature == 0.4
    assert loaded.system_prompt == "свой"
    assert loaded.n_gpu_layers == "all"


def test_broken_config_falls_back(tmp_path):
    paths = resolve_paths(Settings(), root=tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.config.write_text("{", encoding="utf-8")
    loaded = load_settings(paths)
    assert loaded.system_prompt == DEFAULT_PROMPT
