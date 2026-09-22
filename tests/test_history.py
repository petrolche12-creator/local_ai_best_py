from localai.history import export_markdown, load_chat, new_chat, retitle, save_chat


def test_roundtrip_and_rename(tmp_path):
    chat = new_chat(
        model_id="qwen3-4b-q4",
        model_name="Qwen3 4B Q4_K_M",
        mode="fast",
        system_prompt="будь краток",
        thinking_model=True,
        repo_id="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q4_K_M.gguf",
    )
    chat.messages.append({"role": "user", "content": "привет"})
    chat.messages.append({"role": "assistant", "content": "здарова", "reasoning": "хм"})
    first = save_chat(tmp_path, chat)
    assert first.exists()
    retitle(chat, "Привет, как дела?")
    second = save_chat(tmp_path, chat)
    assert second.exists()
    assert not first.exists()
    loaded = load_chat(second)
    assert loaded.title.startswith("Привет")
    assert loaded.repo_id == "Qwen/Qwen3-4B-GGUF"
    assert loaded.messages[1]["reasoning"] == "хм"


def test_export_hides_thinking_by_default(tmp_path):
    chat = new_chat(
        model_id="m",
        model_name="M",
        mode="smart",
        system_prompt="sys",
        thinking_model=True,
    )
    chat.messages.append({"role": "assistant", "content": "ответ", "reasoning": "тайна"})
    dest = export_markdown(chat, tmp_path / "out.md")
    text = dest.read_text(encoding="utf-8")
    assert "ответ" in text
    assert "тайна" not in text
    full = export_markdown(chat, tmp_path / "full.md", include_thinking=True)
    assert "тайна" in full.read_text(encoding="utf-8")
