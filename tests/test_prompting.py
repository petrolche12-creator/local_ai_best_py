from localai.prompting import drop_old_messages, model_messages, render_chat_prompt


TEMPLATE = (
    "{% if messages[0].role == 'system' %}"
    "<|im_start|>system\n{{ messages[0].content }}<|im_end|>\n"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if not (loop.first and message.role == 'system') %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)


def test_render_disables_thinking():
    text = render_chat_prompt(
        TEMPLATE,
        [{"role": "system", "content": "будь краток"}, {"role": "user", "content": "привет"}],
        enable_thinking=False,
    )
    assert text is not None
    assert "будь краток" in text
    assert text.rstrip().endswith("</think>")
    assert "<|im_start|>user" in text


def test_render_keeps_thinking_slot_empty():
    text = render_chat_prompt(
        TEMPLATE,
        [{"role": "user", "content": "привет"}],
        enable_thinking=True,
    )
    assert text is not None
    assert "</think>" not in text
    assert text.rstrip().endswith("<|im_start|>assistant")


def test_bad_template_returns_none():
    assert render_chat_prompt("{{ missing | no_such }}", [], enable_thinking=False) is None


def test_model_messages_restore_reasoning():
    history = [{"role": "assistant", "content": "ответ", "reasoning": "ход"}]
    messages = model_messages("sys", history, thinking_model=True)
    assert messages[0]["role"] == "system"
    assert "<think>" in messages[1]["content"]
    assert messages[1]["content"].endswith("ответ")


def test_drop_old_keeps_system_and_recent():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old-a"},
        {"role": "user", "content": "new"},
    ]

    def fits(batch):
        return sum(len(item["content"]) for item in batch) <= 5

    trimmed, dropped = drop_old_messages(messages, fits)
    assert dropped == 2
    assert trimmed[0]["content"] == "s"
    assert trimmed[-1]["content"] == "new"
