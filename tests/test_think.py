from localai.think import ThinkFilter, strip_think


def test_strip_complete_block():
    answer, reasoning = strip_think("Привет <think>секрет</think> мир")
    assert answer == "Привет  мир"
    assert reasoning == "секрет"


def test_split_across_tokens():
    filt = ThinkFilter()
    chunks = []
    for part in ["Hello <thi", "nk>secret</thi", "nk> world"]:
        chunks.extend(filt.feed(part))
    chunks.extend(filt.finish())
    answer = "".join(text for kind, text in chunks if kind == "answer")
    thinking = "".join(text for kind, text in chunks if kind == "think")
    assert answer == "Hello  world"
    assert thinking == "secret"
    assert filt.answer == "Hello  world"


def test_no_tag_is_answer():
    answer, reasoning = strip_think("просто текст")
    assert answer == "просто текст"
    assert reasoning == ""


def test_unclosed_think_stays_reasoning():
    filt = ThinkFilter()
    filt.feed("<think>ещё думаю")
    filt.finish()
    assert filt.thinking == "ещё думаю"
    assert filt.answer == ""
