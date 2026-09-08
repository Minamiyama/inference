from ..minicpm5_tool_parser import MiniCPM5ToolParser


def test_extracts_minicpm5_xml_tool_calls():
    parser = MiniCPM5ToolParser()

    result = parser.extract_tool_calls(
        'Before <function name="weather"><param name="city">"Beijing"</param>'
        '<param name="days">3</param><param name="note"><![CDATA[a < b & c]]>'
        "</param></function> after"
    )

    assert result == [
        ("Before ", None, None),
        (None, "weather", {"city": "Beijing", "days": 3, "note": "a < b & c"}),
        (" after", None, None),
    ]


def test_waits_for_a_complete_minicpm5_tool_call_when_streaming():
    parser = MiniCPM5ToolParser()
    partial = 'before <function name="weather"><param name="city">Beijing'
    complete = partial + "</param></function>"

    assert parser.extract_tool_calls_streaming([], partial, partial) == (
        "before ",
        None,
        None,
    )
    assert parser.extract_tool_calls_streaming(
        [partial], complete, "</param></function>"
    ) == (
        None,
        "weather",
        {"city": "Beijing"},
        0,
    )
