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


def test_preserves_cdata_with_literal_closing_tags():
    parser = MiniCPM5ToolParser()
    output = (
        '<function name="write_file"><param name="content"><![CDATA['
        "example </param> and </function> end]]></param></function>"
    )

    assert parser.extract_tool_calls(output) == [
        (None, "write_file", {"content": "example </param> and </function> end"})
    ]


def test_streaming_waits_for_cdata_with_literal_closing_tags():
    parser = MiniCPM5ToolParser()
    complete = (
        '<function name="write_file"><param name="content"><![CDATA['
        "example </param> and </function> end]]></param></function>"
    )
    split_param = complete.index("</param>") + len("</par")
    split_function = complete.index("</function>") + len("</func")
    partial_param = complete[:split_param]
    partial_function = complete[:split_function]

    assert parser.extract_tool_calls_streaming([], partial_param, partial_param) is None
    assert (
        parser.extract_tool_calls_streaming(
            [partial_param], partial_function, partial_function[len(partial_param) :]
        )
        is None
    )
    assert parser.extract_tool_calls_streaming(
        [partial_function], complete, complete[len(partial_function) :]
    ) == (
        None,
        "write_file",
        {"content": "example </param> and </function> end"},
        0,
    )
