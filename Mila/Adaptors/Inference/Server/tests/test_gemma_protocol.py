"""
Parity tests for the Gemma 4 native grammar renderer (gemma_protocol.py).

These pin the PARITY CONTRACT with the C++ twin (Dnn.Components.GemmaProtocol,
Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Protocol.ixx). The golden strings
in EXPECTED_* below are asserted verbatim by the matching C++ tests in
Mila/Tests/Dnn/Components/Transformers/Gemma/Gemma.Protocol.cpp -- the two suites
agreeing on the same literals is what makes "parity" mean something. Change a
golden string here and the C++ counterpart must change in the same commit.

Reference spec for the grammar is Google's canonical chat_template.jinja
(https://huggingface.co/google/gemma-4-12B-it/raw/main/chat_template.jinja).
"""
from mila_llm_server import gemma_protocol


# Argument keys render SORTED, matching the template's `| dictsort` and the C++
# twin (nlohmann::json is a std::map, so it sorts implicitly). Inputs below are
# deliberately supplied in NON-sorted order so insertion order cannot pass.
# No whitespace around ':' or ',' -- the trained format has none.
EXPECTED_SORTED_CALL = (
    '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,units:<|"|>c<|"|>}<tool_call|>'
)

# The exact case that exposed the C++/Python split: format_tool_response builds
# {result, error} in that insertion order, so Python emitted `result, error` while
# C++ emitted `error, result` for identical input.
EXPECTED_SORTED_RESPONSE = (
    '<|tool_response>response:run{error:<|"|>boom<|"|>,result:<|"|>partial<|"|>}<tool_response|>'
)


class TestArgumentOrderParity:
    def test_tool_call_arguments_render_sorted_not_insertion_ordered(self):
        rendered = gemma_protocol.format_tool_call(
            "get_weather", '{"units":"c","location":"Paris"}')

        assert rendered == EXPECTED_SORTED_CALL

    def test_tool_call_argument_order_is_input_order_independent(self):
        forward = gemma_protocol.format_tool_call(
            "get_weather", '{"location":"Paris","units":"c"}')
        reversed_input = gemma_protocol.format_tool_call(
            "get_weather", '{"units":"c","location":"Paris"}')

        assert forward == reversed_input == EXPECTED_SORTED_CALL

    def test_failed_tool_response_sorts_error_before_result(self):
        rendered = gemma_protocol.format_tool_response(
            "run", '{"output":"partial","error":"boom"}')

        assert rendered == EXPECTED_SORTED_RESPONSE

    def test_many_arguments_render_in_sorted_order(self):
        rendered = gemma_protocol.format_tool_call(
            "configure", '{"zeta":1,"alpha":2,"middle":3}')

        assert rendered == "<|tool_call>call:configure{alpha:2,middle:3,zeta:1}<tool_call|>"


class TestTrainedValueGrammar:
    """
    Non-scalar values must recurse into the trained DSL rather than collapsing to raw
    JSON. These are the cases that previously emitted plain double quotes -- untrained
    tokens -- on every call carrying an object or array parameter.
    """

    def test_array_of_strings_uses_the_trained_delimiter_not_json(self):
        rendered = gemma_protocol.format_tool_call("edit", '{"lines":["a","b"]}')

        assert rendered == '<|tool_call>call:edit{lines:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>'
        assert '"a"' not in rendered

    def test_nested_object_recurses_with_bare_keys(self):
        # Argument bodies render with escape_keys=False, so nested keys stay bare.
        rendered = gemma_protocol.format_tool_call("configure", '{"opts":{"n":1,"deep":true}}')

        assert rendered == "<|tool_call>call:configure{opts:{deep:true,n:1}}<tool_call|>"

    def test_null_and_bool_render_as_bare_literals(self):
        rendered = gemma_protocol.format_tool_call("set", '{"flag":false,"missing":null}')

        assert rendered == "<|tool_call>call:set{flag:false,missing:null}<tool_call|>"

    def test_array_of_objects_recurses(self):
        rendered = gemma_protocol.format_tool_call("todo", '{"items":[{"id":1,"tag":"x"}]}')

        assert rendered == '<|tool_call>call:todo{items:[{id:1,tag:<|"|>x<|"|>}]}<tool_call|>'

    def test_non_mapping_response_uses_value_key(self):
        # The canonical template emits `value:` (not `result:`) for a non-mapping response.
        rendered = gemma_protocol.format_tool_response("echo", "plain text")

        assert rendered == '<|tool_response>response:echo{value:<|"|>plain text<|"|>}<tool_response|>'


class TestPromptShape:
    """
    Prompt assembly, pinned against Google's canonical chat_template.jinja. These
    goldens were established by rendering that template with jinja2 and diffing
    (scripts in the session scratchpad: render_ref.py / diff_prompt.py) -- both cases
    match the reference byte-for-byte.

    The rule the two cases encode: the empty-thought prime belongs at the START of a
    fresh model turn and NOWHERE else. Priming a second one mid-turn, after a tool
    response, is off-distribution -- the model parrots it back (measured: empty-channel
    echoes, one 682-char runaway).
    """

    TOOL_SPAN = (
        '<|tool_call>call:Write{file_path:<|"|>a.txt<|"|>}<tool_call|>'
        '<|tool_response>response:Write{value:<|"|>done<|"|>}<tool_response|>'
    )

    def test_fresh_turn_ends_with_the_thought_prime(self):
        prompt = gemma_protocol.assemble_prompt(
            "S", [{"role": "user", "content": "hi"}], continue_open=False)

        assert prompt.endswith("<|turn>model\n<|channel>thought\n<channel|>")

    def test_resumed_turn_ends_at_the_tool_response(self):
        # No trailing prime AND no trailing newline -- the reference stops dead here.
        prompt = gemma_protocol.assemble_prompt(
            "S", [{"role": "user", "content": "hi"},
                  {"role": "model", "content": self.TOOL_SPAN, "tool": True}],
            continue_open=True)

        assert prompt.endswith("<tool_response|>")

    def test_resumed_turn_has_exactly_one_thought_channel(self):
        # The one the model itself opened before the call -- not a second, empty one.
        span = "<|channel>thought\n<channel|>Checking." + self.TOOL_SPAN
        prompt = gemma_protocol.assemble_prompt(
            "S", [{"role": "user", "content": "hi"},
                  {"role": "model", "content": span, "tool": True}],
            continue_open=True)

        assert prompt.count("<|channel>") == 1

    def test_resumed_turn_is_left_open(self):
        prompt = gemma_protocol.assemble_prompt(
            "S", [{"role": "user", "content": "hi"},
                  {"role": "model", "content": self.TOOL_SPAN, "tool": True}],
            continue_open=True)

        # The model turn must NOT be closed -- generation resumes it.
        assert not prompt.endswith("<turn|>")
        assert prompt.count("<|turn>model") == 1


class TestRoundTripOracle:
    """
    render -> parse identity. This is the oracle for the recursive-descent parser: the
    parser is correct exactly when it is the inverse of the renderer over the value
    grammar. Before the rewrite only flat-scalar calls round-tripped.
    """

    CASES = [
        ("flat strings", {"location": "Paris", "units": "c"}),
        ("array of strings", {"lines": ["a", "b"]}),
        ("nested object", {"opts": {"deep": True, "n": 1}}),
        ("array of objects", {"items": [{"id": 1, "tag": "x"}, {"id": 2, "tag": "y"}]}),
        ("null and bool", {"flag": False, "missing": None, "on": True}),
        ("numbers", {"count": 42, "ratio": 0.5, "negative": -7}),
        ("empty containers", {"nothing": [], "blank": {}}),
        ("string with comma", {"location": "Toronto, Canada"}),
        # The case that motivated the whole grammar: punctuation that IS grammar
        # elsewhere must survive inside a delimited string.
        ("string with braces and brackets", {"code": "func() { return [1,2]; }"}),
        ("string with plain quotes", {"say": 'he said "hi"'}),
        ("deep nesting", {"a": {"b": {"c": ["d", {"e": 1}]}}}),
        ("mixed", {"path": "/tmp/x.cpp", "opts": {"n": 3, "tags": ["p", "q"]}, "dry": False}),
    ]

    def test_render_parse_round_trip(self):
        for label, arguments in self.CASES:
            rendered = gemma_protocol.format_tool_call("t", gemma_protocol.json.dumps(arguments))
            parsed = gemma_protocol.parse_tool_call(rendered)

            assert parsed is not None, f"{label}: did not parse at all"
            back = gemma_protocol.json.loads(parsed["arguments"])
            assert back == arguments, f"{label}: {back!r} != {arguments!r}"

    def test_container_values_are_containers_not_strings(self):
        rendered = gemma_protocol.format_tool_call("edit", '{"lines":["a","b"]}')
        back = gemma_protocol.json.loads(gemma_protocol.parse_tool_call(rendered)["arguments"])

        assert back["lines"] == ["a", "b"]
        assert isinstance(back["lines"], list)


class TestContainerShreddingRegression:
    """
    Regression for the live failure seen in the 2026-07-16 Claude Code A/B, where the old
    parser truncated a container AND spilled its remaining contents out as sibling
    arguments, corrupting keys that had nothing to do with it. Observed verbatim:
        {"metadata": "{alpha:Done", "beta": "Done", "gamma": "Done",
         "},subject": "Create todo list"}

    WILD below RECONSTRUCTS that shape (the exact bytes the model emitted were not
    captured -- only the parsed wreckage was). Verified to reproduce the shredding under
    the old parser: metadata truncates to "{alpha:Done" and beta/gamma leak as siblings.
    """

    WILD = ('<|tool_call>call:TaskUpdate{metadata:{alpha:Done,beta:Done,gamma:Done},'
            'subject:<|"|>Create todo list<|"|>,taskId:1}<tool_call|>')

    def test_wild_container_parses_as_a_container(self):
        parsed = gemma_protocol.parse_tool_call(self.WILD)
        args = gemma_protocol.json.loads(parsed["arguments"])

        assert args["metadata"] == {"alpha": "Done", "beta": "Done", "gamma": "Done"}

    def test_sibling_arguments_survive_a_container(self):
        # The regression that mattered most: `subject` was destroyed into `},subject`.
        args = gemma_protocol.json.loads(gemma_protocol.parse_tool_call(self.WILD)["arguments"])

        assert args["subject"] == "Create todo list"
        assert args["taskId"] == 1
        assert set(args) == {"metadata", "subject", "taskId"}

    def test_no_shredded_sibling_keys_leak(self):
        args = gemma_protocol.json.loads(gemma_protocol.parse_tool_call(self.WILD)["arguments"])

        for bogus in ("beta", "gamma", "},subject"):
            assert bogus not in args


class TestParserTolerance:
    """
    The parser reads what the model EMITS, which is inconsistent -- it must stay lenient
    on input while the renderer stays strict on output.
    """

    def test_whitespace_around_separators_is_tolerated(self):
        # Off-spec spacing, but the model does it; accepting it costs nothing.
        parsed = gemma_protocol.parse_tool_call(
            '<|tool_call>call:t{a: <|"|>x<|"|> , b: [1, 2]}<tool_call|>')
        args = gemma_protocol.json.loads(parsed["arguments"])

        assert args == {"a": "x", "b": [1, 2]}

    def test_plain_quoted_strings_still_parse(self):
        parsed = gemma_protocol.parse_tool_call('<|tool_call>call:t{a:"x",b:["y"]}<tool_call|>')

        assert gemma_protocol.json.loads(parsed["arguments"]) == {"a": "x", "b": ["y"]}

    def test_truncated_body_keeps_what_parsed(self):
        # A cut-off call should surface partially, not blank the turn.
        parsed = gemma_protocol.parse_tool_call(
            '<|tool_call>call:t{a:<|"|>x<|"|>,b:{c:1}<tool_call|>')

        assert parsed is not None
        assert gemma_protocol.json.loads(parsed["arguments"])["a"] == "x"

    def test_pipe_token_scrub_reaches_nested_strings(self):
        parsed = gemma_protocol.parse_tool_call(
            '<|tool_call>call:t{files:[<|"|>foo.cpp<|><|"|>]}<tool_call|>')

        assert gemma_protocol.json.loads(parsed["arguments"])["files"] == ["foo.cpp"]


class TestTrainedToolDeclarations:
    """
    Declarations mirror format_function_declaration in the canonical template: a full
    parameters DSL with UPPERCASED types, not the compact JSON blob this used to emit.
    Declaration grammar is MIS-only -- the C++ runtime module has no declaration
    renderer -- so there is no parity counterpart for this class.
    """

    WEATHER_TOOL = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "The city"}},
                "required": ["city"],
            },
        },
    }]

    def test_declaration_renders_the_full_trained_parameter_dsl(self):
        rendered = gemma_protocol.build_tool_injection(
            self.WEATHER_TOOL)

        assert rendered == (
            '<|tool>declaration:get_weather{description:<|"|>Look up weather<|"|>,'
            'parameters:{properties:{city:{description:<|"|>The city<|"|>,'
            'type:<|"|>STRING<|"|>}},required:[<|"|>city<|"|>],'
            'type:<|"|>OBJECT<|"|>}}<tool|>'
        )

    def test_declaration_emits_no_raw_json_blob(self):
        rendered = gemma_protocol.build_tool_injection(
            self.WEATHER_TOOL)

        # The old approximation leaked JSON punctuation the model never trained on.
        assert '"type"' not in rendered
        assert '"properties"' not in rendered

    def test_enum_property_renders_in_the_trained_grammar(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "pick",
                "description": "Pick one",
                "parameters": {
                    "type": "object",
                    "properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}},
                },
            },
        }]
        rendered = gemma_protocol.build_tool_injection(tools)

        assert 'enum:[<|"|>fast<|"|>,<|"|>slow<|"|>]' in rendered
        assert 'type:<|"|>STRING<|"|>' in rendered

    def test_array_property_renders_items_block(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "tag",
                "description": "Tag things",
                "parameters": {
                    "type": "object",
                    "properties": {"names": {"type": "array", "items": {"type": "string"}}},
                },
            },
        }]
        rendered = gemma_protocol.build_tool_injection(tools)

        assert 'names:{items:{type:<|"|>STRING<|"|>},type:<|"|>ARRAY<|"|>}' in rendered

    def test_declarations_concatenate_with_no_separator(self):
        # The template appends declarations straight onto the system turn; <|tool> is an
        # atomic special token and needs no whitespace to separate it from the prose.
        tools = self.WEATHER_TOOL + [{
            "type": "function",
            "function": {"name": "ping", "description": "Ping", "parameters": {}},
        }]
        rendered = gemma_protocol.build_tool_injection(tools)

        assert rendered.startswith("<|tool>")
        assert "<tool|><|tool>" in rendered
        assert rendered.endswith('<|tool>declaration:ping{description:<|"|>Ping<|"|>}<tool|>')


class TestRenderRoundTrip:
    def test_sorted_render_still_round_trips_through_the_parser(self):
        rendered = gemma_protocol.format_tool_call(
            "get_weather", '{"units":"c","location":"Paris"}')
        parsed = gemma_protocol.parse_tool_call(rendered)

        assert parsed is not None
        assert parsed["name"] == "get_weather"
        assert gemma_protocol.json.loads(parsed["arguments"]) == {
            "location": "Paris", "units": "c"}

    def test_string_values_use_the_trained_delimiter(self):
        rendered = gemma_protocol.format_tool_call("run", '{"cmd":"ls -F"}')

        assert '<|"|>ls -F<|"|>' in rendered
        assert '"ls -F"' not in rendered
