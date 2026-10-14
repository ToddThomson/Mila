// Parity and contract tests for Qwen 3.8's chat template behind the Protocol interface
// (Dnn.Models.QwenChatProtocol).
//
// The goldens are the checkpoint's own chat_template.jinja rendered by transformers'
// render_jinja_template -- the renderer apply_chat_template calls -- and are recorded by
// Mila/Tools/ChatTemplates/record_template_goldens.py, whose cases these histories mirror one for one.
// Re-record rather than edit: a golden that disagrees with this renderer is a finding about the
// renderer, never about the template (ModelHandle.md 3.4).

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <string_view>
#include <vector>

import Mila;
import nlohmann.json;

namespace Mila::Tests::Dnn::Models
{
    namespace Conversation = Mila::Dnn::Conversation;

    using Mila::Dnn::Qwen::QwenChatProtocol;

    namespace
    {
        // ---- Goldens: record_template_goldens.py qwen ---------------------------
        constexpr std::string_view kPlain =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the capital of Australia?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kSystemGolden =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "You are a concise assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the capital of Australia?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kTrimmed =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "Be brief.<|im_end|>\n"
            "<|im_start|>user\n"
            "hi<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kMultiTurn =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
            "<|im_start|>user\n"
            "Capital of Australia?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n"
            "Canberra.<|im_end|>\n"
            "<|im_start|>user\n"
            "And of Canada?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kTools =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "# Tools\n"
            "\n"
            "You have access to the following functions:\n"
            "\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city\"}, \"units\": {\"type\": \"string\", \"enum\": [\"c\", \"f\"]}}, \"required\": [\"city\"]}}}\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_time\", \"description\": \"Get the local time in a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n"
            "</tools>\n"
            "\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
            "\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT>\n"
            "\n"
            "You are a concise assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the weather in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kToolsNoSystem =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "# Tools\n"
            "\n"
            "You have access to the following functions:\n"
            "\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city\"}, \"units\": {\"type\": \"string\", \"enum\": [\"c\", \"f\"]}}, \"required\": [\"city\"]}}}\n"
            "</tools>\n"
            "\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
            "\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT><|im_end|>\n"
            "<|im_start|>user\n"
            "What is the weather in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kOneCallResume =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "# Tools\n"
            "\n"
            "You have access to the following functions:\n"
            "\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city\"}, \"units\": {\"type\": \"string\", \"enum\": [\"c\", \"f\"]}}, \"required\": [\"city\"]}}}\n"
            "</tools>\n"
            "\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
            "\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT>\n"
            "\n"
            "You are a concise assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the weather in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\n"
            "Paris\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call><|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Sunny, 21 C\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kTwoCallsResume =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "# Tools\n"
            "\n"
            "You have access to the following functions:\n"
            "\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city\"}, \"units\": {\"type\": \"string\", \"enum\": [\"c\", \"f\"]}}, \"required\": [\"city\"]}}}\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_time\", \"description\": \"Get the local time in a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n"
            "</tools>\n"
            "\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
            "\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT><|im_end|>\n"
            "<|im_start|>user\n"
            "Weather and time in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n"
            "Let me check both.\n"
            "\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\n"
            "Paris\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=get_time>\n"
            "<parameter=city>\n"
            "Paris\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call><|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Sunny, 21 C\n"
            "</tool_response>\n"
            "<tool_response>\n"
            "14:05\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kTwoCallsOutOfOrder =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "# Tools\n"
            "\n"
            "You have access to the following functions:\n"
            "\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city\"}, \"units\": {\"type\": \"string\", \"enum\": [\"c\", \"f\"]}}, \"required\": [\"city\"]}}}\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_time\", \"description\": \"Get the local time in a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n"
            "</tools>\n"
            "\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
            "\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT><|im_end|>\n"
            "<|im_start|>user\n"
            "Weather and time in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\n"
            "Paris\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=get_time>\n"
            "<parameter=city>\n"
            "Paris\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call><|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "14:05\n"
            "</tool_response>\n"
            "<tool_response>\n"
            "Sunny, 21 C\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kClosedCycle =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "# Tools\n"
            "\n"
            "You have access to the following functions:\n"
            "\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city\"}, \"units\": {\"type\": \"string\", \"enum\": [\"c\", \"f\"]}}, \"required\": [\"city\"]}}}\n"
            "</tools>\n"
            "\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
            "\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT><|im_end|>\n"
            "<|im_start|>user\n"
            "What is the weather in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "I need the weather tool.\n"
            "</think>\n"
            "\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\n"
            "Paris\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call><|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Sunny, 21 C\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "The tool said sunny.\n"
            "</think>\n"
            "\n"
            "It is sunny and 21 C.<|im_end|>\n"
            "<|im_start|>user\n"
            "And tomorrow?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kTypedArguments =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
            "<|im_start|>user\n"
            "Configure it.<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n"
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=tokens>\n"
            "4096\n"
            "</parameter>\n"
            "<parameter=enabled>\n"
            "true\n"
            "</parameter>\n"
            "<parameter=mode>\n"
            "high\n"
            "</parameter>\n"
            "<parameter=opts>\n"
            "{\"z\": 1, \"a\": [1, 2]}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call><|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "ok\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kJsonResult =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "# Tools\n"
            "\n"
            "You have access to the following functions:\n"
            "\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city\"}, \"units\": {\"type\": \"string\", \"enum\": [\"c\", \"f\"]}}, \"required\": [\"city\"]}}}\n"
            "</tools>\n"
            "\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
            "\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT><|im_end|>\n"
            "<|im_start|>user\n"
            "What is the weather in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\n"
            "Paris\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call><|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "{\"output\": \"Sunny\", \"chunk_id\": 7}\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kThinkingOff =
            "<|im_start|>system\n"
            "You are a concise assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the capital of Australia?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n";

        constexpr std::string_view kEffortLow =
            "<|im_start|>system\n"
            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.\n"
            "\n"
            "You are a concise assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the capital of Australia?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kEffortMedium =
            "<|im_start|>system\n"
            "You are a concise assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the capital of Australia?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kEffortXhigh =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "You are a concise assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the capital of Australia?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kEffortLowThinkingOff =
            "<|im_start|>system\n"
            "You are a concise assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the capital of Australia?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n";

        constexpr std::string_view kClosedCyclePreserveOff =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n"
            "\n"
            "# Tools\n"
            "\n"
            "You have access to the following functions:\n"
            "\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city\"}, \"units\": {\"type\": \"string\", \"enum\": [\"c\", \"f\"]}}, \"required\": [\"city\"]}}}\n"
            "</tools>\n"
            "\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
            "\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT><|im_end|>\n"
            "<|im_start|>user\n"
            "What is the weather in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\n"
            "Paris\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call><|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Sunny, 21 C\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "It is sunny and 21 C.<|im_end|>\n"
            "<|im_start|>user\n"
            "And tomorrow?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";

        constexpr std::string_view kOneCallResumeThinkingOff =
            "<|im_start|>system\n"
            "# Tools\n"
            "\n"
            "You have access to the following functions:\n"
            "\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The city\"}, \"units\": {\"type\": \"string\", \"enum\": [\"c\", \"f\"]}}, \"required\": [\"city\"]}}}\n"
            "</tools>\n"
            "\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
            "\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT>\n"
            "\n"
            "You are a concise assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the weather in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>\n"
            "Paris\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call><|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "Sunny, 21 C\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
            "\n"
            "</think>\n"
            "\n";

        // ---- The recorder's cases ------------------------------------------------

        // Key order matters: the template writes each signature through tojson in the order given.
        const std::string kWeather = R"({"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city"}, "units": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}})";
        const std::string kTime = R"({"type": "function", "function": {"name": "get_time", "description": "Get the local time in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})";

        std::string tools( std::initializer_list<std::string> schemas )
        {
            std::string json = "[";

            for ( const auto& schema : schemas )
            {
                json += json.size() == 1 ? "" : ",";
                json += schema;
            }

            return json + "]";
        }

        Conversation::Turn system( std::string content )
        {
            return { Conversation::Role::System, std::move( content ) };
        }

        Conversation::Turn user( std::string content )
        {
            return { Conversation::Role::User, std::move( content ) };
        }

        Conversation::Turn assistant( std::string content, std::vector<Conversation::ToolCall> calls = {}, std::string reasoning = {} )
        {
            return { Conversation::Role::Assistant, std::move( content ), std::move( calls ), {}, std::move( reasoning ) };
        }

        Conversation::Turn result( std::string name, std::string content )
        {
            return { Conversation::Role::Tool, std::move( content ), {}, std::move( name ) };
        }

        const Conversation::ToolCall kWeatherCall{ "get_weather", R"({"city": "Paris"})" };
        const Conversation::ToolCall kTimeCall{ "get_time", R"({"city": "Paris"})" };

        const std::string kSystem = "You are a concise assistant.";
        const std::string kCapital = "What is the capital of Australia?";
        const std::string kAsk = "What is the weather in Paris?";
        const std::string kAskBoth = "Weather and time in Paris?";

        std::vector<Conversation::Turn> oneCallResume()
        {
            return { system( kSystem ), user( kAsk ), assistant( "", { kWeatherCall } ), result( "get_weather", "Sunny, 21 C" ) };
        }

        std::vector<Conversation::Turn> closedCycle()
        {
            return {
                user( kAsk ),
                assistant( "", { kWeatherCall }, "I need the weather tool." ),
                result( "get_weather", "Sunny, 21 C" ),
                assistant( "It is sunny and 21 C.", {}, "The tool said sunny." ),
                user( "And tomorrow?" ) };
        }

        Conversation::PromptOptions options( std::string tools_json = {}, std::map<std::string, std::string> variables = {} )
        {
            return { std::move( variables ), std::move( tools_json ) };
        }

        std::string render( const std::vector<Conversation::Turn>& history, const Conversation::PromptOptions& prompt_options = {} )
        {
            return QwenChatProtocol{}.formatPrompt( history, prompt_options ).text;
        }
    }

    // ---- P1: byte-identical to the template ------------------------------------

    TEST( QwenChatProtocolParity, Plain )
    {
        EXPECT_EQ( render( { user( kCapital ) } ), kPlain );
    }

    TEST( QwenChatProtocolParity, System )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kCapital ) } ), kSystemGolden );
    }

    TEST( QwenChatProtocolParity, ContentIsTrimmed )
    {
        EXPECT_EQ( render( { system( "  Be brief.\n" ), user( "  hi  \n" ) } ), kTrimmed );
    }

    TEST( QwenChatProtocolParity, MultiTurn )
    {
        EXPECT_EQ( render( { user( "Capital of Australia?" ), assistant( "Canberra." ), user( "And of Canada?" ) } ), kMultiTurn );
    }

    TEST( QwenChatProtocolParity, ToolsKeepTheirKeyOrderAndTojsonSeparators )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kAsk ) }, options( tools( { kWeather, kTime } ) ) ), kTools );
    }

    TEST( QwenChatProtocolParity, ToolsWithNoSystemTurn )
    {
        EXPECT_EQ( render( { user( kAsk ) }, options( tools( { kWeather } ) ) ), kToolsNoSystem );
    }

    TEST( QwenChatProtocolParity, OneCallResume )
    {
        EXPECT_EQ( render( oneCallResume(), options( tools( { kWeather } ) ) ), kOneCallResume );
    }

    TEST( QwenChatProtocolParity, TwoCallsWithContentAndConsecutiveResultsInOneUserTurn )
    {
        EXPECT_EQ( render( {
            user( kAskBoth ),
            assistant( "Let me check both.", { kWeatherCall, kTimeCall } ),
            result( "get_weather", "Sunny, 21 C" ),
            result( "get_time", "14:05" ) }, options( tools( { kWeather, kTime } ) ) ), kTwoCallsResume );
    }

    TEST( QwenChatProtocolParity, ResultsRenderInTheOrderGiven )
    {
        EXPECT_EQ( render( {
            user( kAskBoth ),
            assistant( "", { kWeatherCall, kTimeCall } ),
            result( "get_time", "14:05" ),
            result( "get_weather", "Sunny, 21 C" ) }, options( tools( { kWeather, kTime } ) ) ), kTwoCallsOutOfOrder );
    }

    TEST( QwenChatProtocolParity, ClosedCycleRendersKeptReasoning )
    {
        EXPECT_EQ( render( closedCycle(), options( tools( { kWeather } ) ) ), kClosedCycle );
    }

    TEST( QwenChatProtocolParity, TypedArgumentsKeepTheirOrderThroughTojson )
    {
        EXPECT_EQ( render( {
            user( "Configure it." ),
            assistant( "", { { "configure", R"({"tokens": 4096, "enabled": true, "mode": "high", "opts": {"z": 1, "a": [1, 2]}})" } } ),
            result( "configure", "ok" ) } ), kTypedArguments );
    }

    TEST( QwenChatProtocolParity, JsonResultIsRenderedAsWritten )
    {
        EXPECT_EQ( render( {
            user( kAsk ),
            assistant( "", { kWeatherCall } ),
            result( "get_weather", R"({"output": "Sunny", "chunk_id": 7})" ) }, options( tools( { kWeather } ) ) ), kJsonResult );
    }

    TEST( QwenChatProtocolParity, ThinkingOff )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kCapital ) }, options( {}, { { "enable_thinking", "false" } } ) ), kThinkingOff );
    }

    TEST( QwenChatProtocolParity, EffortLow )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kCapital ) }, options( {}, { { "reasoning_effort", "low" } } ) ), kEffortLow );
    }

    TEST( QwenChatProtocolParity, EffortMediumWritesNoInstruction )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kCapital ) }, options( {}, { { "reasoning_effort", "medium" } } ) ), kEffortMedium );
    }

    TEST( QwenChatProtocolParity, EffortXhighIsTheDefault )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kCapital ) }, options( {}, { { "reasoning_effort", "xhigh" } } ) ), kEffortXhigh );
        EXPECT_EQ( render( { system( kSystem ), user( kCapital ) } ), kEffortXhigh );
    }

    TEST( QwenChatProtocolParity, EffortIsIgnoredWithThinkingOff )
    {
        EXPECT_EQ( render( { system( kSystem ), user( kCapital ) },
            options( {}, { { "enable_thinking", "false" }, { "reasoning_effort", "low" } } ) ), kEffortLowThinkingOff );
    }

    TEST( QwenChatProtocolParity, PreserveThinkingOffDropsReasoningBeforeTheLastQuestion )
    {
        EXPECT_EQ( render( closedCycle(), options( tools( { kWeather } ), { { "preserve_thinking", "false" } } ) ), kClosedCyclePreserveOff );
    }

    TEST( QwenChatProtocolParity, OneCallResumeThinkingOff )
    {
        EXPECT_EQ( render( oneCallResume(), options( tools( { kWeather } ), { { "enable_thinking", "false" } } ) ), kOneCallResumeThinkingOff );
    }

    // ---- Template variables -------------------------------------------------------

    TEST( QwenChatProtocolVariables, AreTheTemplatesOwnWithItsDefaults )
    {
        const QwenChatProtocol protocol;
        std::map<std::string, std::string> defaults;

        for ( const auto& variable : protocol.templateVariables() )
        {
            defaults[ variable.name ] = variable.default_value;
        }

        EXPECT_EQ( defaults, ( std::map<std::string, std::string>{
            { "enable_thinking", "true" }, { "reasoning_effort", "xhigh" }, { "preserve_thinking", "true" } } ) );
    }

    TEST( QwenChatProtocolVariables, AValueTheTemplateDoesNotAcceptIsRefused )
    {
        // OpenAI's "high" is not one of this template's levels, and is not approximated to "xhigh".
        EXPECT_THROW( (void)render( { user( kCapital ) }, options( {}, { { "reasoning_effort", "high" } } ) ), std::invalid_argument );
    }

    TEST( QwenChatProtocolVariables, AVariableTheTemplateDoesNotTakeIsRefused )
    {
        EXPECT_THROW( (void)render( { user( kCapital ) }, options( {}, { { "thinking_budget", "1024" } } ) ), std::invalid_argument );
    }

    TEST( QwenChatProtocolVariables, TheReplyBeginsInReasoningExactlyWhenThinkingIsOn )
    {
        const QwenChatProtocol protocol;
        const std::vector<Conversation::Turn> history{ user( kCapital ) };

        EXPECT_TRUE( protocol.formatPrompt( history, {} ).reply_begins_in_reasoning );
        EXPECT_FALSE( protocol.formatPrompt( history, options( {}, { { "enable_thinking", "false" } } ) ).reply_begins_in_reasoning );
    }

    // ---- Histories the template refuses -------------------------------------------

    TEST( QwenChatProtocolRefusal, ASystemTurnAfterTheFirstIsRefused )
    {
        EXPECT_THROW( (void)render( { user( kCapital ), system( kSystem ) } ), std::invalid_argument );
    }

    TEST( QwenChatProtocolRefusal, AHistoryWithNoUserQuestionIsRefused )
    {
        EXPECT_THROW( (void)render( { system( kSystem ) } ), std::invalid_argument );
    }

    // ---- parseReply ----------------------------------------------------------------

    namespace
    {
        Conversation::Reply parse( std::string_view text, bool begins_in_reasoning )
        {
            return QwenChatProtocol{}.parseReply( text, { {}, begins_in_reasoning } );
        }
    }

    TEST( QwenChatProtocolReply, AReplyThatBeganInsideTheChannelSplitsAtItsClose )
    {
        const auto reply = parse( "The capital is Canberra.\n</think>\n\nCanberra.", true );

        EXPECT_EQ( reply.reasoning, "The capital is Canberra." );
        EXPECT_EQ( reply.content, "Canberra." );
        EXPECT_TRUE( reply.tool_calls.empty() );
    }

    TEST( QwenChatProtocolReply, AReplyCutOffInsideTheChannelIsAllReasoning )
    {
        const auto reply = parse( "Still thinking about", true );

        EXPECT_EQ( reply.reasoning, "Still thinking about" );
        EXPECT_TRUE( reply.content.empty() );
    }

    TEST( QwenChatProtocolReply, EveryCallIsReturnedInOrder )
    {
        const auto reply = parse(
            "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_time>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", false );

        ASSERT_EQ( reply.tool_calls.size(), 2u );
        EXPECT_EQ( reply.tool_calls[ 0 ].name, "get_weather" );
        EXPECT_EQ( reply.tool_calls[ 1 ].name, "get_time" );
        EXPECT_EQ( nlohmann::json::parse( reply.tool_calls[ 1 ].arguments ).at( "city" ), "Paris" );
        EXPECT_EQ( reply.content, "Checking." );
        EXPECT_TRUE( reply.malformed_tool_calls.empty() );
    }

    TEST( QwenChatProtocolReply, AMalformedCallIsReportedNotThrown )
    {
        const std::string span = "<tool_call>\nget_weather\n</tool_call>";
        Conversation::Reply reply;

        EXPECT_NO_THROW( reply = parse( "Sure. " + span, false ) );
        ASSERT_EQ( reply.malformed_tool_calls.size(), 1u );
        EXPECT_EQ( reply.malformed_tool_calls[ 0 ], span );
        EXPECT_TRUE( reply.tool_calls.empty() );
        EXPECT_EQ( reply.content, "Sure." );
    }

    TEST( QwenChatProtocolReply, AnUnclosedCallIsReportedAsMalformed )
    {
        const auto reply = parse( "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar", false );

        ASSERT_EQ( reply.malformed_tool_calls.size(), 1u );
        EXPECT_TRUE( reply.tool_calls.empty() );
    }

    TEST( QwenChatProtocolReply, AParsedCallRendersBackAsTheTemplateWritesIt )
    {
        const auto reply = parse( "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", false );

        ASSERT_EQ( reply.tool_calls.size(), 1u );

        const std::vector<Conversation::Turn> history{
            system( kSystem ), user( kAsk ), assistant( "", reply.tool_calls ), result( "get_weather", "Sunny, 21 C" ) };

        EXPECT_EQ( render( history, options( tools( { kWeather } ) ) ), kOneCallResume );
    }

    TEST( QwenChatProtocolResult, IsTheTemplatesResponseSpan )
    {
        EXPECT_EQ( QwenChatProtocol{}.formatToolResult( "get_weather", " Sunny, 21 C\n" ), "<tool_response>\nSunny, 21 C\n</tool_response>" );
    }
}
