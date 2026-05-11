from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained( model_name )
    model = AutoModelForCausalLM.from_pretrained( model_name, torch_dtype=torch.bfloat16 )
    model.eval()

    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Mila, an AI assistant that has access to tools. When a user request requires real-time or external information, use the appropriate tool. Always use a tool when one is relevant rather than guessing at the answer. Return tool results to the user clearly and concisely.

You have access to the following tools:
[
  {
    "name": "get_weather",
    "description": "Get the current weather conditions for a specified location.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "City and country, e.g. London, UK"
        },
        "units": {
          "type": "string",
          "description": "Temperature units: celsius or fahrenheit. Default: celsius."
        }
      },
      "required": [ "location" ]
    }
  }
]

When calling a tool, you MUST respond using exactly this format and nothing else:
<|tool_call|>{"name": "tool_name", "arguments": {"param": "value"}}<|eom_id|>
Do not describe what you are doing. Do not include any other text. Emit only the tool call token sequence and stop.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the temperature in Vancouver today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    input_ids = tokenizer.encode( prompt, add_special_tokens=False, return_tensors="pt" )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=64,
            do_sample=False,
            temperature=1.0,
        )

    generated_ids = output_ids[ 0, input_ids.shape[ -1 ]: ]

    print( f"Generated token IDs: { generated_ids.tolist() }" )
    print( f"Generated text:      { tokenizer.decode( generated_ids ) }" )

if __name__ == "__main__":
    main()