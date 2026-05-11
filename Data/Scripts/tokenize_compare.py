from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained( model_name )
    model = AutoModelForCausalLM.from_pretrained( model_name, torch_dtype=torch.bfloat16 )
    model.eval()

    tools_json = """[
  {
    "name": "get_weather",
    "description": "Get the current weather conditions for a specified location.",
    "parameters": {
      "type": "dict",
      "required": [ "location" ],
      "properties": {
        "location": {
          "type": "string",
          "description": "City and country, e.g. London, UK"
        },
        "units": {
          "type": "string",
          "description": "Temperature units: celsius or fahrenheit. Default: celsius."
        }
      }
    }
  }
]"""

    system_prompt = f"""You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke:
{tools_json}"""

    user_message = "What is the temperature in Vancouver today?"

    tool_call = '[get_weather(location="Vancouver, Canada", units="celsius")]'
    tool_result = "The current weather in Vancouver, Canada is 22C and sunny."

    # --- Turn 1: user question ---
    turn1_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    print( "=== Turn 1: Tool call ===" )

    input_ids = tokenizer.encode( turn1_prompt, add_special_tokens=False, return_tensors="pt" )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=64,
            do_sample=False,
        )

    generated_ids = output_ids[ 0, input_ids.shape[ -1 ]: ]
    turn1_response = tokenizer.decode( generated_ids )

    print( f"Generated: {turn1_response}" )

    # --- Turn 2: tool result -> final answer ---
    turn2_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{tool_call}<|eot_id|><|start_header_id|>ipython<|end_header_id|>

{tool_result}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    print( "\n=== Turn 2: Final answer ===" )

    input_ids = tokenizer.encode( turn2_prompt, add_special_tokens=False, return_tensors="pt" )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=64,
            do_sample=False,
        )

    generated_ids = output_ids[ 0, input_ids.shape[ -1 ]: ]
    turn2_response = tokenizer.decode( generated_ids )

    print( f"Generated: {turn2_response}" )

if __name__ == "__main__":
    main()