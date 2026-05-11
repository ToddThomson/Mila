import requests
import json

def main():
    url = "http://localhost:11434/api/chat"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather conditions for a specified location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": { "type": "string", "description": "City and country, e.g. London, UK" },
                        "units": { "type": "string", "description": "celsius or fahrenheit. Default: celsius." }
                    },
                    "required": [ "location" ]
                }
            }
        }
    ]

    messages = [
        { "role": "user", "content": "What is the temperature in Vancouver today?" }
    ]

    payload = {
        "model": "llama3.2:3b",
        "messages": messages,
        "tools": tools,
        "stream": False
    }

    response = requests.post( url, json=payload )
    result = response.json()

    print( "=== Turn 1 ===" )
    print( json.dumps( result[ "message" ], indent=2 ) )

    # Simulate tool result and send Turn 2
    tool_call = result[ "message" ].get( "tool_calls", [] )

    if tool_call:
        messages.append( result[ "message" ] )
        messages.append( {
            "role": "tool",
            "content": "The current weather in Vancouver, Canada is 22C and sunny."
        } )

        payload[ "messages" ] = messages

        response2 = requests.post( url, json=payload )
        result2 = response2.json()

        print( "\n=== Turn 2 ===" )
        print( json.dumps( result2[ "message" ], indent=2 ) )
    else:
        print( "No tool call detected in Turn 1" )

if __name__ == "__main__":
    main()