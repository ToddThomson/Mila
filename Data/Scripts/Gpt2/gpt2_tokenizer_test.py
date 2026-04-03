from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "You are a helpful AI Assistant. Your name is Mila"
tokens = tokenizer.encode(text)
print(f"Token count: {len(tokens)}")
print(f"Token IDs: {tokens}")
print(f"\nDecoded back: '{tokenizer.decode(tokens)}'")
