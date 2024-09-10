from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, AutoTokenizer

# model_name = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default


def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()
    outputs = model.generate(inputs, attention_mask=attention_mask, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    chat_dialogue = "User: Hello!; Chatbot: Hi! How can I help you today?; User: Can you tell me a joke?; Chatbot: Sure! Why did the scarecrow win an award? Because he was outstanding in his field!; User: That's funny!; Chatbot: I'm glad you liked it! Is there anything else I can do for you?; User: No, that's all. Thanks!; Chatbot: "
    response = generate_response(chat_dialogue)
    print(f"Chatbot: {response}")
    chat_dialogue += response
    while True:
        user_input = input("You: ")
        chat_dialogue += user_input + "; Chatbot: "
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = generate_response(user_input)
        chat_dialogue += response
        print(response)