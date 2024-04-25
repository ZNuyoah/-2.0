import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the local language model
model_path = "/home/zhaoyibo/桌面/LLaMA-Factory-main/Model/Version_1.0"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define the device to load the model onto
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define a function to generate response
def generate_response(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [output_ids[len(model_inputs.input_ids):] for output_ids in generated_ids]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Main loop for chatting
while True:
    prompt = input("You: ")
    if prompt.lower() == "exit":
        print("Goodbye!")
        break
    response = generate_response(prompt)
    print("Bot:", response)