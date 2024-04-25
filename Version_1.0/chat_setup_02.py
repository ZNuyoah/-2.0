from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 加载模型和分词器
    model_path = "/home/zhaoyibo/桌面/LLaMA-Factory-main/Model/Version_1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 设置对话历史和停止条件
    conversation_history = []
    stop_condition = "quit"

    while True:
        # 用户输入
        user_input = input("User: ")

        # 添加用户输入到对话历史
        conversation_history.append({"role": "user", "content": user_input})

        # 如果用户输入停止条件，则退出循环
        if user_input.strip().lower() == stop_condition:
            print("Conversation ended.")
            break

        # 将对话历史转换为模型输入的文本格式
        text = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

        # 将文本转换为模型输入的张量，并移动到设备上
        model_inputs = tokenizer([text], return_tensors="pt")

        # 生成回复
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )

        # 解码生成的 token，生成回复
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 输出模型回复
        print("Model:", response)
        
        # 添加模型回复到对话历史
        conversation_history.append({"role": "system", "content": response})

if __name__ == "__main__":
    main()