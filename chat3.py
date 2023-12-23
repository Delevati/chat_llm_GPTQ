import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

def generate_response(prompt, model, tokenizer):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.8, top_k=20, max_new_tokens=1024)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def print_message(timestamp, sender, content):
    print(f"[{timestamp}] {sender}: {content}")

def save_conversation_cache(cache, filename="conversation_cache.json"):
    with open(filename, "w") as file:
        json.dump(cache, file)

def load_conversation_cache(filename="conversation_cache.json"):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
        print(f"Error loading conversation cache: {e}")
        return []

def main():
    # Configurar PyTorch para GPU, se disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_or_path = "TheBloke/WizardLM-13B-V1.1-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device, trust_remote_code=True, revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    print_message(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Assistant", "Hello! Type 'exit' to end the conversation.")
    
    conversation_cache = load_conversation_cache()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print_message(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Assistant", "Exiting the conversation.")
            save_conversation_cache(conversation_cache)
            break

        # Adicione um prefixo único ao prompt do usuário
        prompt_template = 'USER:' + user_input + '\nASSISTANT:'
        response = generate_response(prompt_template, model, tokenizer)

        # Atualize o cache da conversa
        conversation_cache.append(("USER", user_input))
        conversation_cache.append(("ASSISTANT", response))

        # Imprima a resposta mantendo o histórico organizado
        print_message(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "ASSISTANT", response)

if __name__ == "__main__":
    main()
