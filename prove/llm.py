import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

recipes = [
    "pasta al forno", "pasta al pesto", "pasta al pomodoro", 
    "insalata di pollo", "insalata di tonno", "insalata di riso", "insalata di farro"
]

user_content = "From the following recipes select the more sustainable and healthy recipe and explain why."
user_content = "\n".join([f"{i+1}. Recipe:{recipe}" for i, recipe in enumerate(recipes)])

prompt = [
  {"role": "system", "content": "You are a bot that has the task to advocate per healthy and sustainable diates. You will be given recipes to suggest to the user, select the most sustainable and healthy recipe and explain why."},
  {"role": "user", "content": user_content},
]

tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()

model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  quantization_config=nf4_config,
  device_map="auto",
)

outputs = model.generate(inputs, do_sample=True, max_new_tokens=256)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))