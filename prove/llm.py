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

user_content = "Choose the most healthy and sustainable recipe from the list below and explain why it is the best in a user friendly paragraph, without comparing it to the others."
user_content += "\n".join([f"Recipe: {recipe}" for i, recipe in enumerate(recipes)])

prompt = [
  {"role": "system", "content": "You are an AI assistant that helps users make informed choices about healthy and sustainable diets. Choose the most healthy and sustainable recipe and explain your selection in a user friendly paragraph, without making comparisons with other recipes."},
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
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

response_start = response.find("assistant\n\n") + len("assistant\n\n")
clean_response = response[response_start:]

print(clean_response)