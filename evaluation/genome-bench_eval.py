####-----------------------
#### Formal evaluation
####-----------------------

import json
import transformers
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import openai


#### Load data
dataset_name = './dataset/Genome-Bench'
dataset_loaded = load_from_disk(dataset_name)['test']
# Convert to the desired list format with the name 'questions'
questions = [{"question": q, "answer": a} for q, a in zip(dataset_loaded["question"], dataset_loaded["answer"])]


# System Prompt
R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a single-choice Multiple Choice question, and the Assistant solves it. Please answer the multiple choice question by selecting only one from optiona a., option b., option c., option d., option e..
The assistant first thinks about the explanation process in the mind and then provides the user
with the answer. The explanation process and answer are enclosed within <explanation> </explanation> and
<answer> </answer> tags, respectively, i.e., <explanation> explanation process here </explanation>
<answer> answer here </answer>."""


TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single letter from a,b,c,d,e."

def extract_xml_answer(text: str) -> str:
    """Extracts the answer from a response using the <answer> tag."""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""


# Initialize the pipeline with Qwen2.5-7B-Instruct
name = 'Qwen2.5-7B-Instruct' 
model_id = name  # Replace with your model ID


llm = LLM(model=model_id, dtype="half", max_model_len=1024, device="cuda:0")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=1024, padding_side='right')

import random
random_number = random.randint(1, 10000)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024,
    stop_token_ids=[tokenizer.eos_token_id],
    seed = random_number,
)


BATCH_SIZE = 8
# Evaluate questions in batches
results = []
correct = 0
total = 0

# Progress bar
progress_bar = tqdm(total=len(questions), desc="Processing", unit="examples", dynamic_ncols=True)

for i in range(0, len(questions), BATCH_SIZE):
    batch_data = questions[i:i + BATCH_SIZE]
    
    # Prepare prompts using few-shot learning
    prompts = [
        [
            {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
            {"role": "user", "content": q["question"]},
        ]

        for q in batch_data
    ]

    # Convert prompts to formatted strings
    formatted_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    # Generate responses using vLLM
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Process responses
    for j, output in enumerate(outputs):
        response = output.outputs[0].text
        generated_answer = extract_xml_answer(response)
        true_answer = extract_xml_answer(batch_data[j]["answer"])

        # Store the result
        result = {
            "Question": batch_data[j]["question"],
            "Generated Answer": generated_answer,
            "Correct Answer": true_answer,
            "Full Response": response,
            "Correct": generated_answer == true_answer
        }
        results.append(result)

        if generated_answer == true_answer:
            correct += 1
        total += 1

    # Update progress bar
    progress_bar.update(len(batch_data))
    progress_bar.set_postfix({
        "Accuracy": f"{(correct / total) * 100:.2f}%",
        "Correct": f"{correct}/{total}",
    })

progress_bar.close()


# Save the results to a JSON file
output_file = name + '_evaluation_results.json'
with open(output_file, 'w') as file:
    json.dump(results, file, indent=4)

print(f"Evaluation complete. Results saved to {output_file}.")
