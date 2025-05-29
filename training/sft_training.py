####### ----------------
## SFT Training 
####### ----------------


import os
import re
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import wandb

# System prompt and task instructions
R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a single-choice Multiple Choice question, and the Assistant solves it. Please answer the multiple choice question by selecting only one from option a., option b., option c., option d., option e..
The assistant first thinks about the explanation process in the mind and then provides the user
with the answer. The explanation process and answer are enclosed within <explanation> </explanation> and
<answer> </answer> tags, respectively, i.e., <explanation> explanation process here </explanation>
<answer> answer here </answer>."""

TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single letter from a, b, c, d, e."

EXAMPLE = "Question context: PersonA, a novice in the field of selecting mouse ES cells for screening, raises several detailed questions regarding colony selection and verification processes. PersonB provides thorough answers focused on optimizing the selection and genetic verification processes in embryonic stem cells. Question: Would anyone recommendation colony picking versus 96-well dilution for screening colonies? Please choose one of the following options: a. \"Colony picking is preferred for mouse ES cells because they grow as perfect colonies that are easy to pick\" b. \"96-well dilution is more efficient because it allows for high-throughput screening of colonies\" c. \"Colony picking is more labor-intensive and should be avoided when possible\" d. \"96-well dilution is the best method for ensuring genetic consistency across colonies\" e. \"Both methods are equally effective, so the choice depends on available resources\""+"<explanation>For mouse ES cells there's no reason to do limited dilution, as they grow as perfect colonies that are easy to pick.</explanation>\n<answer>a</answer>"
           
def extract_hash_answer(text: str) -> str:
    try:
        explanation, answer = text.split("####", 1)
        return f"<explanation> {explanation.strip()} </explanation> <answer> {answer.strip()} </answer>"
    except ValueError:
        return ""

def preprocess_dataset(dataset_name, split="train", chunk_size=1000) -> Dataset:
    """
    Load the dataset from disk and process each batch to generate chat-style prompts.
    The resulting dataset will have a "text" field (a string) and an "answer" field.
    """
    dataset = load_from_disk(dataset_name)[split]

    def process_batch(batch):
        chats = [
            f"System: {R1_STYLE_SYSTEM_PROMPT}\n{TASK_SPECIFIC_INSTRUCTIONS}\n{EXAMPLE}\nUser: {q.strip()}"
            for q in batch['question']
        ]
        return {
            'text': chats,  # Ensure it's a list of strings, not a list of dictionaries
            'answer': [extract_hash_answer(a) for a in batch['answer']]
        }

    return dataset.map(process_batch, batched=True, batch_size=chunk_size)

def main():
    # Load and preprocess the dataset
    dataset_name = './dataset/Genome-Bench'
    dataset = preprocess_dataset(dataset_name, chunk_size=500)

    learning_rate = 1e-5

    epoch = 2

    # Define model and output paths
    model_name = "Qwen2.5-7B-Instruct" 
    output_dir = f"{model_name.split('/')[-1]}-SFT"
    run_name = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"

    # Set memory-related environment variable
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Create SFT training configuration
    training_args = SFTConfig(
        learning_rate = learning_rate,
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        num_train_epochs=epoch,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=output_dir,
        run_name=run_name,
        log_on_each_node=False,
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load the tokenizer and set pad token
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=512,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the SFT trainer using the tokenizer as the processing class
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Initialize wandb in offline mode for experiment tracking
    wandb.init(project="crispr_sft", name=run_name, mode="offline")
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()


