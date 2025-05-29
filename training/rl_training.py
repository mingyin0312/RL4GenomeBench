####### ----------------
## GRPO Training 
####### ----------------


import os
import re
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer import GRPOConfig, GRPOTrainer
import wandb


# ==========================  SYSTEM PROMPT  ==========================
R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a single-choice Multiple Choice question, and the Assistant solves it. Please answer the multiple choice question by selecting only one from optiona a., option b., option c., option d., option e..
The assistant first thinks about the explanation process in the mind and then provides the user
with the answer. The explanation process and answer are enclosed within <explanation> </explanation> and
<answer> </answer> tags, respectively, i.e., <explanation> explanation process here </explanation>
<answer> answer here </answer>."""

TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single letter from a,b,c,d,e."


# =====================================================================
#                       Utility functions
# =====================================================================
def extract_xml_answer(text: str) -> str:
    try:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    except IndexError:
        return ""


def preprocess_dataset(dataset_name, split="train", chunk_size=1000) -> Dataset:
    dataset = load_from_disk(dataset_name)[split]

    def process_batch(batch):
        prompts = [
            [
                {
                    "role": "system",
                    "content": R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS,
                },
                {"role": "user", "content": q.strip()},
            ]
            for q in batch["question"]
        ]

        return {
            "prompt": prompts,
            "answer": [extract_xml_answer(a) for a in batch["answer"]],
        }

    return dataset.map(process_batch, batched=True, batch_size=chunk_size)


# ------------------------- Reward functions --------------------------
def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward 1 pt if completion matches required XML template."""
    pattern = r"^<explanation>(?:(?!</explanation>).)*</explanation>\n<answer>(?:(?!</answer>).)*</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    return [1.0 if re.match(pattern, r) else 0.0 for r in responses]


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward 2 pt if extracted answer matches ground-truth letter."""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]

    print(
        f"\n\n==================== DEBUG ====================\n"
        f"User Question:\n{prompts[0][-1]['content']}"
        f"\n\nCorrect Answer:\n{answer[0]}\n"
        f"\nFirst generated response:\n{responses[0]}"
        f"\nExtracted: {extracted[0]}"
        f"\nCorrectness flags: {''.join('Y' if r==a else 'N' for r,a in zip(extracted,answer))}"
    )

    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]


# =====================================================================
#                              MAIN
# =====================================================================
def main():
    dataset_name = "./dataset/Genome-Bench"
    dataset = preprocess_dataset(dataset_name, chunk_size=500)

    model_name = "Qwen2.5-7B-Instruct"
    epoch = 2
    learning_rate = 1e-5
    num_generations = 4

    output_dir = f"./../{model_name.split('/')[-1]}-RL"
    run_name = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"

    # --- memory env ---
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # ------------------ TRAINING ARGS ------------------
    training_args = GRPOConfig(
        learning_rate=learning_rate,
        beta=0.005,
        optim="adamw_torch",        
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=8,
        num_generations=num_generations,
        gradient_accumulation_steps=4,
        max_prompt_length=256,
        max_completion_length=512,
        num_train_epochs=epoch,
        save_steps=100_000,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=output_dir,
        run_name=run_name,
        log_on_each_node=False,
    )

    # ------------------  Model / Tokenizer --------------
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="balanced"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=training_args.max_completion_length
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------  Trainer ------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    wandb.init(project="crispr_grpo", name=run_name, mode="offline")
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()

