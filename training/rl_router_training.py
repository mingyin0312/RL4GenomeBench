####### --------------------------
## GRPO Training for Router Model
####### --------------------------

import random
import os
import re
import warnings
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer import GRPOConfig, GRPOTrainer
from accelerate.utils import gather
import wandb
import warnings
import torch
from typing import Any, Union
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.trainer import GRPOConfig, GRPOTrainer




def extract_xml_answer(text: str) -> str:
    """Extracts the answer from between <answer> ... </answer> tags."""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""

##########################################################################
# New Trainer Subclass: PreGeneratedGRPOTrainer
##########################################################################
class PreGeneratedGRPOTrainer(GRPOTrainer):
    """
    A custom GRPOTrainer that uses pre‐generated completions instead of on‐the‐fly generation.
    
    Each training sample is expected to have a "pre_generated" key containing a list of 4 dictionaries.
    Each dictionary must include:
      - "content": the candidate output from the router assistant (e.g. "<choice>2</choice>")
      - Optionally, a detailed response under a key like "response by model 2"
    
    For training, we use all 4 candidates. We replicate the prompt for each candidate so that the effective
    batch size is (original batch size × num_generations). The ground‐truth answer is repeated accordingly,
    and later the rewards are computed and grouped per sample.
    """
    
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        #prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]


        pre_generated_list = [x["pre_generated"] for x in inputs]  # length: N; each element is list of 4 dicts
        candidate_texts = []  # will have length: N * 4
        reward_texts = []     # will have length: N * 4
        
        for pg in pre_generated_list:
            entry = random.choice(pg)  # randomly sample one entry from pg
            candidate = entry["content"].strip()
            candidate_texts.append(candidate)

            digit_list = re.findall(r"\d", candidate)
            if not digit_list:
                raise ValueError(f"Candidate text '{candidate}' does not contain a digit.")
            digit = digit_list[0]
            response_key = f"response by model {digit}"
            reward_texts.append(entry.get(response_key, candidate))

        completion_inputs = self.processing_class(
            text=candidate_texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
        )
        completion_ids = completion_inputs["input_ids"].to(device)      # This is analogous to completion_ids produced by generation.
        completion_mask = completion_inputs["attention_mask"].to(device)  # Similarly, candidate_mask.


        # Construct prompt_completion_ids from pre-generated completions
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = (completion_ids == self.processing_class.eos_token_id).to(device)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)

        # Make sure the mask and argmax are on the same device
        any_eos = is_eos.any(dim=1)
        argmax_eos = is_eos.int().argmax(dim=1)

        eos_idx[any_eos] = argmax_eos[any_eos]  # no device mismatch here
        sequence_indices = torch.arange(is_eos.size(1), device=is_eos.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )


        # Decode the generated completions
        completions_text = reward_texts
        completions = completions_text
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion","pre_generated"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)


        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    


##########################################################################
# Reward Function: Correctness
##########################################################################
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Uses the detailed responses (reward_texts) to compute correctness.
    It extracts the answer (e.g. the letter "a") from the detailed response and compares it with the ground truth.
    Returns 1.0 if they match, otherwise -1.0.
    """
    reward_texts = kwargs.get("reward_texts", None)
    if reward_texts is not None:
        extracted_answers = [extract_xml_answer(rt) for rt in reward_texts]
    else:
        extracted_answers = [extract_xml_answer(c) for c in completions]

    print("\n\n===============================================================\n"
          f"User question (sample): {prompts[0]}\n"
          f"Ground truth answer: {answer[0]}\n"
          f"Extracted answers (from reward texts): {extracted_answers}\n")
    return [1.0 if r == a else -1.0 for r, a in zip(extracted_answers, answer)]

##########################################################################
# Dataset Preprocessing
##########################################################################
R1_STYLE_SYSTEM_PROMPT = """There are four models capable of answering single-choice multiple choice questions. You are the Router Assistant.

In a conversation between the User and the Router Assistant, the User provides a single-choice multiple choice question. Your job as the Router Assistant is to suggest which of the four models is most likely to provide the best answer.

You must select only one model from the following options: model 1, model 2, model 3, or model 4.

Provide your recommendation enclosed within <choice> </choice> tags, for example: <choice>1</choice>
"""
TASK_SPECIFIC_INSTRUCTIONS = "The choice must be a single digit: 1, 2, 3, or 4."

def preprocess_dataset(dataset_name, split="train", chunk_size=1000) -> Dataset:
    dataset = load_from_disk(dataset_name)[split]
    def process_batch(batch):
        # Build the prompt as a list of two messages:
        # System instruction (with task-specific instructions) and the user question.
        prompts = [
            [
                {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
                {'role': 'user', 'content': q.strip()}
            ]
            for q in batch['question']
        ]
        return {
            'prompt': prompts,
            'answer': [extract_xml_answer(a) for a in batch['answer']],
            'pre_generated': batch['pre_generated']
        }
    return dataset.map(process_batch, batched=True, batch_size=chunk_size)

##########################################################################
# Main Training 
##########################################################################
def main():
    dataset_name = './dataset/Genome-Bench-Router' # Genome-Bench-Router dataset contains pre-generated responses from different RL models
    dataset = preprocess_dataset(dataset_name, chunk_size=500)

    model_name = "Qwen2.5-7B-Instruct"
    epoch = 2
    learning_rate = 1e-5
    output_dir = f"{model_name.split('/')[-1]}-Router"
    run_name = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    training_args = GRPOConfig(
        learning_rate=learning_rate,
        beta=0.005,
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=8,
        num_generations=8,  # Use 4 candidate generations per sample
        gradient_accumulation_steps=4,
        max_prompt_length=256,
        max_completion_length=512,
        num_train_epochs=epoch,
        save_steps=100000,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=output_dir,
        run_name=run_name,
        log_on_each_node=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=training_args.max_completion_length,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Instantiate the custom trainer.
    trainer = PreGeneratedGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    wandb.init(project="crispr_grpo_router", name=run_name, mode="offline")
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()