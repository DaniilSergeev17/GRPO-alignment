import re
from typing import Union, List
import torch
from tqdm import tqdm
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

PatchFastRL("GRPO", FastLanguageModel)

# Constants
lora_rank = 64

system_prompt = """
  Respond in the following format:
  <reasoning>
  ...
  </reasoning>
  <answer>
  ...
  </answer>
"""

sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=4096
)

# Model Initialization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    dtype=torch.float16,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    random_state=42,
)


# Dataset Preparing for GRPO
def extract_answer(answer: str) -> Union[str, None]:
    """Get only the answer after '####' tokens (in gsm8k)"""
    return answer.split('####')[-1].strip().replace(',', '') if '####' in answer else None


def extract_reasoning(answer):
    """Get only the reasoning before '####' tokens (in gsm8k)"""
    return answer.split('####')[0].strip() if '####' in answer else None


def preproc(example):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': example['question']}
    ]
    clear_answer = extract_answer(example['answer'])
    reasoning = extract_reasoning(example['answer'])

    return {
        'prompt': messages,
        'answer': clear_answer,
        'target_reasoning': reasoning
    }


dataset = load_dataset('openai/gsm8k', 'main', trust_remote_code=True)
dataset = dataset.map(preproc)


# Auxiliary functions for using inside reward functions
def get_reasoning(content: str) -> Union[str, None]:
    """Get only the reasoning between <reasoning>REASONING</reasoning> tokens"""
    if '<reasoning>' in content and '</reasoning>' in content:
        return content.split('<reasoning>')[1].split('</reasoning>')[0].strip()
    return None


def isnumber(answer: str) -> bool:
    """Check if the string is a number (any: float, int, scientific, etc.)"""
    try:
        float(answer)
        return True
    except ValueError:
        return False


def get_answer(content: str) -> Union[str, None]:
    """Get only the answer between <answer>ANSWER</answer> tokens"""
    if '<answer>' in content and '</answer>' in content:
        return content.split('<answer>')[1].split('</answer>')[0].strip()
    return None


def calc_structure_reward(content: str) -> float:
    """Check the structure correctness of model's answer"""
    reward = 0.0
    if '<reasoning>' in content and '</reasoning>' in content:
        reward += 0.5
    if '<answer>' in content and '</answer>' in content:
        reward += 0.5
    return reward


# Rule-based rewards for GRPO
def target_length_reward_func(completions, target_reasoning, **kwargs) -> List[float]:
    """Give reward for the same reasoning len of model's answer as in target"""
    rewards = []
    contents = [completion[0]['content'] for completion in completions]

    for generated_content, target in zip(contents, target_reasoning):
        generated_reasoning = get_reasoning(generated_content)
        if generated_reasoning is None:
            rewards.append(0.0)
            continue

        generated_len = len(tokenizer(generated_reasoning)['input_ids'])
        target_len = len(tokenizer(target)['input_ids'])

        if target_len == 0:
            rewards.append(1.0 if generated_len == 0 else 0.0)
            continue

        reward = max(0.0, 1.0 - abs(generated_len - target_len) / target_len)
        rewards.append(reward)

    return rewards


def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """Give reward for structure correctness of model's answer"""
    completion_contents = [completion[0]['content'] for completion in completions]
    return [calc_structure_reward(content) for content in completion_contents]


def isnumber_reward_func(completions, **kwargs) -> List[float]:
    """Give reward if the model's answer is a number"""
    completion_contents = [completion[0]['content'] for completion in completions]
    completion_answers = [get_answer(completion) for completion in completion_contents]
    return [0.25 if ans is not None and isnumber(ans) else 0.0 for ans in completion_answers]


def exact_match_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """Give reward for exact match with target"""
    completion_contents = [completion[0]['content'] for completion in completions]
    completion_answers = [get_answer(completion) for completion in completion_contents]
    return [1.0 if pred == target else 0.0 for pred, target in zip(completion_answers, answer)]


def strict_format_reward_func(completions, **kwargs) -> List[float]:
    """Give reward for strict following the response format"""
    pattern = r'^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$'
    completion_contents = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, content, re.S) for content in
               completion_contents]  # re.S for allowing '\n' in answer as in prompt
    return [1.0 if match else 0.0 for match in matches]


# L1: LCPO rewards
def lcpo_exact_length_reward_func(completions, target_reasoning, **kwargs) -> List[float]:
    """LCPO exact length reward: reward = -alpha * abs(target_len - generated_len)"""
    rewards = []
    for completion, target in zip(completions, target_reasoning):
        content = completion[0]['content']
        generated_reasoning = get_reasoning(content)
        if generated_reasoning is None or target is None:
            rewards.append(0.0)
            continue

        generated_len = len(tokenizer(generated_reasoning)['input_ids'])
        target_len = len(tokenizer(target)['input_ids'])
        rewards.append(-0.03 * abs(target_len - generated_len))

    return rewards


# reward grows if the model does not exceed the target_len and trying to answer more concise
def lcpo_max_length_reward_func(completions, target_reasoning, **kwargs) -> List[float]:
    """LCPO max: reward = clip(alpha * (target_len - generated_len) + beta, 0, 1) """
    rewards = []
    for completion, target in zip(completions, target_reasoning):
        content = completion[0]['content']
        generated_reasoning = get_reasoning(content)
        if generated_reasoning is None or target is None:
            rewards.append(0.0)
            continue

        generated_len = len(tokenizer(generated_reasoning)['input_ids'])
        target_len = len(tokenizer(target)['input_ids'])

        reward = 0.03 * (target_len - generated_len) + 0.5
        rewards.append(max(0.0, min(1.0, reward)))

    return rewards


# Training params
training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    max_grad_norm=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=10,
    fp16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_generations=8,
    max_prompt_length=512,
    max_completion_length=512,
    num_train_epochs=1,
    save_steps=934,  # 1 epoch
    report_to='tensorboard',
    output_dir='outputs',
)

# you can place any rule-based reward function inside reward_funcs=[]
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        strict_format_reward_func,
        isnumber_reward_func,
        soft_format_reward_func,
        exact_match_reward_func,
    ],
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args=training_args,
    processing_class=tokenizer
)


# Auxiliary functions for evaluation
def check_format(text: str) -> bool:
    """Check the correctness of the model's answer format for calculating format_accuracy"""
    pattern = r'^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$'
    return bool(re.match(pattern, text, re.S))


def extract_num_from_answer(text: str) -> Union[float, None]:
    """Get last number between <answer>ANSWER</answer> tokens for calculating accuracy"""
    answer = get_answer(text)
    if answer:
        all_digits = re.findall(r'-?\d+(?:\.\d+)?', answer.replace(',', ''))
        if len(all_digits) > 0:
            return float(all_digits[-1])
        return None
    return None


# Evaluation function
def evaluate_model(model, tokenizer, dataset, lora_request=None):
    """
    Args:
        lora_request - controls if it is a baseline model (=None) or GRPO-trained
    """
    ans_correct = 0
    format_correct = 0
    for sample in tqdm(dataset):
        question = sample['question']
        target = float(sample['answer'])

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = model.fast_generate(
            [text], sampling_params=sampling_params, lora_request=lora_request
        )[0].outputs[0].text

        pred = extract_num_from_answer(out)
        if pred == target: ans_correct += 1
        if check_format(out): format_correct += 1

    accuracy = ans_correct / len(dataset)
    format_accuracy = format_correct / len(dataset)

    model_format = 'after GRPO' if lora_request is not None else 'baseline'
    print(f'{model_format}: Accuracy -> {accuracy}, Right format percent -> {format_accuracy}')


if __name__ == '__main__':
    # Uncomment if you want to calculate baseline accuracy and format_accuracy
    # evaluate_model(model, tokenizer, dataset['test'])

    trainer.train()
    model.save_lora('lora_after_grpo')
    # evaluate GRPO-trained model
    evaluate_model(model, tokenizer, dataset['test'], lora_request=model.load_lora('lora_after_grpo'))