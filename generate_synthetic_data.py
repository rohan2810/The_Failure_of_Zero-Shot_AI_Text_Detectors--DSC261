"""
EUR-Lex synthetic data generation (simple version).

- Generates 2500 samples per model (total = len(MODELS) * SAMPLES_PER_MODEL)
- Uses a 50/50 mix of preserve_case_numbers True/False, shuffled across the dataset
- Saves original samples and synthetic outputs to CSV in OUTPUT_DIR
"""

import os
import random
import re

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Configuration
# -----------------------------
MODELS = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-3-4b-it",
]

SAMPLES_PER_MODEL = 2500
DATASET_NAME = "jonathanli/eurlex"
DATASET_CONFIG = "eurlex57k"

OUTPUT_DIR = "eurlex_synthetic_data"

BATCH_SIZE = 32
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_INPUT_TOKENS = 4096

# Qwen3-specific token id for </think> (defensive; thinking is disabled below anyway).
QWEN3_END_THINK_TOKEN_ID = 151668


def create_prompt(text, preserve_case_numbers=False):
    """
    Create a prompt for generating synthetic legal text.

    Notes:
      - preserve_case_numbers=False: full anonymization, including case numbers
      - preserve_case_numbers=True: two-shot in-context prompt to enforce case-number preservation
    """
    if preserve_case_numbers:
        return f"""Task: Rewrite legal documents while keeping case numbers identical.

Example 1:
Original: "Case T-123/15 was decided on 12 January 2016 in Brussels."
Rewritten: "Case T-123/15 was resolved on 5 March 2018 in Luxembourg."

Example 2:
Original: "In judgment C-456/19, the applicant Smith from Italy appealed."
Rewritten: "In judgment C-456/19, the appellant Johnson from Spain contested."

Your turn:
Original: {text}

Rewritten:"""

    return f"""You are tasked with creating a synthetic version of a legal document text. Rewrite and paraphrase the following legal text while:
1. Changing all dates, years, reference numbers, and case numbers
2. Changing any person names, organization names, or location names (PII)
3. Changing specific details (e.g., diseases, countries, rivers, etc.) while keeping the legal structure
4. Maintaining the formal legal language style
5. Keeping similar length and structure

Original Text: {text}

Provide ONLY the rewritten text, without any explanation or additional text."""


def clean_model_output(text):
    """Remove thinking tags and other unwanted artifacts from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def apply_chat_template(tokenizer, messages, model_name):
    """
    Apply the tokenizer's chat template if available.
    For Qwen3, explicitly disable thinking if supported.
    """
    is_qwen3 = "qwen3" in model_name.lower()
    try:
        if is_qwen3:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError) as e:
        print(f"   [WARNING] apply_chat_template failed for {model_name}: {e}")
        if "system" in str(e).lower():
            return f"User: {messages[1]['content']}\n\nAssistant:"
        return f"{messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"


def generate_synthetic_text_batch(model, tokenizer, prompts, model_name, device):
    """Generate synthetic text for a batch of prompts."""
    is_qwen3 = "qwen3" in model_name.lower()

    all_texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates synthetic legal documents."},
            {"role": "user", "content": prompt},
        ]
        all_texts.append(apply_chat_template(tokenizer, messages, model_name))

    model_inputs = tokenizer(
        all_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=pad_token_id,
        )

    results = []
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
        new_tokens = output_ids[len(input_ids) :].tolist()

        start_idx = 0
        if is_qwen3:
            try:
                last_pos = len(new_tokens) - 1 - new_tokens[::-1].index(QWEN3_END_THINK_TOKEN_ID)
                start_idx = last_pos + 1
            except (ValueError, IndexError):
                start_idx = 0

        text = tokenizer.decode(new_tokens[start_idx:], skip_special_tokens=True).strip("\n")
        results.append(clean_model_output(text))

    return results


def load_model_and_tokenizer(model_name, device):
    """Load a model and tokenizer."""
    print(f"   Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if not torch.cuda.is_available():
        model = model.to(device)

    model.eval()
    return model, tokenizer

def build_preserve_flags(n):
    """Exactly half True/False (off by one if n is odd), shuffled."""
    midpoint = n // 2
    flags = [False] * midpoint + [True] * (n - midpoint)
    random.shuffle(flags)
    return flags


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_size = len(MODELS) * SAMPLES_PER_MODEL

    print("=" * 80)
    print("EUR-Lex Synthetic Data Generation")
    print("=" * 80)
    print(f"Models: {MODELS}")
    print(f"Samples per model: {SAMPLES_PER_MODEL}")
    print(f"Total samples: {sample_size}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\n1. Loading dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    print("\n2. Sampling records...")
    all_data = []
    for split in ["train", "validation", "test"]:
        if split in dataset:
            all_data.extend(dataset[split])

    random.shuffle(all_data)
    sampled_data = all_data[:sample_size]
    if len(sampled_data) != sample_size:
        raise RuntimeError(f"Expected {sample_size} samples but got {len(sampled_data)}.")
    print(f"   Sampled {len(sampled_data)} records (seed={SEED})")

    print("\n3. Saving original samples...")
    original_records = [
        {"celex_id": s["celex_id"], "title": s.get("title", ""), "text": s.get("text", "")}
        for s in sampled_data
    ]
    original_df = pd.DataFrame(original_records)
    original_file = os.path.join(OUTPUT_DIR, f"original_{sample_size}_samples.csv")
    original_df.to_csv(original_file, index=False)
    print(f"   Saved {len(original_df)} rows to {original_file}")

    preserve_case_numbers_flags = build_preserve_flags(sample_size)
    print(f"\n   {sum(preserve_case_numbers_flags)} samples will preserve case numbers")
    print(f"   {sample_size - sum(preserve_case_numbers_flags)} samples will generate new case numbers")

    all_results = []

    for model_idx, model_name in enumerate(MODELS):
        print("\n" + "=" * 80)
        print(f"Processing Model {model_idx + 1}/{len(MODELS)}: {model_name}")
        print("=" * 80)

        start_idx = model_idx * SAMPLES_PER_MODEL
        end_idx = start_idx + SAMPLES_PER_MODEL

        model_samples = sampled_data[start_idx:end_idx]
        model_flags = preserve_case_numbers_flags[start_idx:end_idx]
        if len(model_samples) != SAMPLES_PER_MODEL:
            raise RuntimeError(
                f"{model_name}: expected {SAMPLES_PER_MODEL} samples but got {len(model_samples)}"
            )

        print(f"   Processing samples {start_idx} to {end_idx - 1} ({len(model_samples)} samples)")
        model, tokenizer = load_model_and_tokenizer(model_name, device)

        model_results = []
        num_batches = (len(model_samples) + BATCH_SIZE - 1) // BATCH_SIZE

        with tqdm(total=len(model_samples), desc=f"Processing {model_name}") as pbar:
            for batch_idx in range(num_batches):
                batch_start = batch_idx * BATCH_SIZE
                batch_end = min(batch_start + BATCH_SIZE, len(model_samples))
                batch = model_samples[batch_start:batch_end]

                try:
                    batch_prompts = []
                    for i, sample in enumerate(batch):
                        preserve = model_flags[batch_start + i]
                        batch_prompts.append(
                            create_prompt(sample.get("text", ""), preserve_case_numbers=preserve)
                        )

                    synthetic_texts = generate_synthetic_text_batch(
                        model, tokenizer, batch_prompts, model_name, device
                    )

                    for i, sample in enumerate(batch):
                        preserve = model_flags[batch_start + i]
                        model_results.append(
                            {
                                "celex_id": sample["celex_id"],
                                "preserve_case_numbers": preserve,
                                "model_name": model_name,
                                "synthetic_text": synthetic_texts[i],
                            }
                        )

                    pbar.update(len(batch))

                except Exception as e:
                    print(f"\n   Error processing batch {batch_idx}: {e}")
                    for i, sample in enumerate(batch):
                        preserve = model_flags[batch_start + i]
                        model_results.append(
                            {
                                "celex_id": sample["celex_id"],
                                "preserve_case_numbers": preserve,
                                "model_name": model_name,
                                "synthetic_text": f"ERROR: {str(e)}",
                            }
                        )
                    pbar.update(len(batch))
                    continue
        all_results.extend(model_results)
        print(f"   Completed {model_name}: {len(model_results)} samples processed")

    print("\n4. Saving synthetic results...")
    synthetic_df = pd.DataFrame(all_results)
    synthetic_file = os.path.join(OUTPUT_DIR, f"synthetic_{sample_size}_samples.csv")
    synthetic_df.to_csv(synthetic_file, index=False)
    print(f"   Saved {len(synthetic_df)} rows to {synthetic_file}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total samples processed: {len(all_results)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Models used: {len(MODELS)}")
    for model_name in MODELS:
        model_count = sum(1 for r in all_results if r["model_name"] == model_name)
        print(f"  - {model_name}: {model_count} samples")
    preserve_count = sum(1 for r in all_results if r.get("preserve_case_numbers", False))
    print(f"Samples with preserved case numbers: {preserve_count}")
    print(f"Samples with synthetic case numbers: {len(all_results) - preserve_count}")
    error_count = sum(1 for r in all_results if str(r["synthetic_text"]).startswith("ERROR:"))
    print(f"Errors: {error_count}")
    print(f"Success rate: {((len(all_results) - error_count) / len(all_results) * 100):.1f}%")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - Original samples: {original_file}")
    print(f"  - Synthetic samples: {synthetic_file}")
    print("\n✓ Synthetic data generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


