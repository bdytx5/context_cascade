"""
FULL TRAINING SCRIPT: C3 with 100 latent tokens
Actually trains the model, not just setup bullshit.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

# Add parent path for imports
sys.path.insert(0, '/home/cloud/c3/C3-Context-Cascade-Compression/C3-master')

from C3.model.C3 import C3QwenForCausalLM, C3Config
from C3.utils.constants import IGNORE_INDEX
from C3.utils.conversation import conv_templates, SeparatorStyle

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


# ============ EXPAND LATENT TOKENS ============

def expand_latent_tokens(model, new_latent_len=100):
    """Expand Q embedding from 32 -> new_latent_len"""
    old_Q = model.model.Q
    old_latent_len = old_Q.num_embeddings
    hidden_size = old_Q.embedding_dim

    print(f"Expanding Q: {old_latent_len} -> {new_latent_len} tokens")

    new_Q = nn.Embedding(new_latent_len, hidden_size)

    with torch.no_grad():
        new_Q.weight[:old_latent_len] = old_Q.weight.clone()
        mean, std = old_Q.weight.mean(), old_Q.weight.std()
        nn.init.normal_(new_Q.weight[old_latent_len:], mean=mean.item(), std=std.item())

    model.model.Q = new_Q
    model.config.latent_token_len = new_latent_len
    model.model.config.latent_token_len = new_latent_len

    return model


# ============ DATA ============

DATA_PATH = "/home/cloud/c3/C3-Context-Cascade-Compression/dataset/train_test_data.json"


class C3Dataset(Dataset):
    """Simple dataset for C3 training"""

    def __init__(self, data, tokenizer, latent_token_len, max_length=4096):
        self.data = data
        self.tokenizer = tokenizer
        self.latent_token_len = latent_token_len
        self.max_length = max_length

        self.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        self.im_start_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN])[0]
        self.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_END_TOKEN])[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        convs = item["conversations"]

        # Extract context and conversation
        context = ""
        human_msg = ""
        gpt_msg = ""

        for turn in convs:
            if turn["from"] == "context":
                context = turn["value"]
            elif turn["from"] == "human":
                human_msg = turn["value"]
            elif turn["from"] == "gpt":
                gpt_msg = turn["value"]

        # Build context with latent token placeholder
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.latent_token_len + DEFAULT_IM_END_TOKEN
        context_with_placeholder = context + replace_token

        # Build prompt with latent token placeholder
        human_msg_processed = human_msg.replace("<image>", replace_token)

        # Use MPT conversation format
        conv = conv_templates["mpt"].copy()
        conv.append_message(conv.roles[0], human_msg_processed)
        conv.append_message(conv.roles[1], gpt_msg)
        prompt = conv.get_prompt()

        # Tokenize
        input_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = input_encoding.input_ids[0]

        context_encoding = self.tokenizer(
            context_with_placeholder,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        context_ids = context_encoding.input_ids[0]

        # Create labels (mask the prompt part, only predict gpt response)
        labels = input_ids.clone()

        # Find where assistant response starts and mask everything before
        sep = conv.sep + conv.roles[1]
        prompt_part = prompt.split(sep)[0] + sep
        prompt_len = len(self.tokenizer(prompt_part).input_ids)
        labels[:prompt_len] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "context_ids": context_ids,
        }


def collate_fn(batch, tokenizer):
    """Collate function for DataLoader"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    context_ids = [item["context_ids"] for item in batch]

    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    context_ids = nn.utils.rnn.pad_sequence(context_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "context_ids": context_ids,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
        "context_attention_mask": context_ids.ne(tokenizer.pad_token_id),
    }


# ============ TRAINING ============

def train():
    # Config
    NEW_LATENT_LEN = 100
    MODEL_NAME = 'liufanfanlff/C3-Context-Cascade-Compression'
    OUTPUT_DIR = './c3_100_latent_trained'
    BATCH_SIZE = 1
    GRAD_ACCUM = 8
    EPOCHS = 3
    LR = 2e-5

    print("=" * 80)
    print(f"C3 TRAINING - {NEW_LATENT_LEN} LATENT TOKENS - FULL FINETUNE")
    print("=" * 80)

    # 1. Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load model
    print("\n[2/6] Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_safetensors=True,
    )

    # 3. Expand latent tokens
    print("\n[3/6] Expanding latent tokens...")
    model = expand_latent_tokens(model, NEW_LATENT_LEN)

    # Initialize special tokens
    model.initialize_special_tokenizer(tokenizer)

    # FULL FINETUNE - no freezing
    model.requires_grad_(True)
    if model.model.llm1 is not None:
        for p in model.model.llm1.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 4. Get training data
    print("\n[4/6] Loading training data...")
    with open(DATA_PATH, "r") as f:
        train_data = json.load(f)
    print(f"   Loaded {len(train_data)} training samples from {DATA_PATH}")

    dataset = C3Dataset(train_data, tokenizer, NEW_LATENT_LEN, max_length=4096)

    # 5. Training setup
    print("\n[5/6] Setting up training...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer),
    )

    # 6. TRAIN
    print("\n[6/6] TRAINING...")
    print("=" * 80)

    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    train()
