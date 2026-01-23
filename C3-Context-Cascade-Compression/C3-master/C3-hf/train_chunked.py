"""
CHUNKED TRAINING: Train LLM2 to handle 160 latent tokens (5 chunks × 32)

Following the original C3 training code patterns.
"""

import os
import sys
import copy
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from dataclasses import dataclass

sys.path.insert(0, '/home/cloud/c3/C3-Context-Cascade-Compression/C3-master')

from C3.utils.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from C3.utils import conversation as conversation_lib

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

# ============ CONFIG ============

NUM_CHUNKS = 5
TOKENS_PER_CHUNK = 32
TOTAL_LATENT_TOKENS = NUM_CHUNKS * TOKENS_PER_CHUNK  # 160
CHUNK_CHAR_SIZE = 6000
DATA_PATH = "/home/cloud/c3/C3-Context-Cascade-Compression/dataset/train_test_data.json"


# ============ CHUNKED MODEL ============

class C3ChunkedModel(nn.Module):
    """
    Wrapper for chunked context encoding.
    Freezes LLM1, trains LLM2 on concatenated latents.
    """

    def __init__(self, base_model, tokenizer, num_chunks=5):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.num_chunks = num_chunks
        self.tokens_per_chunk = base_model.config.latent_token_len  # 32
        self.total_latent_tokens = num_chunks * self.tokens_per_chunk
        self.hidden_size = base_model.model.llm1.config.hidden_size  # 1536

        # Freeze everything first
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Unfreeze LLM2 decoder + mm_projector + lm_head
        for name, p in self.base_model.named_parameters():
            if 'llm1' in name:
                continue  # Keep frozen
            if name == 'model.Q.weight':
                continue  # Keep frozen
            p.requires_grad = True

    def encode_chunk(self, chunk_context_ids, chunk_attention_mask):
        """Encode a single chunk with LLM1. Returns zeros for empty chunks."""
        batch_size = chunk_context_ids.shape[0]
        device = chunk_context_ids.device
        dtype = next(self.base_model.model.llm1.parameters()).dtype

        # Zero latents as fallback
        zero_latents = torch.zeros(batch_size, self.tokens_per_chunk, self.hidden_size,
                                   device=device, dtype=dtype)

        with torch.no_grad():
            context_embeds = self.base_model.model.llm1.model.embed_tokens(chunk_context_ids)
            im_start_token = self.base_model.config.im_start_token

            batch_latents = []
            for i in range(batch_size):
                cur_ids = chunk_context_ids[i]
                cur_embeds = context_embeds[i]
                cur_mask = chunk_attention_mask[i]

                # Check for content
                if cur_mask.sum().item() < 10:
                    batch_latents.append(zero_latents[0])
                    continue

                # Find <img> position
                start_pos = (cur_ids == im_start_token).nonzero(as_tuple=True)[0]
                if len(start_pos) == 0:
                    batch_latents.append(zero_latents[0])
                    continue
                start_pos = start_pos[0].item()

                # Insert Q tokens
                Q = self.base_model.model.Q.weight.to(device=device, dtype=dtype)
                new_embeds = torch.cat([
                    cur_embeds[:start_pos + 1],
                    Q,
                    cur_embeds[start_pos + self.tokens_per_chunk + 1:]
                ], dim=0)

                # Run LLM1
                new_mask = torch.ones(1, new_embeds.shape[0], device=device)
                try:
                    out = self.base_model.model.llm1(
                        inputs_embeds=new_embeds.unsqueeze(0),
                        attention_mask=new_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    hidden = out.hidden_states[-1][0]
                    latent = hidden[start_pos + 1: start_pos + 1 + self.tokens_per_chunk]
                    batch_latents.append(latent)
                except:
                    batch_latents.append(zero_latents[0])

            return torch.stack(batch_latents, dim=0)

    def forward(self, input_ids, labels, context_chunks_ids, context_chunks_mask,
                attention_mask=None, **kwargs):
        """Forward: encode chunks, concat latents, decode with LLM2."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Encode all chunks
        all_latents = []
        for c in range(self.num_chunks):
            chunk_ids = context_chunks_ids[:, c, :]
            chunk_mask = context_chunks_mask[:, c, :]
            latents = self.encode_chunk(chunk_ids, chunk_mask)
            all_latents.append(latents)

        # Concatenate: [batch, 160, 1536]
        concat_latents = torch.cat(all_latents, dim=1)

        # Project to decoder dim: [batch, 160, 2048]
        concat_latents = self.base_model.model.mm_projector(concat_latents)

        # Get decoder input embeddings
        inputs_embeds = self.base_model.model.embed_tokens(input_ids)

        # Replace placeholders with latents
        im_start_token = self.base_model.config.im_start_token
        new_embeds_list = []

        for i in range(batch_size):
            cur_ids = input_ids[i]
            cur_embeds = inputs_embeds[i]
            cur_latents = concat_latents[i]

            start_pos = (cur_ids == im_start_token).nonzero(as_tuple=True)[0]
            if len(start_pos) == 0:
                new_embeds_list.append(cur_embeds)
                continue
            start_pos = start_pos[0].item()

            new_embeds = torch.cat([
                cur_embeds[:start_pos + 1],
                cur_latents,
                cur_embeds[start_pos + 1 + self.total_latent_tokens:]
            ], dim=0)
            new_embeds_list.append(new_embeds)

        # Pad sequences
        max_len = max(e.shape[0] for e in new_embeds_list)
        padded_embeds = []
        padded_labels = []
        padded_mask = []

        for i, embeds in enumerate(new_embeds_list):
            pad_len = max_len - embeds.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, embeds.shape[1], device=device, dtype=embeds.dtype)
                embeds = torch.cat([embeds, pad], dim=0)
            padded_embeds.append(embeds)

            cur_labels = labels[i]
            if cur_labels.shape[0] < max_len:
                label_pad = torch.full((max_len - cur_labels.shape[0],), IGNORE_INDEX,
                                       device=device, dtype=cur_labels.dtype)
                cur_labels = torch.cat([cur_labels, label_pad], dim=0)
            padded_labels.append(cur_labels[:max_len])

            mask = torch.ones(max_len, device=device)
            if pad_len > 0:
                mask[-pad_len:] = 0
            padded_mask.append(mask)

        inputs_embeds = torch.stack(padded_embeds, dim=0)
        labels = torch.stack(padded_labels, dim=0)
        attention_mask = torch.stack(padded_mask, dim=0)

        # Forward through LLM2
        outputs = self.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = self.base_model.lm_head(outputs.last_hidden_state)

        # Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return {"loss": loss, "logits": logits}


# ============ DATASET (following original C3 pattern) ============

class ChunkedDataset(Dataset):
    """Dataset that provides chunked contexts."""

    def __init__(self, data_path, tokenizer, num_chunks=5, chunk_size=6000):
        self.tokenizer = tokenizer
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.total_latent_tokens = num_chunks * 32

        # Set up conversation format
        conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]

        # Load and prepare data
        raw_data = json.load(open(data_path, "r"))
        self.data = self._prepare_long_contexts(raw_data)
        random.shuffle(self.data)

        print(f"Dataset: {len(self.data)} samples with {num_chunks} chunks each")

    def _prepare_long_contexts(self, raw_data):
        """Concatenate samples to create longer contexts."""
        prepared = []
        buffer_ctx = ""
        buffer_resp = ""

        for item in raw_data:
            for turn in item.get("conversations", []):
                if turn["from"] == "context":
                    buffer_ctx += turn["value"] + " "
                elif turn["from"] == "gpt":
                    buffer_resp += turn["value"] + " "

            min_len = self.chunk_size * self.num_chunks
            if len(buffer_ctx) >= min_len:
                prepared.append({
                    "context": buffer_ctx[:min_len],
                    "response": buffer_resp[:8000]
                })
                buffer_ctx = ""
                buffer_resp = ""

        return prepared

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item["context"]
        response = item["response"]

        # Split context into chunks and tokenize each
        chunk_ids_list = []
        chunk_mask_list = []
        placeholder = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 32 + DEFAULT_IM_END_TOKEN

        for i in range(self.num_chunks):
            start = i * self.chunk_size
            chunk = context[start:start + self.chunk_size] if start < len(context) else ""
            chunk_with_ph = chunk + placeholder

            enc = self.tokenizer(chunk_with_ph, return_tensors="pt", truncation=True,
                                 max_length=self.tokenizer.model_max_length)
            chunk_ids_list.append(enc.input_ids[0])
            chunk_mask_list.append(enc.attention_mask[0])

        # Pad chunks to same length
        max_chunk_len = max(c.shape[0] for c in chunk_ids_list)
        for i in range(len(chunk_ids_list)):
            pad_len = max_chunk_len - chunk_ids_list[i].shape[0]
            if pad_len > 0:
                chunk_ids_list[i] = torch.cat([chunk_ids_list[i],
                    torch.full((pad_len,), self.tokenizer.pad_token_id)])
                chunk_mask_list[i] = torch.cat([chunk_mask_list[i],
                    torch.zeros(pad_len, dtype=torch.long)])

        context_chunks_ids = torch.stack(chunk_ids_list, dim=0)
        context_chunks_mask = torch.stack(chunk_mask_list, dim=0)

        # Build decoder prompt with 160 placeholders
        full_ph = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.total_latent_tokens + DEFAULT_IM_END_TOKEN
        human_msg = full_ph + "\nRepeat the text: "

        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], human_msg)
        conv.append_message(conv.roles[1], response)
        prompt = conv.get_prompt()

        # Tokenize (following original - no padding here, collator handles it)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=self.tokenizer.model_max_length)
        input_ids = enc.input_ids[0]

        # Mask labels (only predict response)
        labels = input_ids.clone()
        sep = conv.sep + conv.roles[1]
        parts = prompt.split(sep)
        if len(parts) >= 2:
            prompt_part = parts[0] + sep
            prompt_len = len(self.tokenizer(prompt_part).input_ids)
            labels[:prompt_len] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "context_chunks_ids": context_chunks_ids,
            "context_chunks_mask": context_chunks_mask,
        }


# ============ COLLATOR (following original C3 pattern) ============

@dataclass
class ChunkedDataCollator:
    tokenizer: any

    def __call__(self, instances):
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        context_chunks_ids = [inst["context_chunks_ids"] for inst in instances]
        context_chunks_mask = [inst["context_chunks_mask"] for inst in instances]

        # Pad input_ids and labels
        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                               padding_value=self.tokenizer.pad_token_id)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                            padding_value=IGNORE_INDEX)

        # Stack chunk tensors (already padded per-sample)
        context_chunks_ids = torch.stack(context_chunks_ids, dim=0)
        context_chunks_mask = torch.stack(context_chunks_mask, dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "context_chunks_ids": context_chunks_ids,
            "context_chunks_mask": context_chunks_mask,
        }


# ============ TRAINING ============

def train():
    print("=" * 80)
    print(f"CHUNKED TRAINING: {NUM_CHUNKS} chunks × {TOKENS_PER_CHUNK} = {TOTAL_LATENT_TOKENS} latent tokens")
    print("LLM1: FROZEN | LLM2 + mm_projector: TRAINING")
    print("=" * 80)

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    model_name = 'liufanfanlff/C3-Context-Cascade-Compression'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    # Load model
    print("\n[2/5] Loading model...")
    base_model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map='auto', use_safetensors=True,
    )
    base_model.initialize_special_tokenizer(tokenizer)

    # Wrap
    print("\n[3/5] Creating chunked model...")
    model = C3ChunkedModel(base_model, tokenizer, num_chunks=NUM_CHUNKS)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"   Trainable: {trainable:,} | Frozen: {frozen:,}")

    # Dataset
    print("\n[4/5] Loading data...")
    dataset = ChunkedDataset(DATA_PATH, tokenizer, num_chunks=NUM_CHUNKS, chunk_size=CHUNK_CHAR_SIZE)
    collator = ChunkedDataCollator(tokenizer=tokenizer)

    # Train
    print("\n[5/5] Training...")
    training_args = TrainingArguments(
        output_dir='./c3_chunked_160',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("\n" + "=" * 80)
    trainer.train()

    print("\nSaving...")
    trainer.save_model('./c3_chunked_160')
    tokenizer.save_pretrained('./c3_chunked_160')
    print("DONE!")


if __name__ == "__main__":
    train()
