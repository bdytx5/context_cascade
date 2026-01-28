
import sys
import logging
import pathlib
import torch
import torch.nn as nn
import transformers
import deepspeed
import json
import random
import wandb

from C3.train.trainer_C3 import C3Trainer
from C3.model import *
from C3.data import make_supervised_data_module
from C3.data.conversation_dataset_qwen import ConversationDataset
from C3.utils.arguments import *
from C3.utils.constants import *
from C3.utils.utils import smart_tokenizer_and_embedding_resize
import os
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM, \
                         TrainerCallback


os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['OSS_ENDPOINT'] = "http://oss.i.shaipower.com"

# Target latent token length
NEW_LATENT_TOKEN_LEN = 160

# Multi-task training data paths (balanced sampling between tasks)
DATASET_DIR = "/home/cloud/c3/C3-Context-Cascade-Compression/dataset"
TRAIN_DATA_PATHS = [
    f"{DATASET_DIR}/arxiv_ai_papers.json",      # Repeat/reconstruction task
    f"{DATASET_DIR}/arxiv_summaries.json",       # Summarization task
]

# Validation config
VAL_DATA_PATH = "/home/cloud/c3/C3-Context-Cascade-Compression/dataset/arxiv_ai_papers_val.json"
NUM_VAL_SAMPLES = 0  # Set to 0 to disable eval, or 10+ to enable

# Wandb config
WANDB_PROJECT = "c3-context-compression"
WANDB_ENTITY = 'byyoung3'  # Set to your wandb username/team if needed


def create_val_dataset(tokenizer, data_args, num_samples=NUM_VAL_SAMPLES):
    """Create a small validation dataset from the validation JSON file"""
    if num_samples <= 0:
        return None
    if not os.path.exists(VAL_DATA_PATH):
        print(f"Warning: Validation file not found at {VAL_DATA_PATH}")
        return None

    # Create multimodal config for the dataset
    multimodal_cfg = dict(
        is_multimodal=True,
        image_token_len=data_args.image_token_len,
        image_aspect_ratio=getattr(data_args, 'image_aspect_ratio', 'square'),
        use_im_start_end=data_args.use_im_start_end,
    )

    # Load validation dataset
    val_dataset = ConversationDataset(
        data_path=VAL_DATA_PATH,
        tokenizer=tokenizer,
        multimodal_cfg=multimodal_cfg,
    )

    # Subsample to num_samples
    if len(val_dataset) > num_samples:
        indices = random.sample(range(len(val_dataset)), num_samples)
        val_dataset.list_data_dict = [val_dataset.list_data_dict[i] for i in indices]

    print(f"Validation dataset: {len(val_dataset.list_data_dict)} samples")
    return val_dataset


class BestModelCallback(TrainerCallback):
    """Callback to track and save only the best model based on eval loss"""

    def __init__(self, trainer, output_dir):
        self.trainer = trainer
        self.output_dir = output_dir
        self.best_loss_file = os.path.join(output_dir, "best_eval_loss.txt")

        # Load previous best loss if resuming
        if os.path.exists(self.best_loss_file):
            with open(self.best_loss_file, 'r') as f:
                self.best_eval_loss = float(f.read().strip())
            print(f">>> Resuming with previous best eval_loss: {self.best_eval_loss:.4f}")
        else:
            self.best_eval_loss = float('inf')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        eval_loss = metrics.get('eval_loss', float('inf'))
        print(f"\n>>> Eval loss: {eval_loss:.4f} (best: {self.best_eval_loss:.4f})")

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            best_dir = os.path.join(self.output_dir, "best_model")
            print(f">>> New best model! Overwriting {best_dir}")

            # Remove old best model if exists
            import shutil
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)

            # Save new best model (use save_model which handles ZeRO-3 properly)
            self.trainer.save_model(output_dir=best_dir)

            # Save best loss for resume
            with open(self.best_loss_file, 'w') as f:
                f.write(str(eval_loss))

            # Log to wandb
            wandb.log({"best_eval_loss": eval_loss})
            wandb.run.summary["best_eval_loss"] = eval_loss

            print(f">>> Best model saved (eval_loss={eval_loss:.4f})")


def expand_latent_tokens(model, new_latent_len):
    """Expand Q embedding from current size to new_latent_len tokens"""
    old_Q = model.get_model().Q
    old_latent_len = old_Q.num_embeddings
    hidden_size = old_Q.embedding_dim

    if new_latent_len <= old_latent_len:
        print(f"New latent len ({new_latent_len}) <= old ({old_latent_len}), no expansion needed")
        return model

    print(f"Expanding Q embedding: {old_latent_len} -> {new_latent_len} tokens")

    # Create new larger embedding
    new_Q = nn.Embedding(new_latent_len, hidden_size)

    # Copy old weights and initialize new ones
    # Use GatheredParameters for ZeRO-3 compatibility
    with torch.no_grad():
        # Check if weights are sharded (empty due to ZeRO-3)
        if old_Q.weight.numel() == 0:
            print("Detected ZeRO-3 sharded weights, gathering parameters...")
            with deepspeed.zero.GatheredParameters(old_Q.weight, modifier_rank=0):
                old_weight = old_Q.weight.data.clone()
                new_Q.weight[:old_latent_len] = old_weight
                mean = old_weight.mean().item()
                std = old_weight.std().item()
                nn.init.normal_(new_Q.weight[old_latent_len:], mean=mean, std=std)
        else:
            new_Q.weight[:old_latent_len] = old_Q.weight.clone()
            mean = old_Q.weight.mean().item()
            std = old_Q.weight.std().item()
            nn.init.normal_(new_Q.weight[old_latent_len:], mean=mean, std=std)

    # Replace Q and update config
    model.get_model().Q = new_Q
    model.config.latent_token_len = new_latent_len
    model.get_model().config.latent_token_len = new_latent_len

    print(f"Q embedding expanded successfully. New shape: {new_Q.weight.shape}")
    return model


def train():
    # Check for --resume flag before parsing (so we can remove it)
    resume_from_checkpoint = "--resume" in sys.argv
    if resume_from_checkpoint:
        sys.argv.remove("--resume")

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"c3-latent{NEW_LATENT_TOKEN_LEN}",
        config={
            "latent_token_len": NEW_LATENT_TOKEN_LEN,
            "model": model_args.model_name_or_path,
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "multi_task": True,
            "tasks": ["repeat", "summarization"],
        },
        resume="allow" if resume_from_checkpoint else None,
    )

    # Enable wandb logging in trainer
    training_args.report_to = ["wandb"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
        model_max_length=training_args.model_max_length,
    )

    model = C3QwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_safetensors=True,
        low_cpu_mem_usage=False,  # Ensure weights are fully loaded
    )

    # Expand latent tokens BEFORE moving to device (ZeRO-3 compat)
    model = expand_latent_tokens(model, NEW_LATENT_TOKEN_LEN)

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    model.to(dtype=dtype, device=training_args.device)

    # Now get the (updated) latent token length
    data_args.image_token_len = model.get_model().config.latent_token_len
    data_args.use_im_start_end = model_args.use_im_start_end

    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

    # Set up multi-task data paths for balanced training
    data_args.data_paths = TRAIN_DATA_PATHS
    print(f"Multi-task training with {len(TRAIN_DATA_PATHS)} datasets:")
    for p in TRAIN_DATA_PATHS:
        print(f"  - {p}")

    data_module = make_supervised_data_module(
        interleave=training_args.interleave,
        tokenizer=tokenizer,
        data_args=data_args,
        seed=training_args.seed
    )

    # Create validation dataset
    val_dataset = create_val_dataset(tokenizer, data_args, num_samples=NUM_VAL_SAMPLES)

    # Enable evaluation if we have validation data
    if val_dataset is not None:
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = training_args.save_steps  # Eval at same frequency as save
        training_args.save_strategy = "no"  # Disable regular checkpoint saving - only save best model via callback
        training_args.load_best_model_at_end = False  # We handle this manually
        data_module['eval_dataset'] = val_dataset
        print(f"Evaluation enabled: every {training_args.eval_steps} steps on {len(val_dataset)} samples")

    trainer = C3Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module)

    # Add best model callback
    if val_dataset is not None:
        best_model_callback = BestModelCallback(trainer, training_args.output_dir)
        trainer.add_callback(best_model_callback)

    if resume_from_checkpoint:
        checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        if checkpoints:
            print(f">>> Resuming from checkpoint in {training_args.output_dir}")
            trainer.train(resume_from_checkpoint=True)
        else:
            print(f">>> --resume specified but no checkpoints found in {training_args.output_dir}")
            print(f">>> Starting fresh training")
            trainer.train()
    else:
        trainer.train()
    trainer.save_state()
    last_model_dir = os.path.join(training_args.output_dir, "last_model")
    trainer.save_model(output_dir=last_model_dir)
    print(f"Last model saved to: {last_model_dir}")

    # Print final best eval loss
    if val_dataset is not None:
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation loss: {best_model_callback.best_eval_loss:.4f}")
        print(f"Best model saved to: {training_args.output_dir}/best_model")
        print(f"{'='*60}")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    train()
