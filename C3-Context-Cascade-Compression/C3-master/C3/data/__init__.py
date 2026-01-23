
import torch
import transformers
import random
from dataclasses import dataclass, field

from C3.utils.constants import *


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):

        input_ids, labels, context_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "context_ids"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        context_ids = torch.nn.utils.rnn.pad_sequence(
            context_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            context_ids=context_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            context_attention_mask=context_ids.ne(self.tokenizer.pad_token_id),
        )
        return batch


class InterleavedMultiTaskDataset(torch.utils.data.Dataset):
    """Dataset that interleaves multiple datasets for balanced multi-task training.

    Upsamples smaller datasets to match the largest, then interleaves.
    This ensures roughly equal representation of each task per epoch.
    """

    def __init__(self, datasets, names=None):
        self.datasets = datasets
        self.names = names or [f"task_{i}" for i in range(len(datasets))]

        # Find max size and upsample indices
        sizes = [len(ds) for ds in datasets]
        max_size = max(sizes)

        # Create interleaved index list
        # Each dataset contributes max_size samples (with repetition if needed)
        self.index_map = []  # [(dataset_idx, sample_idx), ...]

        for ds_idx, ds in enumerate(datasets):
            ds_size = len(ds)
            # Create indices for this dataset, repeating if necessary
            indices = list(range(ds_size))
            if ds_size < max_size:
                # Upsample by repeating
                repeats = max_size // ds_size
                remainder = max_size % ds_size
                indices = indices * repeats + random.sample(indices, remainder)
            random.shuffle(indices)

            for sample_idx in indices:
                self.index_map.append((ds_idx, sample_idx))

        # Shuffle the interleaved indices
        random.shuffle(self.index_map)

        print(f"InterleavedMultiTaskDataset: {len(self.index_map)} samples")
        for i, (ds, name) in enumerate(zip(datasets, self.names)):
            print(f"  [{name}] {len(ds)} original -> {max_size} upsampled")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.index_map[idx]
        return self.datasets[ds_idx][sample_idx]


def make_supervised_data_module(interleave, tokenizer, data_args):

    if data_args.conversation_version == 'mpt':
        from C3.data.conversation_dataset_qwen import ConversationDataset
        dataset_cls = ConversationDataset

    multimodal_cfg = dict(
        image_token_len=data_args.image_token_len,
        use_im_start_end=data_args.use_im_start_end,
    )

    # Support multiple data paths for multi-task training
    data_paths = getattr(data_args, 'data_paths', None)

    if data_paths and len(data_paths) > 1:
        # Multi-task: load and interleave datasets
        datasets = []
        names = []

        for path in data_paths:
            ds = dataset_cls(
                tokenizer=tokenizer,
                data_path=path,
                multimodal_cfg=multimodal_cfg,
            )
            datasets.append(ds)
            # Extract task name from filename
            name = path.split('/')[-1].replace('.json', '')
            names.append(name)
            print(f"Loaded {path}: {len(ds)} samples")

        # Create interleaved dataset (handles balancing internally)
        train_dataset = InterleavedMultiTaskDataset(datasets, names)

        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
    else:
        # Single dataset (original behavior)
        train_dataset = dataset_cls(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            multimodal_cfg=multimodal_cfg,
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset,
                    eval_dataset=None,
                    data_collator=data_collator)
