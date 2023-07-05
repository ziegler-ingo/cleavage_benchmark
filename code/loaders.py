import random
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader

from utils import random_mask, random_mask_t5


class CleavageLoader:
    def __init__(
        self, train_data, val_data, test_data, tokenizer, batch_size, num_workers
    ):
        self.train_seqs, self.train_lbls = train_data[0], train_data[1]
        self.val_seqs, self.val_lbls = val_data[0], val_data[1]
        self.test_seqs, self.test_lbls = test_data[0], test_data[1]
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.collator = BatchCollator(tokenizer)

    def load(self, model_type, nad, unk_idx):
        assert model_type in self.collator.model_types, "unknown model name"

        train_set = CleavageDataset(self.train_seqs, self.train_lbls)
        val_set = CleavageDataset(self.val_seqs, self.val_lbls)
        test_set = CleavageDataset(self.test_seqs, self.test_lbls)

        train_collate = self.collator.collate_fn(
            model_type, nad, train=True, unk_idx=unk_idx
        )
        val_collate = self.collator.collate_fn(
            model_type, nad, train=False, unk_idx=unk_idx
        )

        if model_type == self.collator.model_types[3]:
            train_bucket_sampler = BucketSampler(
                self.train_seqs, self.tokenizer, self.batch_size
            )
            val_bucket_sampler = BucketSampler(
                self.val_seqs, self.tokenizer, self.batch_size
            )
            test_bucket_sampler = BucketSampler(
                self.test_seqs, self.tokenizer, self.batch_size
            )

            train_sampler = BatchSampler(
                train_bucket_sampler, self.batch_size, drop_last=False
            )
            val_sampler = BatchSampler(
                val_bucket_sampler, self.batch_size, drop_last=False
            )
            test_sampler = BatchSampler(
                test_bucket_sampler, self.batch_size, drop_last=False
            )

            train_loader = DataLoader(
                train_set,
                batch_sampler=train_sampler,
                collate_fn=train_collate,
                num_workers=self.num_workers,
            )
            val_loader = DataLoader(
                val_set,
                batch_sampler=val_sampler,
                collate_fn=val_collate,
                num_workers=self.num_workers,
            )
            test_loader = DataLoader(
                test_set,
                batch_sampler=test_sampler,
                collate_fn=val_collate,
                num_workers=self.num_workers,
            )
            return train_loader, val_loader, test_loader

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=train_collate,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=val_collate,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=val_collate,
            num_workers=self.num_workers,
        )
        return train_loader, val_loader, test_loader


class BatchCollator:
    def __init__(self, tokenizer):
        self.model_types = [
            "BiLSTM",
            "CNN",
            "MLP",
            "Padded",
            "ESM2",
            "T5",
            "FwBw",
        ]
        self.tokenizer = tokenizer

    def collate_fn(self, model_type, nad, train, unk_idx):
        """
        Returns the collate function for the given training arguments.
        """
        assert model_type in self.model_types, "unknown model name"

        if model_type == self.model_types[0]:
            # BiLSTM, BiLSTMAttention, BiLSTMProt2Vec
            return partial(
                self._collate_base_batch, train=train, nad=nad, cnn=False, mlp=False
            )
        elif model_type == self.model_types[1]:
            # CNN
            return partial(
                self._collate_base_batch, train=train, nad=nad, cnn=True, mlp=False
            )
        elif model_type == self.model_types[2]:
            # MLP
            return partial(
                self._collate_base_batch, train=train, nad=nad, cnn=False, mlp=True
            )
        elif model_type == self.model_types[3]:
            # BiLSTMPadded
            return partial(
                self._collate_padded_batch, train=train, unk_idx=unk_idx, nad=nad
            )
        elif model_type == self.model_types[4]:
            # ESM2BiLSTM, ESM2
            return partial(self._collate_esm_batch, train=train, nad=nad)
        elif model_type == self.model_types[5]:
            # T5
            return partial(self._collate_t5_batch, train=train, nad=nad)
        elif model_type == self.model_types[6]:
            # FwBwBiLSTM
            return partial(self._collate_fwbw_batch, train=train, nad=nad)

    def _collate_esm_batch(self, batch, train, nad):
        # ESM2-based models
        batch = [(t[1], t[0]) for t in batch]
        lbl, _, seq = self.tokenizer(batch)
        lbl = torch.tensor([int(l) for l in lbl]).float()

        if nad:
            lbl = lbl.long()
        if train:
            seq = random_mask(seq, unk_idx=3, esm=True)
        return seq.long(), lbl

    def _collate_t5_batch(self, batch, train, nad):
        # T5
        ordered_batch = list(zip(*batch))
        encoded = self.tokenizer.batch_encode_plus(
            [seq.replace("", " ").strip() for seq in ordered_batch[0]]
        )
        seq = torch.tensor(encoded["input_ids"]).long()
        att = torch.tensor(encoded["attention_mask"]).long()
        lbl = torch.tensor([int(l) for l in ordered_batch[1]]).float()

        if nad:
            lbl = lbl.long()
        if train:
            seq, att = random_mask_t5(seq, att, unk_idx=2)
        return seq, att, lbl

    def _collate_padded_batch(self, batch, train, unk_idx, nad):
        # BBPE, WordPiece models
        ordered_batch = list(zip(*batch))
        pad_idx = self.tokenizer.padding["pad_id"]
        seq = torch.tensor(
            [
                s.ids
                for s in self.tokenizer.encode_batch(
                    ordered_batch[0], add_special_tokens=False
                )
            ]
        ).long()
        lbl = torch.tensor([int(l) for l in ordered_batch[1]]).float()
        lengths = torch.sum(seq != pad_idx, dim=1)

        if nad:
            lbl = lbl.long()
        if train:
            seq = random_mask(seq, unk_idx=unk_idx)
        return seq, lbl, lengths

    def _collate_fwbw_batch(self, batch, train, nad):
        # Forward-Backward models
        ordered_batch = list(zip(*batch))
        fw_seq = torch.tensor([self.tokenizer(seq) for seq in ordered_batch[0]]).long()
        bw_seq = torch.flip(fw_seq, dims=(1,))
        lbl = torch.tensor([int(l) for l in ordered_batch[1]]).float()

        if nad:
            lbl = lbl.long()
        if train:
            fw_seq = random_mask(fw_seq, unk_idx=0)
            bw_seq = torch.flip(fw_seq, dims=(1,))
        return bw_seq, fw_seq, lbl

    def _collate_base_batch(self, batch, train, nad, cnn, mlp):
        ordered_batch = list(zip(*batch))
        seq = torch.tensor([self.tokenizer(seq) for seq in ordered_batch[0]]).long()
        lbl = torch.tensor([int(l) for l in ordered_batch[1]]).float()

        if nad:
            lbl = lbl.long()
        if train:
            seq = random_mask(seq, unk_idx=0)
        if mlp:
            seq = F.one_hot(seq, num_classes=21).float().view(seq.shape[0], -1)
        if cnn:
            seq = seq.float()
        return seq, lbl


class CleavageDataset(Dataset):
    def __init__(self, seq, lbl):
        self.seq = seq
        self.lbl = lbl

    def __getitem__(self, idx):
        return self.seq[idx], self.lbl[idx]

    def __len__(self):
        return len(self.lbl)


class BucketSampler(Sampler):
    def __init__(self, seqs, tokenizer, batch_size):
        # pair each sequence with their *tokenized* length
        indices = [(idx, len(tokenizer.encode(s).ids)) for idx, s in enumerate(seqs)]
        random.shuffle(indices)

        idx_pools = []
        # generate pseudo-random batches of (arbitrary) size batch_size * 100
        # each batch of size batch_size * 100 is sorted in itself by seq length
        for i in range(0, len(indices), batch_size * 100):
            idx_pools.extend(
                sorted(indices[i : i + batch_size * 100], key=lambda x: x[1])
            )

        # filter only indices
        self.idx_pools = [x[0] for x in idx_pools]

    def __iter__(self):
        return iter(self.idx_pools)

    def __len__(self):
        return len(self.idx_pools)
