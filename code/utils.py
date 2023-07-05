import os
import csv
import math
import random
import numpy as np

import torch


def read_data(path):
    with open(path, "r") as csvfile:
        train_data = list(csv.reader(csvfile))[1:]  # skip col name
        sents, lbls = [], []
        for s, l in train_data:
            sents.append(s)
            lbls.append(l)
    return sents, lbls


def read_data_3mer(path):
    with open(path, "r") as f:
        seqs, lbls = [], []
        for l in f.readlines()[1:]:
            seq, lbl = l.strip().split("\t")
            seqs.append(seq)
            lbls.append(lbl)
    return seqs, lbls


def read_embeddings(path):
    with open(path, "r") as f:
        seq, vec = [], []
        for line in f.readlines()[2:]:  # skip first special chars
            lst = line.split()
            seq.append(lst[0].upper())
            vec.append([float(i) for i in lst[1:]])
        vocab = {s: i for i, s in enumerate(seq)}
        prot2vec = torch.tensor(vec, dtype=torch.float)
    return vocab, prot2vec


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def trainable_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_model_params(model):
    return sum(p.numel() for p in model.parameters())


def linear_rampup(lambda_u, current_epoch, warm_up, rampup_len):
    current = torch.clip((current_epoch - warm_up) / rampup_len, 0.0, 1.0)
    return lambda_u * current


def gelu(x):
    """
    Facebook Research implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def random_mask(seq, unk_idx, esm=False):
    """
    Mask `seq_len // 10` tokens as UNK at random positions per sequence.
    """
    num_samples, seq_len = seq.shape
    if not esm:
        mask_idx = torch.randint(0, seq_len, (num_samples, seq_len // 10))
    else:
        mask_idx = torch.randint(1, seq_len - 1, (num_samples, seq_len // 10))
    masked_seq = torch.scatter(seq, 1, mask_idx, unk_idx)
    return masked_seq


def random_mask_t5(seq1, seq2, unk_idx):
    """
    Mask `seq_len // 10` tokens as UNK at random positions per sequence.
    The mask is applied to two sequences where the same random indices are masked.
    """
    num_samples, seq_len = seq1.shape
    mask_idx = torch.randint(0, seq_len, (num_samples, seq_len // 10))
    masked_seq1 = torch.scatter(seq1, 1, mask_idx, unk_idx)
    masked_seq2 = torch.scatter(seq2, 1, mask_idx, unk_idx)
    return masked_seq1, masked_seq2


def regularized_auc(train_auc, dev_auc, threshold=0.0025):
    """
    Returns development AUC if overfitting is below threshold, otherwise 0.
    """
    return dev_auc if (train_auc - dev_auc) < threshold else 0


def save_metrics_base(*args, path):
    if not os.path.isfile(path):
        with open(path, "w", newline="\n") as f:
            f.write(
                ",".join(
                    [
                        "fold",
                        "epoch",
                        "train_loss",
                        "train_acc",
                        "train_auc",
                        "val_loss",
                        "val_acc",
                        "val_auc",
                    ]
                )
            )
            f.write("\n")
    if args:
        with open(path, "a", newline="\n") as f:
            f.write(",".join([str(arg) for arg in args]))
            f.write("\n")


def save_metrics_coteaching(*args, path):
    if not os.path.isfile(path):
        with open(path, "w", newline="\n") as f:
            f.write(
                ",".join(
                    [
                        "fold",
                        "epoch",
                        "train_loss1",
                        "train_loss2",
                        "train_acc1",
                        "train_acc2",
                        "train_auc1",
                        "train_auc2",
                        "val_loss1",
                        "val_loss2",
                        "val_acc1",
                        "val_acc2",
                        "val_auc1",
                        "val_auc2",
                    ]
                )
            )
            f.write("\n")
    if args:
        with open(path, "a", newline="\n") as f:
            f.write(",".join([str(arg) for arg in args]))
            f.write("\n")


def save_metrics_jocor(*args, path):
    if not os.path.isfile(path):
        with open(path, "w", newline="\n") as f:
            f.write(
                ",".join(
                    [
                        "fold",
                        "epoch",
                        "train_loss",
                        "train_acc1",
                        "train_acc2",
                        "train_auc1",
                        "train_auc2",
                        "val_loss",
                        "val_acc1",
                        "val_acc2",
                        "val_auc1",
                        "val_auc2",
                    ]
                )
            )
            f.write("\n")
    if args:
        with open(path, "a", newline="\n") as f:
            f.write(",".join([str(arg) for arg in args]))
            f.write("\n")


def save_metrics_nad(*args, path):
    if not os.path.isfile(path):
        with open(path, "w", newline="\n") as f:
            f.write(
                ",".join(
                    [
                        "fold",
                        "epoch",
                        "hybrid_loss",
                        "model_loss",
                        "noise_model_loss",
                        "train_acc",
                        "train_auc",
                        "val_loss",
                        "val_acc",
                        "val_auc",
                    ]
                )
            )
            f.write("\n")
    if args:
        with open(path, "a", newline="\n") as f:
            f.write(",".join([str(arg) for arg in args]))
            f.write("\n")
