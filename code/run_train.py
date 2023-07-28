import os
import json
import argparse
import numpy as np

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim

from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer  # type: ignore
from transformers import T5Tokenizer  # type: ignore

from utils import (
    read_data,
    read_data_3mer,
    read_embeddings,
    seed_everything,
    trainable_model_params,
    total_model_params,
)
from loaders import CleavageLoader
from denoise import CoteachingLoss, JoCoRLoss
from processors import (
    run_epochs_base,
    run_epochs_coteach,
    run_epochs_jocor,
    run_epochs_nad,
    train_or_eval_base,
)

from models import (
    BiLSTM,
    BiLSTMAttention,
    BiLSTMProt2Vec,
    CNNAttention,
    MLP,
    BiLSTMPadded,
    FwBwBiLSTM,
    ESM2,
    ESM2BiLSTM,
    T5BiLSTM,
)


#############################################################################
# *** Argparse Setup *** #
#############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=1234)
parser.add_argument(
    "--terminus",
    type=str,
    default="c",
    choices=["c", "n"],
    help="The terminus to train on",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="BiLSTM",
    choices=["BiLSTM", "CNN", "MLP", "Padded", "ESM2", "T5", "FwBw"],
    help="The overall model type to train",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="BiLSTM",
    choices=[
        "BiLSTM",
        "BiLSTMAttention",
        "BiLSTMProt2Vec",
        "BiLSTM_bbpe1",
        "BiLSTM_bbpe50",
        "BiLSTM_wp50",
        "BiLSTM_fwbw",
        "CNNAttention",
        "MLP",
        "ESM2",
        "ESM2BiLSTM",
        "T5BiLSTM",
    ],
    help="Specific model architecture to train",
)
parser.add_argument(
    "--denoising_method",
    type=str,
    default="None",
    choices=["None", "nad", "coteaching", "coteaching_plus", "jocor", "dividemix"],
    help="The denoising method to use for training",
)
parser.add_argument(
    "--nad",
    default=False,
    action="store_true",
    help="Whether or not noise adaptation is used",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "val", "test"],
    help="Which dataset split to use for training",
)
parser.add_argument(
    "--val_split",
    type=str,
    default="val",
    choices=["train", "val", "test"],
    help="Which dataset split to use for validation",
)
parser.add_argument(
    "--test_split",
    type=str,
    default="test",
    choices=["train", "val", "test"],
    help="Which dataset split to use for testing",
)
parser.add_argument("--seq_len", type=int, default=10, help="Sequence length of input")
parser.add_argument(
    "--vocab_size",
    type=int,
    default=1000,
    choices=[1000, 50000],
    help="Vocab size for trainable tokenizers",
)
parser.add_argument(
    "--k_fold", type=int, default=5, help="Number of cross-validation runs"
)
parser.add_argument(
    "--early_stop",
    type=int,
    default=5,
    help="Number of overfitting or negative progress epochs before training is stopped",
)
parser.add_argument("--saving_path", type=str, default="../results/")
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--batch_norm", default=False, action="store_true")
parser.add_argument(
    "--unk_idx",
    type=int,
    default=0,
    choices=[0, 1, 2, 3],
    help="Index of UNK token in vocabulary",
)
parser.add_argument(
    "--pad_idx",
    type=int,
    default=1,
    choices=[0, 1],
    help="Index of PAD token in vocabulary. Only relevant for `Padded` models.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=2,
    help="Number of workers to be used in torch dataloaders",
)
parser.add_argument("--num_epochs", type=int, default=15)
parser.add_argument(
    "--num_warmup", type=int, default=1, help="Warmup epochs for noise adaptation"
)
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument(
    "--embedding_dim", type=int, default=150, help="Embedding dimension"
)
parser.add_argument(
    "--rnn_size1", type=int, default=256, help="First RNN hidden dimension"
)
parser.add_argument(
    "--rnn_size2", type=int, default=512, help="Second RNN hidden dimension"
)
parser.add_argument(
    "--linear_size1", type=int, default=128, help="First Linear layer hidden size"
)
parser.add_argument(
    "--linear_size2", type=int, default=64, help="Second Linear layer hidden size"
)
parser.add_argument(
    "--out_neurons", type=int, default=1, help="Number of output neurons"
)
parser.add_argument(
    "--num_heads1", type=int, default=2, help="Attention heads of first attention layer"
)
parser.add_argument(
    "--num_heads2",
    type=int,
    default=2,
    help="Attention heads of second attention layer",
)
parser.add_argument(
    "--seq_enc_emb_dim",
    type=int,
    default=100,
    help="Embedding size of fwbw sequence encoder",
)
parser.add_argument(
    "--seq_enc_rnn_size",
    type=int,
    default=200,
    help="RNN size of fwbw sequence encoder",
)
parser.add_argument(
    "--num_filters1", type=int, default=220, help="Filters in first CNN layer"
)
parser.add_argument(
    "--filter_size1", type=int, default=1, help="Filter size of first CNN layer"
)
parser.add_argument(
    "--num_filters2", type=int, default=262, help="Filters in second CNN layer"
)
parser.add_argument(
    "--filter_size2a", type=int, default=3, help="Filter size of second_a CNN layer"
)
parser.add_argument(
    "--filter_size2b", type=int, default=17, help="Filter size of second_b CNN layer"
)
parser.add_argument(
    "--filter_size2c", type=int, default=13, help="Filter size of second_c CNN layer"
)
parser.add_argument(
    "--num_filters3", type=int, default=398, help="Filters in third CNN layer"
)
parser.add_argument(
    "--filter_size3a", type=int, default=11, help="Filter size of third_a CNN layer"
)
parser.add_argument(
    "--filter_size3b", type=int, default=15, help="Filter size of third_b CNN layer"
)
parser.add_argument(
    "--filter_size3c", type=int, default=19, help="Filter size of third_c CNN layer"
)
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

# denoising arguments
parser.add_argument(
    "--num_gradual", type=int, default=10, help="Epochs for linear rate schedule"
)
parser.add_argument(
    "--noisy_rate", type=float, default=0.2, help="Assumed noisy rate of dataset"
)
parser.add_argument(
    "--beta", type=float, default=0.8, help="Weight of noise model loss"
)

args = parser.parse_args()


#############################################################################
# *** Data loading, Tokenizer & Model Config Setup *** #
#############################################################################

seed_everything(seed=1234)
denoise = args.denoising_method if args.denoising_method != "None" else ""
model_path = os.path.abspath(args.saving_path + args.terminus + "_" + args.model_name)
if denoise:
    model_path = model_path + "_" + denoise
param_path = model_path + "/params/"
logging_path = model_path + "/files/"

if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(param_path)
    os.makedirs(logging_path)
    print(f"created directory path {model_path}")
    print(f"created directory path {param_path}")
    print(f"created directory path {logging_path}")

if not "Prot2Vec" in args.model_name:
    train_data = read_data(f"../data/{args.terminus}_{args.train_split}.csv")
    val_data = read_data(f"../data/{args.terminus}_{args.val_split}.csv")
    test_data = read_data(f"../data/{args.terminus}_{args.test_split}.csv")
else:
    train_data = read_data_3mer(f"../data/{args.terminus}_{args.train_split}_3mer.tsv")
    val_data = read_data_3mer(f"../data/{args.terminus}_{args.val_split}_3mer.tsv")
    test_data = read_data_3mer(f"../data/{args.terminus}_{args.test_split}_3mer.tsv")

if args.model_type == "Padded":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load tokenizer for respective model
if "bbpe1" in args.model_name:
    model_to_load = BiLSTMPadded
    optim_to_load = optim.Adam
    bbpe1_vocab = f"../params/{args.terminus}_bbpe1k-vocab.json"
    bbpe1_merges = f"../params/{args.terminus}_bbpe1k-merges.txt"
    tokenizer = ByteLevelBPETokenizer.from_file(
        bbpe1_vocab, bbpe1_merges, lowercase=False
    )
    tokenizer.enable_padding(
        pad_id=args.pad_idx, pad_token="<PAD>"
    )  # pad_id should be 1

    loader = CleavageLoader(
        train_data,
        val_data,
        test_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, test_loader = loader.load(
        "Padded", nad=args.nad, unk_idx=args.unk_idx
    )  # unk_idx should be 0

    model_cfg = {
        "vocab_size": tokenizer.get_vocab_size(),
        "embedding_dim": args.embedding_dim,
        "rnn_size1": args.rnn_size1,
        "rnn_size2": args.rnn_size2,
        "hidden_size": args.linear_size1,
        "dropout": args.dropout,
        "out_neurons": args.out_neurons,
        "pad_idx": args.pad_idx,
    }

elif "bbpe50" in args.model_name:
    model_to_load = BiLSTMPadded
    optim_to_load = optim.Adam
    bbpe50_vocab = f"../params/{args.terminus}_bbpe50k-vocab.json"
    bbpe50_merges = f"../params/{args.terminus}_bbpe50k-merges.txt"
    tokenizer = ByteLevelBPETokenizer.from_file(
        bbpe50_vocab, bbpe50_merges, lowercase=False
    )
    tokenizer.enable_padding(
        pad_id=args.pad_idx, pad_token="<PAD>"
    )  # pad_idx should be 1

    loader = CleavageLoader(
        train_data,
        val_data,
        test_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, test_loader = loader.load(
        "Padded", nad=args.nad, unk_idx=args.unk_idx
    )  # unk_idx should be 0

    model_cfg = {
        "vocab_size": tokenizer.get_vocab_size(),
        "embedding_dim": args.embedding_dim,
        "rnn_size1": args.rnn_size1,
        "rnn_size2": args.rnn_size2,
        "hidden_size": args.linear_size1,
        "dropout": args.dropout,
        "out_neurons": args.out_neurons,
        "pad_idx": args.pad_idx,
    }

elif "wp50" in args.model_name:
    model_to_load = BiLSTMPadded
    optim_to_load = optim.Adam
    wp50_vocab = f"../params/{args.terminus}_wp50k-vocab.txt"
    tokenizer = BertWordPieceTokenizer.from_file(wp50_vocab, lowercase=False)
    tokenizer.enable_padding(
        pad_id=args.pad_idx, pad_token="[PAD]"
    )  # pad_id should be 0

    loader = CleavageLoader(
        train_data,
        val_data,
        test_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, test_loader = loader.load(
        "Padded", nad=args.nad, unk_idx=args.unk_idx
    )  # unk_idx should be 1

    model_cfg = {
        "vocab_size": tokenizer.get_vocab_size(),
        "embedding_dim": args.embedding_dim,
        "rnn_size1": args.rnn_size1,
        "rnn_size2": args.rnn_size2,
        "hidden_size": args.linear_size1,
        "dropout": args.dropout,
        "out_neurons": args.out_neurons,
        "pad_idx": args.pad_idx,
    }

elif "ESM2" in args.model_name:
    esm2, vocab = torch.hub.load("facebookresearch/esm:main", "esm2_t30_150M_UR50D")
    tokenizer = vocab.get_batch_converter()

    loader = CleavageLoader(
        train_data,
        val_data,
        test_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, test_loader = loader.load(
        "ESM2", nad=args.nad, unk_idx=args.unk_idx
    )  # unk_idx should be 3

    if "BiLSTM" in args.model_name:
        model_to_load = ESM2BiLSTM
        optim_to_load = optim.Adam
        model_cfg = {
            "esm2": esm2,
            "rnn_size": args.rnn_size1,
            "hidden_size": args.linear_size1,
            "dropout": args.dropout,
            "out_neurons": args.out_neurons,
        }
    else:
        model_to_load = ESM2
        optim_to_load = optim.AdamW
        model_cfg = {
            "pretrained_model": esm2,
            "dropout": args.dropout,
            "out_neurons": args.out_neurons,
        }

elif "T5" in args.model_name:
    model_to_load = T5BiLSTM
    optim_to_load = optim.Adam
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )

    loader = CleavageLoader(
        train_data,
        val_data,
        test_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, test_loader = loader.load(
        "T5", nad=args.nad, unk_idx=args.unk_idx
    )  # unk_idx should be 2

    model_cfg = {
        "rnn_size": args.rnn_size1,
        "hidden_size": args.linear_size1,
        "dropout": args.dropout,
        "out_neurons": args.out_neurons,
    }

elif "Prot2Vec" in args.model_name:
    model_to_load = BiLSTMProt2Vec
    optim_to_load = optim.Adam
    vocab, embeddings = read_embeddings("../params/uniref_3M.vec")
    tokenizer = lambda seq: [vocab.get(s, 0) for s in seq.split()]  # k-mer sequences

    loader = CleavageLoader(
        train_data,
        val_data,
        test_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, test_loader = loader.load(
        "BiLSTM", nad=args.nad, unk_idx=args.unk_idx
    )  # unk_idx should be 0

    model_cfg = {
        "pretrained_embeds": embeddings,
        "rnn_size": args.rnn_size1,
        "hidden_size": args.linear_size1,
        "dropout": args.dropout,
        "out_neurons": args.out_neurons,
    }

else:
    vocab = torch.load("../params/vocab.pt")
    tokenizer = lambda x: vocab(list(x))

    if "CNN" in args.model_name:
        model_to_load = CNNAttention
        optim_to_load = optim.Adam
        loader = CleavageLoader(
            train_data,
            val_data,
            test_data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train_loader, val_loader, test_loader = loader.load(
            "CNN", nad=args.nad, unk_idx=args.unk_idx
        )  # unk_idx should be 0

        model_cfg = {
            "seq_len": args.seq_len,
            "kernel_size1": args.filter_size1,
            "kernel_size2a": args.filter_size2a,
            "kernel_size2b": args.filter_size2b,
            "kernel_size2c": args.filter_size2c,
            "kernel_size3a": args.filter_size3a,
            "kernel_size3b": args.filter_size3b,
            "kernel_size3c": args.filter_size3c,
            "num_filters1": args.num_filters1,
            "num_filters2": args.num_filters2,
            "num_filters3": args.num_filters3,
            "attention_hidden1": args.num_heads1,
            "attention_hidden2": args.num_heads2,
            "hidden_size1": args.linear_size1,
            "hidden_size2": args.linear_size2,
            "dropout": args.dropout,
            "out_neurons": args.out_neurons,
        }

    elif "MLP" in args.model_name:
        model_to_load = MLP
        optim_to_load = optim.Adam
        loader = CleavageLoader(
            train_data,
            val_data,
            test_data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        train_loader, val_loader, test_loader = loader.load(
            "MLP", nad=args.nad, unk_idx=args.unk_idx
        )  # unk_idx should be 0

        model_cfg = {
            "vocab_size": len(vocab),
            "seq_len": args.seq_len,
            "hidden_size": args.linear_size1,
            "dropout": args.dropout,
            "out_neurons": args.out_neurons,
        }

    elif args.model_name == "BiLSTMAttention":
        model_to_load = BiLSTMAttention
        optim_to_load = optim.Adam
        loader = CleavageLoader(
            train_data,
            val_data,
            test_data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train_loader, val_loader, test_loader = loader.load(
            "BiLSTM", nad=args.nad, unk_idx=args.unk_idx
        )  # unk_idx should be 0

        model_cfg = {
            "vocab_size": len(vocab),
            "embedding_dim": args.embedding_dim,
            "rnn_size": args.rnn_size1,
            "hidden_size": args.linear_size1,
            "num_heads": args.num_heads1,
            "dropout": args.dropout,
            "out_neurons": args.out_neurons,
        }

    elif "fwbw" in args.model_name:
        # FwBwBiLSTM
        model_to_load = FwBwBiLSTM
        optim_to_load = optim.Adam

        loader = CleavageLoader(
            train_data,
            val_data,
            test_data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        train_loader, val_loader, test_loader = loader.load(
            "FwBw", nad=args.nad, unk_idx=args.unk_idx
        )  # unk_idx should be 0

        model_cfg = {
            "vocab_size": len(vocab),
            "seq_enc_emb_dim": args.seq_enc_emb_dim,
            "seq_enc_rnn_size": args.seq_enc_rnn_size,
            "rnn_size1": args.rnn_size1,
            "rnn_size2": args.rnn_size2,
            "hidden_size": args.linear_size1,
            "out_neurons": args.out_neurons,
            "dropout": args.dropout,
        }

    else:
        # normal BiLSTM
        model_to_load = BiLSTM
        optim_to_load = optim.Adam

        loader = CleavageLoader(
            train_data,
            val_data,
            test_data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train_loader, val_loader, test_loader = loader.load(
            "BiLSTM", nad=args.nad, unk_idx=args.unk_idx
        )  # unk_idx should be 0

        model_cfg = {
            "vocab_size": len(vocab),
            "embedding_dim": args.embedding_dim,
            "rnn_size1": args.rnn_size1,
            "rnn_size2": args.rnn_size2,
            "hidden_size": args.linear_size1,
            "dropout": args.dropout,
            "out_neurons": args.out_neurons,
            "seq_len": args.seq_len,
            "batch_norm": args.batch_norm,
        }


#############################################################################
# *** K-Fold Cross-Validation Runs *** #
#############################################################################

if args.k_fold > 0:
    train_seqs = np.array(train_data[0] + val_data[0])
    train_lbls = np.array(train_data[1] + val_data[1])
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.random_seed)

    # get new split
    total_highest_val_auc = 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_seqs), 1):
        fold_highest_val_auc = 0
        X_tr = train_seqs[train_idx]
        y_tr = train_lbls[train_idx]
        X_val = train_seqs[val_idx]
        y_val = train_lbls[val_idx]

        if args.model_type == "Padded":
            # train new tokenizer every fold, use tokenizer parallelism for training
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

            if "bbpe1" in args.model_name or "bbpe50" in args.model_name:
                tokenizer = ByteLevelBPETokenizer(lowercase=False)
                tokenizer.train_from_iterator(
                    iterator=X_tr,
                    vocab_size=args.vocab_size,
                    min_frequency=1,
                    special_tokens=["<UNK>", "<PAD>"],
                )
                tokenizer.enable_padding(pad_id=args.pad_idx, pad_token="<PAD>")
                tokenizer.save_model(param_path, prefix=f"fold{fold}")
            else:
                tokenizer = BertWordPieceTokenizer(lowercase=False)
                tokenizer.train_from_iterator(
                    iterator=X_tr, vocab_size=args.vocab_size, min_frequency=1
                )
                tokenizer.enable_padding(pad_id=args.pad_idx, pad_token="[PAD]")
                tokenizer.save_model(param_path, prefix=f"fold{fold}")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        _loader = CleavageLoader(
            (X_tr, y_tr),
            (X_val, y_val),
            train_data,
            tokenizer,
            args.batch_size,
            args.num_workers,
        )
        train_loader, val_loader, test_loader = _loader.load(
            args.model_type, nad=args.nad, unk_idx=args.unk_idx
        )

        # reset model with each new fold
        if "model" in globals():
            del model  # type: ignore
            print("deleted model")
        if "model1" in globals():
            del model1  # type: ignore
            print("deleted model1")
        if "model2" in globals():
            del model2  # type: ignore
            print("deleted model2")

        # check all denoising methods and execute training through all epochs
        if args.denoising_method == "None":
            print("running epochs for denoising_method=None")
            model = model_to_load(**model_cfg).to(args.device)
            optimizer = optim_to_load(model.parameters(), lr=args.lr)
            criterion = nn.BCEWithLogitsLoss()
            if "T5" in args.model_name:
                scaler = torch.cuda.amp.GradScaler()  # type: ignore
            else:
                scaler = None
            total_highest_val_auc, fold_highest_val_auc = run_epochs_base(
                model=model,
                model_type=args.model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                device=args.device,
                fold=fold,
                num_epochs=args.num_epochs,
                total_highest_val_auc=total_highest_val_auc,
                fold_highest_val_auc=fold_highest_val_auc,
                early_stop=args.early_stop,
                logging_path=logging_path + "train_results.csv",
                param_path=param_path,
                optim=optimizer,
                scaler=scaler,
            )

        elif args.denoising_method == "nad":
            print("running epochs for denoising_method=nad")
            model = model_to_load(**model_cfg).to(args.device)
            optimizer = optim_to_load(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()
            if "T5" in args.model_name:
                scaler = torch.cuda.amp.GradScaler()  # type: ignore
            else:
                scaler = None
            total_highest_val_auc, fold_highest_val_auc = run_epochs_nad(
                model_type=args.model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                device=args.device,
                fold=fold,
                num_epochs=args.num_epochs,
                num_warmup=args.num_warmup,
                learning_rate=args.lr,
                beta=args.beta,
                total_highest_val_auc=total_highest_val_auc,
                fold_highest_val_auc=fold_highest_val_auc,
                early_stop=args.early_stop,
                logging_path=logging_path + "train_results.csv",
                param_path=param_path,
                criterion=criterion,
                optim=optimizer,
                scaler=scaler,
            )

        else:
            forget_rate = args.noisy_rate / 2
            rate_schedule = torch.ones(args.num_epochs) * forget_rate
            rate_schedule[: args.num_gradual] = torch.linspace(
                0, forget_rate, args.num_gradual
            )

            model1 = model_to_load(**model_cfg).to(args.device)
            model2 = model_to_load(**model_cfg).to(args.device)
            criterion = nn.BCEWithLogitsLoss()

            if args.denoising_method == "jocor":
                print("running epochs for denoising_method=jocor")
                jocor_criterion = JoCoRLoss()
                optimizer = optim_to_load(
                    set(list(model1.parameters()) + list(model2.parameters())),
                    lr=args.lr,
                )
                if "T5" in args.model_name:
                    scaler = torch.cuda.amp.GradScaler()  # type: ignore
                else:
                    scaler = None
                total_highest_val_auc, fold_highest_val_auc = run_epochs_jocor(
                    model_type=args.model_type,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model1=model1,
                    model2=model2,
                    rate_schedule=rate_schedule,
                    device=args.device,
                    fold=fold,
                    num_epochs=args.num_epochs,
                    total_highest_val_auc=total_highest_val_auc,
                    fold_highest_val_auc=fold_highest_val_auc,
                    early_stop=args.early_stop,
                    logging_path=logging_path + "train_results.csv",
                    param_path=param_path,
                    jocor_criterion=jocor_criterion,
                    optim=optimizer,
                    scaler=scaler,
                )

            else:
                print("running epochs for denoising_method=coteaching")
                cot_criterion = CoteachingLoss()
                cot_plus_train = True if "plus" in args.denoising_method else None
                if cot_plus_train:
                    print("running coteaching plus version")
                optimizer1 = optim_to_load(model1.parameters(), lr=args.lr)
                optimizer2 = optim_to_load(model2.parameters(), lr=args.lr)
                if "T5" in args.model_name:
                    scaler1 = torch.cuda.amp.GradScaler()  # type: ignore
                    scaler2 = torch.cuda.amp.GradScaler()  # type: ignore
                else:
                    scaler1, scaler2 = None, None
                total_highest_val_auc, fold_highest_val_auc = run_epochs_coteach(
                    model_type=args.model_type,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model1=model1,
                    model2=model2,
                    rate_schedule=rate_schedule,
                    device=args.device,
                    fold=fold,
                    num_epochs=args.num_epochs,
                    total_highest_val_auc=total_highest_val_auc,
                    fold_highest_val_auc=fold_highest_val_auc,
                    early_stop=args.early_stop,
                    logging_path=logging_path + "train_results.csv",
                    param_path=param_path,
                    cot_criterion=cot_criterion,
                    criterion=criterion,
                    optim1=optimizer1,
                    optim2=optimizer2,
                    scaler1=scaler1,
                    scaler2=scaler2,
                    cot_plus_train=cot_plus_train,
                )


#############################################################################
# *** A single non-k-fold run on original train/validation set *** #
#############################################################################

else:
    fold = 0  # placeholder value
    total_highest_val_auc, fold_highest_val_auc = 0, 0

    # check all denoising methods and execute training through all epochs
    if args.denoising_method == "None":
        print("running epochs for denoising_method=None")
        model = model_to_load(**model_cfg).to(args.device)
        optimizer = optim_to_load(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()
        if "T5" in args.model_name:
            scaler = torch.cuda.amp.GradScaler()  # type: ignore
        else:
            scaler = None
        total_highest_val_auc, fold_highest_val_auc = run_epochs_base(
            model=model,
            model_type=args.model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=args.device,
            fold=fold,
            num_epochs=args.num_epochs,
            total_highest_val_auc=total_highest_val_auc,
            fold_highest_val_auc=fold_highest_val_auc,
            early_stop=args.early_stop,
            logging_path=logging_path + "train_results.csv",
            param_path=param_path,
            optim=optimizer,
            scaler=scaler,
        )

    elif args.denoising_method == "nad":
        print("running epochs for denoising_method=nad")
        model = model_to_load(**model_cfg).to(args.device)
        optimizer = optim_to_load(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        if "T5" in args.model_name:
            scaler = torch.cuda.amp.GradScaler()  # type: ignore
        else:
            scaler = None
        total_highest_val_auc, fold_highest_val_auc = run_epochs_nad(
            model_type=args.model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            device=args.device,
            fold=fold,
            num_epochs=args.num_epochs,
            num_warmup=args.num_warmup,
            learning_rate=args.lr,
            beta=args.beta,
            total_highest_val_auc=total_highest_val_auc,
            fold_highest_val_auc=fold_highest_val_auc,
            early_stop=args.early_stop,
            logging_path=logging_path + "train_results.csv",
            param_path=param_path,
            criterion=criterion,
            optim=optimizer,
            scaler=scaler,
        )

    else:
        forget_rate = args.noisy_rate / 2
        rate_schedule = torch.ones(args.num_epochs) * forget_rate
        rate_schedule[: args.num_gradual] = torch.linspace(
            0, forget_rate, args.num_gradual
        )

        model1 = model_to_load(**model_cfg).to(args.device)
        model2 = model_to_load(**model_cfg).to(args.device)
        criterion = nn.BCEWithLogitsLoss()

        if args.denoising_method == "jocor":
            print("running epochs for denoising_method=jocor")
            jocor_criterion = JoCoRLoss()
            optimizer = optim_to_load(
                list(model1.parameters()) + list(model2.parameters()), lr=args.lr
            )
            if "T5" in args.model_name:
                scaler = torch.cuda.amp.GradScaler()  # type: ignore
            else:
                scaler = None
            total_highest_val_auc, fold_highest_val_auc = run_epochs_jocor(
                model_type=args.model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                model1=model1,
                model2=model2,
                rate_schedule=rate_schedule,
                device=args.device,
                fold=fold,
                num_epochs=args.num_epochs,
                total_highest_val_auc=total_highest_val_auc,
                fold_highest_val_auc=fold_highest_val_auc,
                early_stop=args.early_stop,
                logging_path=logging_path + "train_results.csv",
                param_path=param_path,
                jocor_criterion=jocor_criterion,
                optim=optimizer,
                scaler=scaler,
            )

        else:
            print("running epochs for denoising_method=coteaching")
            cot_criterion = CoteachingLoss()
            cot_plus_train = True if "plus" in args.denoising_method else None
            optimizer1 = optim_to_load(model1.parameters(), lr=args.lr)
            optimizer2 = optim_to_load(model2.parameters(), lr=args.lr)
            if "T5" in args.model_name:
                scaler1 = torch.cuda.amp.GradScaler()  # type: ignore
                scaler2 = torch.cuda.amp.GradScaler()  # type: ignore
            else:
                scaler1, scaler2 = None, None
            total_highest_val_auc, fold_highest_val_auc = run_epochs_coteach(
                model_type=args.model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                model1=model1,
                model2=model2,
                rate_schedule=rate_schedule,
                device=args.device,
                fold=fold,
                num_epochs=args.num_epochs,
                total_highest_val_auc=total_highest_val_auc,
                fold_highest_val_auc=fold_highest_val_auc,
                early_stop=args.early_stop,
                logging_path=logging_path + "train_results.csv",
                param_path=param_path,
                cot_criterion=cot_criterion,
                criterion=criterion,
                optim1=optimizer1,
                optim2=optimizer2,
                scaler1=scaler1,
                scaler2=scaler2,
                cot_plus_train=cot_plus_train,
            )

#############################################################################
# *** Load best model and test on test set *** #
#############################################################################

best_model = sorted(
    [f for f in os.listdir(param_path) if f.endswith(".pt")],
    reverse=True,
)[0]
model = model_to_load(**model_cfg).to(args.device)
criterion = nn.BCEWithLogitsLoss() if not args.nad else nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler() if "T5" in args.model_name else None  # type: ignore
model.load_state_dict(torch.load(param_path + best_model))
print("Loaded model for testing: ", best_model)
model.eval()

# if we trained tokenizers, we need to load the tokenizer from the best fold
# for all other cases, we have pre-trained BBPE and WP tokenizers on original train split
# ESM2 and T5 have respective pre-trained tokenizers provided
# everywhere else, we have the default 20 amino acid + UNK vocab
if args.k_fold > 0 and "bbpe" in args.model_name:
    best_fold = best_model.split("_")[1]  # type: ignore
    all_vocabs = [
        f.split("-") for f in os.listdir(param_path) if f.endswith("vocab.json")
    ]
    all_merges = [
        f.split("-") for f in os.listdir(param_path) if f.endswith("merges.txt")
    ]
    best_vocab = [f + "-" + v for f, v in all_vocabs if f == best_fold][0]
    best_merge = [f + "-" + v for f, v in all_merges if f == best_fold][0]
    tokenizer = ByteLevelBPETokenizer.from_file(
        param_path + f"{best_vocab}", param_path + f"{best_merge}", lowercase=False
    )
    tokenizer.enable_padding(pad_id=args.pad_idx, pad_token="<PAD>")
    print(f"BBPE loaded best vocab: {best_vocab}, best merge: {best_merge}")

    loader = CleavageLoader(
        train_data, val_data, test_data, tokenizer, args.batch_size, args.num_workers
    )
    _, _, test_loader = loader.load(args.model_type, nad=args.nad, unk_idx=args.unk_idx)

elif args.k_fold > 0 and "wp50" in args.model_name:
    best_fold = best_model.split("_")[1]  # type: ignore
    all_vocabs = [
        f.split("-") for f in os.listdir(param_path) if f.endswith("vocab.txt")
    ]
    best_vocab = [f + "-" + v for f, v in all_vocabs if f == best_fold][0]
    tokenizer = BertWordPieceTokenizer.from_file(
        param_path + f"{best_vocab}", lowercase=False
    )
    tokenizer.enable_padding(pad_id=args.pad_idx, pad_token="[PAD]")

    print(f"WP loaded best vocab: {best_vocab}")
    loader = CleavageLoader(
        train_data, val_data, test_data, tokenizer, args.batch_size, args.num_workers
    )
    _, _, test_loader = loader.load(args.model_type, nad=args.nad, unk_idx=args.unk_idx)


with torch.no_grad():
    test_res = train_or_eval_base(
        model,
        args.model_type,
        test_loader,
        criterion,
        args.device,
        scaler=scaler,
        nad=args.nad,
    )
num_trainable_params = trainable_model_params(model)
num_total_params = total_model_params(model)

test_results = {
    "test_results": list(test_res),
    "num_params": [num_trainable_params, num_total_params],
}

with open(logging_path + "test_results.json", "w") as f:
    json.dump(test_results, f)

print("Saved test results at: ", logging_path + "test_results.json")
print(
    f"Test Set Performance: Loss: {test_res[0]:.6f}, Acc: {test_res[1]:.4f}, AUC: {test_res[2]:.4f}"
)
print("Number of trainable model parameters: ", num_trainable_params)
print("Number of total model parameters: ", num_total_params)
