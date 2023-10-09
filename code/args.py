import argparse

def get_runtime_args():
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

    return parser