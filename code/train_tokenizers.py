import csv
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer  # type: ignore


def read_data(path):
    with open(path, "r") as csvfile:
        train_data = list(csv.reader(csvfile))[1:]  # skip col name
    return [s for s, _ in train_data]


# load train for both n and c terminus
n_train_seqs = read_data("../data/n_train.csv")
c_train_seqs = read_data("../data/c_train.csv")

n_bbpe1k = ByteLevelBPETokenizer()
n_bbpe50k = ByteLevelBPETokenizer()
n_wp50k = BertWordPieceTokenizer()

c_bbpe1k = ByteLevelBPETokenizer()
c_bbpe50k = ByteLevelBPETokenizer()
c_wp50k = BertWordPieceTokenizer()


n_bbpe1k.train_from_iterator(
    iterator=n_train_seqs,
    vocab_size=1000,
    min_frequency=1,
    special_tokens=["<UNK>", "<PAD>"],
)

n_bbpe50k.train_from_iterator(
    iterator=n_train_seqs,
    vocab_size=50_000,
    min_frequency=1,
    special_tokens=["<UNK>", "<PAD>"],
)

n_wp50k.train_from_iterator(
    iterator=n_train_seqs,
    vocab_size=50_000,
    min_frequency=1,
)


c_bbpe1k.train_from_iterator(
    iterator=c_train_seqs,
    vocab_size=1000,
    min_frequency=1,
    special_tokens=["<UNK>", "<PAD>"],
)

c_bbpe50k.train_from_iterator(
    iterator=c_train_seqs,
    vocab_size=50_000,
    min_frequency=1,
    special_tokens=["<UNK>", "<PAD>"],
)

c_wp50k.train_from_iterator(
    iterator=c_train_seqs,
    vocab_size=50_000,
    min_frequency=1,
)

n_bbpe1k.save_model("../params/", prefix="n_bbpe1k")
n_bbpe50k.save_model("../params/", prefix="n_bbpe50k")
n_wp50k.save_model("../params/", prefix="n_wp50k")

c_bbpe1k.save_model("../params/", prefix="c_bbpe1k")
c_bbpe50k.save_model("../params/", prefix="c_bbpe50k")
c_wp50k.save_model("../params/", prefix="c_wp50k")
