import pandas as pd

#############################################################################
# *** General Dataset Splitting *** #
#############################################################################

df = pd.read_csv("../data/mhc1_cleavages_nc_ranks.csv")
assert df.isna().sum(), "NaN values in dataset"

n_term = df.loc[df["terminus"] == "n", ["cleavage_window", "elution_cleavage"]]
print("Samples in n-term: ", len(n_term))
c_term = df.loc[df["terminus"] == "c", ["cleavage_window", "elution_cleavage"]]
print("Samples in c-term: ", len(c_term))

# n_term
# create train (80%), val (10%), test (10%) files
n_train = n_term.sample(frac=0.8, random_state=1234)
n_val = n_term.drop(n_train.index)
n_test = n_val.sample(frac=0.5, random_state=1234)
n_val = n_val.drop(n_test.index)

assert len(n_train) + len(n_val) + len(n_test) == len(
    n_term
), "Error in dataset splitting"


# c_term
# create train (80%), val (10%), test (10%) files
c_train = c_term.sample(frac=0.8, random_state=1234)
c_val = c_term.drop(c_train.index)
c_test = c_val.sample(frac=0.5, random_state=1234)
c_val = c_val.drop(c_test.index)

assert len(c_train) + len(c_val) + len(c_test) == len(
    c_term
), "Error in dataset splitting"


n_train.to_csv("../data/n_train.csv", index=False)
n_val.to_csv("../data/n_val.csv", index=False)
n_test.to_csv("../data/n_test.csv", index=False)

c_train.to_csv("../data/c_train.csv", index=False)
c_val.to_csv("../data/c_val.csv", index=False)
c_test.to_csv("../data/c_test.csv", index=False)


#############################################################################
# *** k-mer dataset creation for Prot2Vec models, k=3 *** #
#############################################################################

get_3mer = lambda seq: " ".join(seq[i : i + 3] for i in range(len(seq) - 3 + 1))

n_term["kmers"] = n_term["cleavage_window"].map(get_3mer)
c_term["kmers"] = c_term["cleavage_window"].map(get_3mer)

n_term = n_term.rename(columns={"kmers": "sequence", "elution_cleavage": "label"})[
    ["sequence", "label"]
]
c_term = c_term.rename(columns={"kmers": "sequence", "elution_cleavage": "label"})[
    ["sequence", "label"]
]

# n_term
# create train (80%), val (10%), test (10%) files
n_train = n_term.sample(frac=0.8, random_state=1234)
n_val = n_term.drop(n_train.index)
n_test = n_val.sample(frac=0.5, random_state=1234)
n_val = n_val.drop(n_test.index)

assert len(c_train) + len(c_val) + len(c_test) == len(
    c_term
), "Error in dataset splitting"

# c_term
# create train (80%), val (10%), test (10%) files
c_train = c_term.sample(frac=0.8, random_state=1234)
c_val = c_term.drop(c_train.index)
c_test = c_val.sample(frac=0.5, random_state=1234)
c_val = c_val.drop(c_test.index)

assert len(c_train) + len(c_val) + len(c_test) == len(
    c_term
), "Error in dataset splitting"

n_train.to_csv("../data/n_train_3mer.tsv", index=False, sep="\t")
n_val.to_csv("../data/n_val_3mer.tsv", index=False, sep="\t")
n_test.to_csv("../data/n_test_3mer.tsv", index=False, sep="\t")

c_train.to_csv("../data/c_train_3mer.tsv", index=False, sep="\t")
c_val.to_csv("../data/c_val_3mer.tsv", index=False, sep="\t")
c_test.to_csv("../data/c_test_3mer.tsv", index=False, sep="\t")
