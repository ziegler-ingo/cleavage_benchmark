import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import T5EncoderModel  # type: ignore

from utils import gelu


#############################################################################
# *** General models (used for everything except DivideMix *** #
#############################################################################


class BiLSTM(nn.Module):
    """
    Model architecture is based on:

    Ozols, M., Eckersley, A., Platt, C. I., Stewart-McGuinness, C.,
    Hibbert, S. A., Revote, J., ... & Sherratt, M. J. (2021).
    Predicting proteolysis in complex proteomes using deep learning.
    International Journal of Molecular Sciences, 22(6), 3071.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        rnn_size1,
        rnn_size2,
        hidden_size,
        dropout,
        out_neurons,
        seq_len=10,
        batch_norm=False,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(seq_len)
            self.bn2 = nn.BatchNorm1d(seq_len)
            self.bn3 = nn.BatchNorm1d(seq_len)
            self.bn4 = nn.BatchNorm1d(hidden_size)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_size1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=2 * rnn_size1,
            hidden_size=rnn_size2,
            bidirectional=True,
            batch_first=True,
        )

        self.fc1 = nn.Linear(rnn_size2 * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_neurons)

    def forward(self, seq):
        # input shape: (batch_size, seq_len=10)
        embedded = self.embedding(seq)
        if self.batch_norm:
            embedded = self.bn1(embedded)
        embedded = self.dropout(embedded)

        # input shape: (batch_size, seq_len, embedding_dim)
        out, _ = self.lstm1(embedded)
        if self.batch_norm:
            out = self.bn2(out)

        # input shape: (batch_size, seq_len, 2*rnn_size1)
        out, _ = self.lstm2(out)
        if self.batch_norm:
            out = self.bn3(out)

        # input shape: (batch_size, seq_len, 2*rnn_size2)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size2)
        out = self.fc1(pooled)
        if self.batch_norm:
            out = self.bn4(out)
        out = self.dropout(gelu(out))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size)
        return self.fc2(out).squeeze()


class BiLSTMPadded(nn.Module):
    """
    Model architecture is based on:

    Ozols, M., Eckersley, A., Platt, C. I., Stewart-McGuinness, C.,
    Hibbert, S. A., Revote, J., ... & Sherratt, M. J. (2021).
    Predicting proteolysis in complex proteomes using deep learning.
    International Journal of Molecular Sciences, 22(6), 3071.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        rnn_size1,
        rnn_size2,
        hidden_size,
        dropout,
        out_neurons,
        pad_idx,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx
        )

        self.dropout = nn.Dropout(dropout)

        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_size1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=2 * rnn_size1,
            hidden_size=rnn_size2,
            bidirectional=True,
            batch_first=True,
        )

        self.fc1 = nn.Linear(rnn_size2 * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_neurons)

    def forward(self, seq, lengths):
        # input shape: (batch_size, seq_len=10)
        embedded = self.dropout(self.embedding(seq))

        packed_embeddings = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        # input shape: (batch_size, seq_len, embedding_dim)
        out, _ = self.lstm1(packed_embeddings)

        # input shape: (batch_size, seq_len, 2*rnn_size1)
        out, _ = self.lstm2(out)

        unpacked_output, _ = pad_packed_sequence(
            out, batch_first=True, padding_value=self.pad_idx
        )

        # input shape: (batch_size, seq_len, 2*rnn_size2)
        pooled, _ = torch.max(unpacked_output, dim=1)

        # input shape; (batch_size, 2*rnn_size2)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size)
        return self.fc2(out).squeeze()


class SeqEncoder(nn.Module):
    """
    Architecture based on:

    Heigold, G., Neumann, G., & van Genabith, J. (2016).
    Neural morphological tagging from characters for morphologically rich languages.
    arXiv preprint arXiv:1606.06640.
    """

    def __init__(self, vocab_size, embedding_dim, rnn_size, dropout):
        super().__init__()

        self.fw_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        self.bw_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.fw_lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=rnn_size, batch_first=True
        )

        self.bw_lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=rnn_size, batch_first=True
        )

    def forward(self, bw_seq, fw_seq):
        # input shape: (batch_size, seq_len=10)
        fw_embeddings = self.dropout(self.fw_embedding(fw_seq))
        bw_embeddings = self.dropout(self.bw_embedding(bw_seq))

        # input shape: (batch_size, seq_len, embedding_dim)
        fw_out, _ = self.fw_lstm(fw_embeddings)
        bw_out, _ = self.bw_lstm(bw_embeddings)

        # input shape: (batch_size, seq_len, rnn_size)
        # only get representation at last t
        fw_out = self.dropout(fw_out[:, -1, :])
        bw_out = self.dropout(bw_out[:, -1, :])

        # input shape: (batch_size, rnn_size)
        # out shape: (batch_size, 2*rnn_size)
        return torch.cat([fw_out, bw_out], dim=1)


class FwBwBiLSTM(nn.Module):
    """
    Model architecture is based on:

    Ozols, M., Eckersley, A., Platt, C. I., Stewart-McGuinness, C.,
    Hibbert, S. A., Revote, J., ... & Sherratt, M. J. (2021).
    Predicting proteolysis in complex proteomes using deep learning.
    International Journal of Molecular Sciences, 22(6), 3071.

    Heigold, G., Neumann, G., & van Genabith, J. (2016).
    Neural morphological tagging from characters for morphologically rich languages.
    arXiv preprint arXiv:1606.06640.
    """

    def __init__(
        self,
        vocab_size,
        seq_enc_emb_dim,
        seq_enc_rnn_size,
        rnn_size1,
        rnn_size2,
        hidden_size,
        out_neurons,
        dropout,
    ):
        super().__init__()

        # sequence encoder replaces embedding representations
        self.seq_encoder = SeqEncoder(
            vocab_size=vocab_size,
            embedding_dim=seq_enc_emb_dim,
            rnn_size=seq_enc_rnn_size,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self.lstm1 = nn.LSTM(
            input_size=seq_enc_rnn_size * 2,
            hidden_size=rnn_size1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=2 * rnn_size1,
            hidden_size=rnn_size2,
            bidirectional=True,
            batch_first=True,
        )

        self.fc1 = nn.Linear(rnn_size2 * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_neurons)

    def forward(self, bw_seq, fw_seq):
        # input shape: (batch_size, seq_len=10)
        embedded = self.dropout(self.seq_encoder(bw_seq, fw_seq))

        # input shape: (batch_size, seq_enc_rnn_size * 2)
        out, _ = self.lstm1(embedded)

        # input shape: (batch_size, 2*rnn_size)
        out, _ = self.lstm2(out)

        # input shape; (batch_size, 2*rnn_size)
        out = self.dropout(gelu(self.fc1(out)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size)
        return self.fc2(out).squeeze()


class BiLSTMAttention(nn.Module):
    """
    Model architecture based on:

    Liu, J., & Gong, X. (2019).
    Attention mechanism enhanced LSTM with residual architecture and its
    application for protein-protein interaction residue pairs prediction.
    BMC bioinformatics, 20, 1-11.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        rnn_size,
        hidden_size,
        num_heads,
        dropout,
        out_neurons,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_size,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=2 * rnn_size,
            hidden_size=rnn_size,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm3 = nn.LSTM(
            input_size=2 * rnn_size,
            hidden_size=rnn_size,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm4 = nn.LSTM(
            input_size=2 * rnn_size,
            hidden_size=rnn_size,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm5 = nn.LSTM(
            input_size=2 * rnn_size,
            hidden_size=rnn_size,
            bidirectional=True,
            batch_first=True,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=rnn_size * 2, num_heads=num_heads, batch_first=True
        )

        self.fc1 = nn.Linear(rnn_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_neurons)

    def forward(self, seq):
        # input shape: (batch_size, seq_len=10)
        embedded = self.dropout(self.embedding(seq))

        # input shape: (batch_size, seq_len, embedding_dim)
        out1, (hn1, cn1) = self.lstm1(embedded)
        out2, (hn2, cn2) = self.lstm2(out1, (hn1, cn1))
        out3, (hn3, cn3) = self.lstm3(out2, (hn2, cn2))
        out3, hn3, cn3 = [
            torch.add(i, j) for i, j in zip([out1, hn1, cn1], [out3, hn3, cn3])
        ]

        out4, (hn4, cn4) = self.lstm4(out3, (hn3, cn3))
        out5, _ = self.lstm5(out4, (hn4, cn4))
        out5 = torch.add(out3, out5)

        out, _ = self.attention(out5, out1, out5)

        # input shape: (batch_size, seq_len, 2*rnn_size5)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size5)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # shape: (batch_size)
        return self.fc2(out).squeeze()


class BiLSTMProt2Vec(nn.Module):
    """
    Model architecture based on:

    Ozols, M., Eckersley, A., Platt, C. I., Stewart-McGuinness, C.,
    Hibbert, S. A., Revote, J., ... & Sherratt, M. J. (2021).
    Predicting proteolysis in complex proteomes using deep learning.
    International Journal of Molecular Sciences, 22(6), 3071.

    Embeddings based on:

    Asgari, E., & Mofrad, M. R. (2015).
    Continuous distributed representation of biological sequences for
    deep proteomics and genomics.
    PloS one, 10(11), e0141287.
    """

    def __init__(self, pretrained_embeds, rnn_size, hidden_size, dropout, out_neurons):
        super().__init__()

        embeding_dim = pretrained_embeds.shape[1]

        self.embedding = nn.Embedding.from_pretrained(
            embeddings=pretrained_embeds, freeze=True
        )

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embeding_dim,
            hidden_size=rnn_size,
            bidirectional=True,
            batch_first=True,
        )

        self.fc1 = nn.Linear(rnn_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_neurons)

    def forward(self, seq):
        # input shape: (batch_size, seq_len=10)
        embedded = self.dropout(self.embedding(seq))

        # input shape: (batch_size, seq_len, embedding_dim)
        out, _ = self.lstm(embedded)

        # input shape: (batch_size, seq_len, 2*rnn_size)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size)
        return self.fc2(out).squeeze()


class Attention(nn.Module):
    """
    PyTorch implementation ported as part of CNNAttention (see below) from
    TensorFlow implementation from:

    Li, F., Chen, J., Leier, A., Marquez-Lago, T., Liu, Q., Wang, Y., ... & Song, J. (2020).
    DeepCleave: a deep learning predictor for caspase and matrix metalloprotease
    substrates and cleavage sites.
    Bioinformatics, 36(4), 1057-1065.
    """

    def __init__(self, input_dim, hidden):
        super().__init__()

        self.W0 = nn.Parameter(nn.init.kaiming_normal_(torch.empty(input_dim, hidden)))
        self.W = nn.Parameter(nn.init.kaiming_normal_(torch.empty(hidden, 1)))
        self.b0 = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty(hidden, 1)).squeeze()
        )
        self.b = nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, 1)).squeeze(1))

    def forward(self, x):
        # input shape: (batch_size, num_filters, seq_len)
        energy = (
            x.permute(0, 2, 1) @ self.W0 + self.b0
        )  # linear activation, i.e. only identity

        # input shape: (batch_size, seq_len, hidden)
        energy = (energy @ self.W + self.b).squeeze()

        # input shape: (batch_size, seq_len)
        energy = F.softmax(energy, dim=1)

        # input shape: energy=(batch_size, seq_len)
        # output shape: (batch_size, input_dim)
        # batch-wise dot product along dims 1
        res = (energy.unsqueeze(1) @ x.permute(0, 2, 1)).squeeze(1)

        # input shape: (batch_size, input_dim)
        # output shape: (batch_size, input_dim+seq_len)
        return torch.cat([res, energy], dim=-1)


class CNNAttention(nn.Module):
    """
    PyTorch implementation ported from TensorFlow implementation from:

    Li, F., Chen, J., Leier, A., Marquez-Lago, T., Liu, Q., Wang, Y., ... & Song, J. (2020).
    DeepCleave: a deep learning predictor for caspase and matrix metalloprotease
    substrates and cleavage sites.
    Bioinformatics, 36(4), 1057-1065.
    """

    def __init__(
        self,
        seq_len,
        num_filters1,
        kernel_size1,
        num_filters2,
        kernel_size2a,
        kernel_size2b,
        kernel_size2c,
        num_filters3,
        kernel_size3a,
        kernel_size3b,
        kernel_size3c,
        attention_hidden1,
        attention_hidden2,
        hidden_size1,
        hidden_size2,
        dropout,
        out_neurons,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=num_filters1,
            kernel_size=kernel_size1,
            padding="same",
        )

        self.conv2a = nn.Conv1d(
            in_channels=num_filters1,
            out_channels=num_filters2,
            kernel_size=kernel_size2a,
            padding="same",
        )

        self.conv2b = nn.Conv1d(
            in_channels=num_filters1,
            out_channels=num_filters2,
            kernel_size=kernel_size2b,
            padding="same",
        )

        self.conv2c = nn.Conv1d(
            in_channels=num_filters1,
            out_channels=num_filters2,
            kernel_size=kernel_size2c,
            padding="same",
        )

        self.conv3a = nn.Conv1d(
            in_channels=3 * num_filters2,
            out_channels=num_filters3,
            kernel_size=kernel_size3a,
            padding="same",
        )

        self.conv3b = nn.Conv1d(
            in_channels=3 * num_filters2,
            out_channels=num_filters3,
            kernel_size=kernel_size3b,
            padding="same",
        )

        self.conv3c = nn.Conv1d(
            in_channels=3 * num_filters2,
            out_channels=num_filters3,
            kernel_size=kernel_size3c,
            padding="same",
        )

        self.attention1 = Attention(input_dim=num_filters3, hidden=attention_hidden1)
        self.attention1r = Attention(input_dim=seq_len, hidden=attention_hidden2)

        self.attention2 = Attention(input_dim=num_filters3, hidden=attention_hidden1)
        self.attention2r = Attention(input_dim=seq_len, hidden=attention_hidden2)

        self.attention3 = Attention(input_dim=num_filters3, hidden=attention_hidden1)
        self.attention3r = Attention(input_dim=seq_len, hidden=attention_hidden2)

        self.fc1 = nn.Linear(3 * num_filters3 + 3 * seq_len, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, out_neurons)

    def forward(self, seq):
        ### encoder ###

        # in: (batch_size, seq_len), out: (batch_size, num_filters1, seq_len)
        x1 = F.leaky_relu(
            self.dropout(self.conv1(seq.unsqueeze(1)))
        )  # add dim=1 for conv

        # in: (batch_size, num_filters1, seq_len)
        # out: (batch_size, num_filters2, seq_len)
        y1 = F.leaky_relu(self.dropout(self.conv2a(x1)))
        y2 = F.leaky_relu(self.dropout(self.conv2b(x1)))
        y3 = F.leaky_relu(self.dropout(self.conv2c(x1)))
        # cat_y out: (batch_size, 3*num_filters2, seq_len)
        cat_y = self.dropout(torch.cat([y1, y2, y3], dim=1))

        # in: (batch_size, 3*num_filters2, seq_len)
        # out: (batch_size, num_filters3, seq_len)
        z1 = self.dropout(F.leaky_relu(self.conv3a(cat_y)))
        z2 = self.dropout(F.leaky_relu(self.conv3b(cat_y)))
        z3 = self.dropout(F.leaky_relu(self.conv3c(cat_y)))

        ### decoder ###

        # in: (batch_size, num_filters3, seq_len)
        # out w/out r, e.g. decoded_z1: (batch_size, num_filters3)
        # out w/ r: e.g. decoded_z1r: (batch_size, seq_len)
        decoded_z1 = self.attention1(z1)[:, : z1.shape[1]]
        decoded_z1r = self.attention1r(z1.permute(0, 2, 1))[:, : z1.shape[2]]

        decoded_z2 = self.attention2(z2)[:, : z2.shape[1]]
        decoded_z2r = self.attention2r(z2.permute(0, 2, 1))[:, : z2.shape[2]]

        decoded_z3 = self.attention3(z3)[:, : z3.shape[1]]
        decoded_z3r = self.attention3r(z3.permute(0, 2, 1))[:, : z3.shape[2]]

        # out: (batch_size, 3*num_filters3 + 3*seq_len)
        cat = self.dropout(
            torch.cat(
                [
                    decoded_z1,
                    decoded_z1r,
                    decoded_z2,
                    decoded_z2r,
                    decoded_z3,
                    decoded_z3r,
                ],
                dim=-1,
            )
        )

        # in: (batch_size, 3*num_filters3 + 3*seq_len), out: (batch_size, hidden_size1)
        out = self.dropout(F.leaky_relu(self.fc1(cat)))
        # in: (batch_size, hidden_size1), out: (batch_size, hidden_size2)
        out = self.dropout(F.leaky_relu(self.fc2(out)))
        # in: (batch_size, hidden_size2), out: (batch_size)
        return self.fc3(out).squeeze()


class MLP(nn.Module):
    """
    Architecture based on:

    Liu, Z. X., Yu, K., Dong, J., Zhao, L., Liu, Z., Zhang, Q., ... & Cheng, H. (2019).
    Precise prediction of calpain cleavage sites and their aberrance
    caused by mutations in cancer.
    Frontiers in Genetics, 10, 715.

    Yang, C., Li, C., Nip, K. M., Warren, R. L., & Birol, I. (2019).
    Terminitor: Cleavage site prediction using deep learning models.
    bioRxiv, 710699.
    """

    def __init__(self, vocab_size, seq_len, hidden_size, dropout, out_neurons):
        super().__init__()

        self.fc1 = nn.Linear(vocab_size * seq_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_neurons)

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        # input shape: (batch_size, seq_len*vocab_size)
        out = self.dropout(F.relu(self.fc1(seq)))
        return self.fc2(out).squeeze()


class ESM2BiLSTM(nn.Module):
    """
    Model architecture is based on:

    Ozols, M., Eckersley, A., Platt, C. I., Stewart-McGuinness, C.,
    Hibbert, S. A., Revote, J., ... & Sherratt, M. J. (2021).
    Predicting proteolysis in complex proteomes using deep learning.
    International Journal of Molecular Sciences, 22(6), 3071.

    Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., ... & Rives, A. (2023).
    Evolutionary-scale prediction of atomic-level protein
    structure with a language model.
    Science, 379(6637), 1123-1130.
    """

    def __init__(self, esm2, rnn_size, hidden_size, dropout, out_neurons):
        super().__init__()

        self.esm2_encoder = esm2

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=640,
            hidden_size=rnn_size,
            bidirectional=True,
            batch_first=True,
        )

        self.fc1 = nn.Linear(rnn_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_neurons)

    def forward(self, seq):
        with torch.no_grad():
            # input shape: (batch_size, seq_len=10+2(cls, eos))
            # out: (batch_size, seq_len, embedding_dim=1280)
            result = self.esm2_encoder(seq, repr_layers=[30])

        embedded = self.dropout(result["representations"][30][:, 1 : 10 + 1, :])

        # input shape: (batch_size, seq_len, embedding_dim)
        out, _ = self.lstm(embedded)

        # input shape: (batch_size, seq_len=1, 2*rnn_size)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size)
        return self.fc2(out).squeeze()


class ESM2(nn.Module):
    """
    Model taken from:

    Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., ... & Rives, A. (2023).
    Evolutionary-scale prediction of atomic-level protein
    structure with a language model.
    Science, 379(6637), 1123-1130.
    """

    def __init__(self, pretrained_model, dropout, out_neurons):
        super().__init__()

        self.esm2 = pretrained_model

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(33, out_neurons)

    def forward(self, seq):
        # input shape: (batch_size, seq_len=10+2 (cls, eos))
        result = self.esm2(seq)["logits"][
            :, 1 : 10 + 1, :
        ]  # remove cls, eos token position
        result = self.dropout(result)

        # in: (batch_size, seq_len, vocab_size=33)
        result, _ = result.max(dim=1)

        # input shape: (batch_size, 33)
        # out shape: (batch_size)
        return self.fc(result).squeeze()


class T5BiLSTM(nn.Module):
    """
    Model architecture based on:

    Ozols, M., Eckersley, A., Platt, C. I., Stewart-McGuinness, C.,
    Hibbert, S. A., Revote, J., ... & Sherratt, M. J. (2021).
    Predicting proteolysis in complex proteomes using deep learning.
    International Journal of Molecular Sciences, 22(6), 3071.

    Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Wang, Y., Jones,
    L., ... & Rost, B. (2021).
    Prottrans: Toward understanding the language of life through self-supervised learning.
    IEEE transactions on pattern analysis and machine intelligence, 44(10), 7112-7127.
    """

    def __init__(self, rnn_size, hidden_size, dropout, out_neurons):
        super().__init__()

        self.t5_encoder = T5EncoderModel.from_pretrained(
            "Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16
        )

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=self.t5_encoder.config.to_dict()["d_model"],  # type: ignore
            hidden_size=rnn_size,
            bidirectional=True,
            batch_first=True,
        )

        self.fc1 = nn.Linear(rnn_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_neurons)

    def forward(self, seq, att):
        with torch.no_grad():
            # input shape: (batch_size, seq_len=10)
            # out: (batch_size, seq_len+1, embedding_dim=1024)
            embedded = self.dropout(self.t5_encoder(seq, att).last_hidden_state)  # type: ignore

        # input shape: (batch_size, seq_len+1, embedding_dim)
        out, _ = self.lstm(embedded)
        out = out[:, :-1, :]

        # input shape: (batch_size, seq_len=1, 2*rnn_size)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size)
        return self.fc2(out).squeeze()


class SeqNet(nn.Module):
    """
    Architecture based on:

    Weeder, B. R., Wood, M. A., Li, E., Nellore, A., & Thompson, R. F. (2021).
    pepsickle rapidly and accurately predicts proteasomal cleavage sites for
    improved neoantigen identification.
    Bioinformatics, 37(21), 3723-3733.
    """

    def __init__(self, hidden_size1, hidden_size2, hidden_size3, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # input to linear: seq_len * 20
        self.fc1 = nn.Linear(200, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 1)

        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn3 = nn.BatchNorm1d(hidden_size3)

    def forward(self, seq):
        out = self.dropout(F.relu(self.bn1(self.fc1(seq))))
        out = self.dropout(F.relu(self.bn2(self.fc2(out))))
        out = self.dropout(F.relu(self.bn3(self.fc3(out))))
        return self.fc4(out).squeeze()


class MotifNet(nn.Module):
    """
    Architecture based on:

    Weeder, B. R., Wood, M. A., Li, E., Nellore, A., & Thompson, R. F. (2021).
    pepsickle rapidly and accurately predicts proteasomal cleavage sites for
    improved neoantigen identification.
    Bioinformatics, 37(21), 3723-3733.
    """

    def __init__(self, hidden_size1, hidden_size2, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # conv parameters are fixed due to feature assemply process
        # see dictionary variable _features
        self.conv = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, groups=4)

        # input to linear: groups * (seq_len-2)
        self.fc1 = nn.Linear(32, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)

    def forward(self, seq):
        out = self.conv(seq.transpose(1, 2))

        out = self.dropout(F.relu(self.bn1(self.fc1(out.view(out.shape[0], -1)))))
        out = self.dropout(F.relu(self.bn2(self.fc2(out))))
        return self.fc3(out).squeeze()


#############################################################################
# *** Models with DivideMix adjustments in forward pass *** #
#     All below forward-pass adjustments are based on:
#
#     Li, J., Socher, R., & Hoi, S. C. (2020).
#     Dividemix: Learning with noisy labels as semi-supervised learning.
#     arXiv preprint arXiv:2002.07394.
#
#     Guo, H., Mao, Y., & Zhang, R. (2019).
#     Augmenting data with mixup for sentence classification: An empirical study.
#     arXiv preprint arXiv:1905.08941.
#############################################################################


class BiLSTMDivideMix(BiLSTM):
    def forward_no_embed(self, embedded):
        # input shape: (batch_size, seq_len, embedding_dim)
        out, _ = self.lstm1(embedded)

        # input shape: (batch_size, seq_len, 2*rnn_size1)
        out, _ = self.lstm2(out)

        # input shape: (batch_size, seq_len, 2*rnn_size2)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size2)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size, 2)
        return self.fc2(out)

    def forward(self, seq, seq2=None, lam=None, interpolate=False):
        if interpolate:
            assert (
                seq2 is not None and lam is not None
            ), "seq2 and lam have to be defined"

            # input shape: (batch_size, seq_len=10)
            embedded1 = self.embedding(seq)
            embedded2 = self.embedding(seq2)
            embedded_mixed = lam * embedded1 + (1 - lam) * embedded2
            return self.forward_no_embed(embedded_mixed)
        else:
            # input shape: (batch_size, seq_len=10)
            embedded = self.dropout(self.embedding(seq))
            return self.forward_no_embed(embedded)


class BiLSTMPaddedDivideMix(BiLSTMPadded):
    def forward_no_embed(self, embedded, lengths):
        packed_embeddings = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        # input shape: (batch_size, seq_len, embedding_dim)
        out, _ = self.lstm1(packed_embeddings)

        # input shape: (batch_size, seq_len, 2*rnn_size1)
        out, _ = self.lstm2(out)

        unpacked_output, _ = pad_packed_sequence(
            out, batch_first=True, padding_value=self.pad_idx
        )

        # input shape: (batch_size, seq_len, 2*rnn_size2)
        pooled, _ = torch.max(unpacked_output, dim=1)

        # input shape: (batch_size, 2*rnn_size2)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size, 2)
        return self.fc2(out)

    def forward(
        self, seq, lengths, seq2=None, lengths2=None, lam=None, interpolate=False
    ):
        if interpolate:
            assert seq2 is not None, "seq2 needs to be defined"
            assert lengths2 is not None, "lengths2 needs to be defined"
            assert lam is not None, "lam needs to be defined"

            # input shape: (batch_size, seq_len=10)
            embedded1 = self.embedding(seq)
            embedded2 = self.embedding(seq2)
            embedded_mixed = lam * embedded1 + (1 - lam) * embedded2
            return self.forward_no_embed(embedded_mixed, lengths)
        else:
            # input shape: (batch_size, seq_len=10)
            embedded = self.dropout(self.embedding(seq))
            return self.forward_no_embed(embedded, lengths)


class FwBwBiLSTMDivideMix(FwBwBiLSTM):
    def forward_no_embed(self, embedded):
        # input shape: (batch_size, seq_len, embedding_dim)
        out, _ = self.lstm1(embedded)

        # input shape: (batch_size, seq_len, 2*rnn_size1)
        out, _ = self.lstm2(out)

        # input shape: (batch_size, seq_len, 2*rnn_size2)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size2)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size, 2)
        return self.fc2(out)

    def forward(
        self, bw_seq, fw_seq, bw_seq2=None, fw_seq2=None, lam=None, interpolate=False
    ):
        if interpolate:
            assert bw_seq2 is not None, "bw_seq2 needs to be defined"
            assert fw_seq2 is not None, "fw_seq2 needs to be defined"
            assert lam is not None, "lam needs to be defined"

            # input shape: (batch_size, seq_len=10)
            embedded1 = self.seq_encoder(bw_seq, fw_seq)
            embedded2 = self.seq_encoder(bw_seq2, fw_seq2)
            embedded_mixed = lam * embedded1 + (1 - lam) * embedded2
            return self.forward_no_embed(embedded_mixed)
        else:
            # input shape: (batch_size, seq_len=10)
            embedded = self.dropout(self.seq_encoder(bw_seq, fw_seq))
            return self.forward_no_embed(embedded)


class BiLSTMAttentionDivideMix(BiLSTMAttention):
    def forward_no_embed(self, embedded):
        # input shape: (batch_size, seq_len, embedding_dim)
        out1, (hn1, cn1) = self.lstm1(embedded)
        out2, (hn2, cn2) = self.lstm2(out1, (hn1, cn1))
        out3, (hn3, cn3) = self.lstm3(out2, (hn2, cn2))
        out3, hn3, cn3 = [
            torch.add(i, j) for i, j in zip([out1, hn1, cn1], [out3, hn3, cn3])
        ]

        out4, (hn4, cn4) = self.lstm4(out3, (hn3, cn3))
        out5, _ = self.lstm5(out4, (hn4, cn4))
        out5 = torch.add(out3, out5)

        out, _ = self.attention(out5, out1, out5)

        # input shape: (batch_size, seq_len, 2*rnn_size5)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size5)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # shape: (batch_size, 2)
        return self.fc2(out)

    def forward(self, seq, seq2=None, lam=None, interpolate=False):
        if interpolate:
            assert (
                seq2 is not None and lam is not None
            ), "seq2 and lam have to be defined"

            # input shape: (batch_size, seq_len=10)
            embedded1 = self.embedding(seq)
            embedded2 = self.embedding(seq2)
            embedded_mixed = lam * embedded1 + (1 - lam) * embedded2
            return self.forward_no_embed(embedded_mixed)
        else:
            # input shape: (batch_size, seq_len=10)
            embedded = self.dropout(self.embedding(seq))
            return self.forward_no_embed(embedded)


class BiLSTMProt2VecDivideMix(BiLSTMProt2Vec):
    def forward_no_embed(self, embedded):
        # input shape: (batch_size, seq_len, embedding_dim)
        out, _ = self.lstm(embedded)

        # input shape: (batch_size, seq_len, 2*rnn_size)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size2)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size, 2)
        return self.fc2(out)

    def forward(self, seq, seq2=None, lam=None, interpolate=False):
        if interpolate:
            assert (
                seq2 is not None and lam is not None
            ), "seq2 and lam have to be defined"

            # input shape: (batch_size, seq_len=10)
            embedded1 = self.embedding(seq)
            embedded2 = self.embedding(seq2)
            embedded_mixed = lam * embedded1 + (1 - lam) * embedded2
            return self.forward_no_embed(embedded_mixed)
        else:
            # input shape: (batch_size, seq_len=10)
            embedded = self.dropout(self.embedding(seq))
            return self.forward_no_embed(embedded)


class CNNAttentionDivideMix(CNNAttention):
    def forward_no_embed(self, embedded):
        # in: (batch_size, num_filters1, seq_len)
        # out: (batch_size, num_filters2, seq_len)
        y1 = F.leaky_relu(self.dropout(self.conv2a(embedded)))
        y2 = F.leaky_relu(self.dropout(self.conv2b(embedded)))
        y3 = F.leaky_relu(self.dropout(self.conv2c(embedded)))
        # cat_y out: (batch_size, 3*num_filters2, seq_len)
        cat_y = self.dropout(torch.cat([y1, y2, y3], dim=1))

        # in: (batch_size, 3*num_filters2, seq_len)
        # out: (batch_size, num_filters3, seq_len)
        z1 = self.dropout(F.leaky_relu(self.conv3a(cat_y)))
        z2 = self.dropout(F.leaky_relu(self.conv3b(cat_y)))
        z3 = self.dropout(F.leaky_relu(self.conv3c(cat_y)))

        ### decoder ###

        # in: (batch_size, num_filters3, seq_len)
        # out w/out r, e.g. decoded_z1: (batch_size, num_filters3)
        # out w/ r: e.g. decoded_z1r: (batch_size, seq_len)
        decoded_z1 = self.attention1(z1)[:, : z1.shape[1]]
        decoded_z1r = self.attention1r(z1.permute(0, 2, 1))[:, : z1.shape[2]]

        decoded_z2 = self.attention2(z2)[:, : z2.shape[1]]
        decoded_z2r = self.attention2r(z2.permute(0, 2, 1))[:, : z2.shape[2]]

        decoded_z3 = self.attention3(z3)[:, : z3.shape[1]]
        decoded_z3r = self.attention3r(z3.permute(0, 2, 1))[:, : z3.shape[2]]

        # out: (batch_size, 3*num_filters3 + 3*seq_len)
        cat = self.dropout(
            torch.cat(
                [
                    decoded_z1,
                    decoded_z1r,
                    decoded_z2,
                    decoded_z2r,
                    decoded_z3,
                    decoded_z3r,
                ],
                dim=-1,
            )
        )

        # in: (batch_size, 3*num_filters3 + 3*seq_len), out: (batch_size, hidden_size1)
        out = self.dropout(F.leaky_relu(self.fc1(cat)))
        # in: (batch_size, hidden_size1), out: (batch_size, hidden_size2)
        out = self.dropout(F.leaky_relu(self.fc2(out)))
        # in: (batch_size, hidden_size2), out: (batch_size, 2)
        return self.fc3(out)

    def forward(self, seq, seq2=None, lam=None, interpolate=False):
        if interpolate:
            assert (
                seq2 is not None and lam is not None
            ), "seq2 and lam have to be defined"

            # input shape: (batch_size, seq_len=10)
            embedded1 = self.conv1(seq.unsqueeze(1))
            embedded2 = self.conv1(seq2.unsqueeze(1))
            embedded_mixed = lam * embedded1 + (1 - lam) * embedded2
            return self.forward_no_embed(F.leaky_relu(embedded_mixed))
        else:
            # input shape: (batch_size, seq_len=10)
            embedded = F.leaky_relu(self.dropout(self.conv1(seq.unsqueeze(1))))
            return self.forward_no_embed(embedded)


class MLPDivideMix(MLP):
    def forward_no_embed(self, embedded):
        return self.fc2(embedded)

    def forward(self, seq, seq2=None, lam=None, interpolate=False):
        if interpolate:
            assert (
                seq2 is not None and lam is not None
            ), "seq2 and lam have to be defined"

            # input shape: (batch_size, seq_len*vocab_size)
            embedded1 = self.fc1(seq)
            embedded2 = self.fc1(seq2)
            embedded_mixed = lam * embedded1 + (1 - lam) * embedded2
            return self.forward_no_embed(F.relu(embedded_mixed))
        else:
            # input shape: (batch_size, seq_len=10)
            embedded = self.dropout(F.relu(self.fc1(seq)))
            return self.forward_no_embed(embedded)


class ESM2BiLSTMDivideMix(ESM2BiLSTM):
    def forward_no_embed(self, embedded):
        # input shape: (batch_size, seq_len, embedding_dim)
        out, _ = self.lstm(embedded)

        # input shape: (batch_size, seq_len=1, 2*rnn_size)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size, 2)
        return self.fc2(out)

    def forward(self, seq, seq2=None, lam=None, interpolate=False):
        if interpolate:
            assert (
                seq2 is not None and lam is not None
            ), "seq2 and lam have to be defined"

            # input shape: (batch_size, seq_len=10)
            with torch.no_grad():
                result1 = self.esm2_encoder(seq, repr_layers=[30])
                result2 = self.esm2_encoder(seq2, repr_layers=[30])
                embedded1 = result1["representations"][30][:, 1 : 10 + 1, :]
                embedded2 = result2["representations"][30][:, 1 : 10 + 1, :]
                embedded_mixed = lam * embedded1 + (1 - lam) * embedded2
            return self.forward_no_embed(embedded_mixed)
        else:
            # input shape: (batch_size, seq_len=10)
            with torch.no_grad():
                result = self.esm2_encoder(seq, repr_layers=[30])
                embedded = self.dropout(result["representations"][30][:, 1 : 10 + 1, :])
            return self.forward_no_embed(embedded)


class ESM2DivideMix(ESM2):
    def forward_no_embed(self, embedded):
        # in: (batch_size, seq_len, vocab_size=33)
        out, _ = embedded.max(dim=1)

        # input shape: (batch_size, 33)
        # out shape: (batch_size, 2)
        return self.fc(out)

    def forward(self, seq, seq2=None, lam=None, interpolate=False):
        if interpolate:
            assert (
                seq2 is not None and lam is not None
            ), "seq2 and lam have to be defined"

            # input shape: (batch_size, seq_len=10)
            result1 = self.esm2(seq)
            result2 = self.esm2(seq2)
            embedded1 = result1["logits"][:, 1 : 10 + 1, :]
            embedded2 = result2["logits"][:, 1 : 10 + 1, :]
            embedded_mixed = lam * embedded1 + (1 - lam) * embedded2
            return self.forward_no_embed(embedded_mixed)
        else:
            # input shape: (batch_size, seq_len=10)
            result = self.esm2(seq)
            embedded = self.dropout(result["logits"][:, 1 : 10 + 1, :])
            return self.forward_no_embed(embedded)


class T5BiLSTMDivideMix(T5BiLSTM):
    def forward_no_embed(self, embedded):
        # input shape: (batch_size, seq_len+1, embedding_dim)
        out, _ = self.lstm(embedded)
        out = out[:, :-1, :]

        # input shape: (batch_size, seq_len=1, 2*rnn_size)
        pooled, _ = torch.max(out, dim=1)

        # input shape: (batch_size, 2*rnn_size)
        out = self.dropout(gelu(self.fc1(pooled)))

        # input shape: (batch_size, hidden_size)
        # output shape: (batch_size, 2)
        return self.fc2(out)

    def forward(self, seq, att, seq2=None, att2=None, lam=None, interpolate=False):
        if interpolate:
            assert seq2 is not None, "seq2 needs to be defined"
            assert att2 is not None, "att2 needs to be defined"
            assert lam is not None, "lam needs to be defined"

            # input shape: (batch_size, seq_len=10)
            with torch.no_grad():
                embedded1 = self.t5_encoder(seq, att).last_hidden_state  # type: ignore
                embedded2 = self.t5_encoder(seq2, att2).last_hidden_state  # type: ignore
                embedded_mixed = lam * embedded1 + (1 - lam) * embedded2
            return self.forward_no_embed(embedded_mixed)
        else:
            # input shape: (batch_size, seq_len=10)
            with torch.no_grad():
                embedded = self.dropout(self.t5_encoder(seq, att).last_hidden_state)  # type: ignore
            return self.forward_no_embed(embedded)
