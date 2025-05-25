import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torch import nn, optim, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from torchvision import transforms

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torch import nn, optim, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from torchvision import transforms

from typing import List

import math
import matplotlib.pyplot as plt

from torchvision.models import densenet121, DenseNet121_Weights
import collections
import re

from typing import Dict, List, Tuple, Union


class LaTeXTokenizer:
    def __init__(self):
        self.special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text: str) -> List[str]:
        # Tokenize LaTeX using regex to capture commands, numbers and other characters
        return re.findall(r"\\[a-zA-Z]+|\\.|[a-zA-Z0-9]|\S", text)

    def build_vocab(self, texts: List[str]):
        # Add special tokens to vocabulary
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)

        # Create a counter to hold token frequencies
        counter = collections.Counter()

        # Tokenize each text and update the counter
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Add tokens to vocab based on their frequency
        for token, _ in counter.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Build dictionaries for token to ID and ID to token conversion
        self.token_to_id = self.vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        # Tokenize the input text and add start and end tokens
        tokens = ["[BOS]"] + self.tokenize(text) + ["[EOS]"]

        # Map tokens to their IDs, using [UNK] for unknown tokens
        unk_id = self.token_to_id["[UNK]"]
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def decode(self, token_ids: List[int]) -> List[str]:
        # Map token IDs back to tokens
        tokens = [self.id_to_token.get(id, "[UNK]") for id in token_ids]

        # Remove tokens beyond the [EOS] token
        if "[EOS]" in tokens:
            tokens = tokens[: tokens.index("[EOS]")]

        # Replace [UNK] with ?
        tokens = ["?" if token == "[UNK]" else token for token in tokens]

        # Reconstruct the original text, ignoring special tokens
        return "".join([token for token in tokens if token not in self.special_tokens])


class CROHMEDataset(data.Dataset):
    def __init__(self, files, transform):
        super().__init__()
        self.files = list(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_file = self.files[idx]

        x = self.transform(Image.open(image_file))
        return x


class CROHMEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((384, 512))
            ]
        )

    def setup(self):
        self.test_dataset = CROHMEDataset(
            self.data_dir.glob("*.jpg"), self.transform
        )

    def collate_fn(self, batch, max_width: int = 512, max_height: int = 384):
        images = batch

        # Create a white background for each image in the batch
        src = torch.ones((len(images), 3, max_height, max_width))

        # Center and pad individual images to fit into the white background
        for i, img in enumerate(images):
            height_start = (max_height - img.size(1)) // 2
            height_end = height_start + img.size(1)
            width_start = (max_width - img.size(2)) // 2
            width_end = width_start + img.size(2)
            src[i, :, height_start:height_end, width_start:width_end] = img

        return src

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.hparams.pin_memory,
        )

class PositionalEncoding1D(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1000,
        temperature: float = 10000.0,
    ):
        super().__init__()

        # Generate position and dimension tensors for encoding
        position = torch.arange(max_len).unsqueeze(1)
        dim_t = torch.arange(0, d_model, 2)
        div_term = torch.exp(dim_t * (-math.log(temperature) / d_model))

        # Initialize and fill the positional encoding matrix with sine/cosine values
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch, sequence_length, d_model = x.shape
        return self.dropout(x + self.pe[None, :sequence_length, :])


class PositionalEncoding2D(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 30,
        temperature: float = 10000.0,
    ):
        super().__init__()

        # Generate position and dimension tensors for 1D encoding
        position = torch.arange(max_len).unsqueeze(1)
        dim_t = torch.arange(0, d_model, 2)
        div_term = torch.exp(dim_t * (-math.log(temperature) / d_model))

        # Initialize and fill the 1D positional encoding matrix with sine/cosine values
        pe_1D = torch.zeros(max_len, d_model)
        pe_1D[:, 0::2] = torch.sin(position * div_term)
        pe_1D[:, 1::2] = torch.cos(position * div_term)

        # Compute the 2D positional encoding matrix using outer product
        pe_2D = torch.zeros(max_len, max_len, d_model)
        for i in range(d_model):
            pe_2D[:, :, i] = pe_1D[:, i].unsqueeze(-1) + pe_1D[:, i].unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe_2D)

    def forward(self, x):
        batch, height, width, d_model = x.shape
        return self.dropout(x + self.pe[None, :height, :width, :])


class Permute(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Model(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = (
            torch.rand(16, 3, 384, 512),  # batch x channel x height x width
            torch.ones(16, 64, dtype=torch.long),  # batch x sequence length
            torch.zeros(64, 64),  # sequence length x sequence length
        )

        # Define the encoder architecture
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            nn.Sequential(*list(densenet.children())[:-1]),  # remove the final layer
            nn.Conv2d(1024, d_model, kernel_size=1),
            Permute(0, 2, 3, 1),
            PositionalEncoding2D(d_model, dropout),
            nn.Flatten(1, 2),
        )

        # Define the decoder architecture
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.word_positional_encoding = PositionalEncoding1D(d_model, dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            ),
            num_layers,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def greedy_search(self, src, tokenizer, max_seq_len: int = 256) -> List[str]:
        with torch.no_grad():
            batch_size = src.size(0)
            features = self.encoder(src).detach()
            tgt = torch.ones(batch_size, 1).long().to(src.device)
            tgt_mask = torch.triu(
                torch.ones(max_seq_len, max_seq_len) * float("-inf"), diagonal=1
            ).to(src.device)

            for i in range(1, max_seq_len):
                output = self.decoder(features, tgt, tgt_mask[:i, :i])
                next_probs = output[:, -1].log_softmax(dim=-1)
                next_chars = next_probs.argmax(dim=-1, keepdim=True)
                tgt = torch.cat((tgt, next_chars), dim=1)

        return [tokenizer.decode(seq.tolist()) for seq in tgt]

    def decoder(self, features, tgt, tgt_mask):
        padding_mask = tgt.eq(0)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.hparams.d_model)
        tgt = self.word_positional_encoding(tgt)
        tgt = self.transformer_decoder(
            tgt, features, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask
        )
        output = self.fc_out(tgt)
        return output

    def forward(self, src, tgt, tgt_mask):
        features = self.encoder(src)
        output = self.decoder(features, tgt, tgt_mask)
        return output

    def beam_search(
        self,
        src,
        tokenizer,
        max_seq_len: int = 256,
        beam_width: int = 3,
    ) -> List[str]:
        with torch.no_grad():
            batch_size = src.size(0)
            vocab_size = self.hparams.vocab_size
            features = self.encoder(src).detach()
            features_rep = features.repeat_interleave(beam_width, dim=0)
            tgt_mask = torch.triu(
                torch.ones(max_seq_len, max_seq_len) * float("-inf"), diagonal=1
            ).to(src.device)

            # Initialize with [BOS]
            beams = torch.ones(batch_size, 1, 1).long().to(src.device)

            # Handle first step separately
            output = self.decoder(features, beams[:, 0, :], tgt_mask[:1, :1])
            next_probs = output[:, -1, :].log_softmax(dim=-1)
            beam_scores, indices = next_probs.topk(beam_width, dim=-1)
            beams = torch.cat(
                [beams.repeat_interleave(beam_width, dim=1), indices.unsqueeze(2)],
                dim=-1,
            )

            for i in range(2, max_seq_len):
                tgt = beams.view(batch_size * beam_width, i)
                output = self.decoder(features_rep, tgt, tgt_mask[:i, :i])
                next_probs = output[:, -1, :].log_softmax(dim=-1)

                next_probs += beam_scores.view(batch_size * beam_width, 1)
                next_probs = next_probs.view(batch_size, -1)

                beam_scores, indices = next_probs.topk(beam_width, dim=-1)
                beams = torch.cat(
                    [
                        beams[
                            torch.arange(batch_size).unsqueeze(-1),
                            indices // vocab_size,
                        ],
                        (indices % vocab_size).unsqueeze(2),
                    ],
                    dim=-1,
                )

        best_beams = beams[:, 0, :]  # taking the best beam for each batch
        return [tokenizer.decode(seq.tolist()) for seq in best_beams]

