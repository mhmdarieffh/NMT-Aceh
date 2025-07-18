# %%
import torch
import torch.nn as nn
import math
import json
import re
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# %%
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def numericalize(sentence, vocab):
    return [vocab.get(tok, UNK_IDX) for tok in tokenize(sentence)]

def decode(indices, inv_vocab):
    tokens = [inv_vocab.get(str(idx), "<unk>") for idx in indices]
    # Stop at <eos>
    if "<eos>" in tokens:
        tokens = tokens[:tokens.index("<eos>")]
    return tokens

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# %%
class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=0.1
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def create_src_padding_mask(self, src):
        # src: [batch, src_len]
        return (src == PAD_IDX)

    def create_tgt_padding_mask(self, tgt):
        # tgt: [batch, tgt_len]
        return (tgt == PAD_IDX)

    def forward(self, src, tgt):
        # Embedding + Positional Encoding
        src_embed = self.pos_encoder(self.src_embed(src))
        tgt_embed = self.pos_encoder(self.tgt_embed(tgt))

        # Masks
        src_key_padding_mask = self.create_src_padding_mask(src)  # [batch, src_len]
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt)  # [batch, tgt_len]
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)  # [tgt_len, tgt_len]

        # Transformer expects [seq_len, batch, dim]
        memory = self.transformer.encoder(
            src_embed.permute(1, 0, 2),
            src_key_padding_mask=src_key_padding_mask
        )

        out = self.transformer.decoder(
            tgt_embed.permute(1, 0, 2),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        return self.fc_out(out.permute(1, 0, 2))

def load_model_and_vocab(model_path, src_vocab_path, tgt_vocab_path):
    checkpoint = torch.load(model_path, map_location=device)

    with open(src_vocab_path, "r", encoding="utf-8") as f:
        src_vocab = json.load(f)
    with open(tgt_vocab_path, "r", encoding="utf-8") as f:
        tgt_vocab = json.load(f)
    inv_tgt_vocab = {str(v): k for k, v in tgt_vocab.items()}

    model = TransformerNMT(len(src_vocab), len(tgt_vocab)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, src_vocab, tgt_vocab, inv_tgt_vocab


def smart_translate(sentence, model, src_vocab, tgt_vocab, inv_tgt_vocab, max_len=50):
    model.eval()
    tokens = tokenize(sentence)

    input_indices = [BOS_IDX] + [src_vocab.get(tok, UNK_IDX) for tok in tokens] + [EOS_IDX]
    src_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(next(model.parameters()).device)

    unk_check = all(src_vocab.get(tok, UNK_IDX) == UNK_IDX for tok in tokens)
    if unk_check and len(tokens) == 1:
        return tokens[0]

    memory = model.transformer.encoder(
        model.pos_encoder(model.src_embed(src_tensor)).permute(1, 0, 2)
    )

    output_indices = [BOS_IDX]
    translated_tokens = []

    if len(tokens) == 1:
        tgt_tensor = torch.tensor(output_indices, dtype=torch.long).unsqueeze(0).to(src_tensor.device)
        tgt_embed = model.pos_encoder(model.tgt_embed(tgt_tensor))
        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(tgt_tensor.device)

        with torch.no_grad():
            out = model.transformer.decoder(
                tgt_embed.permute(1, 0, 2),
                memory,
                tgt_mask=tgt_mask
            )
            out = model.fc_out(out.permute(1, 0, 2))
            next_token_idx = out[0, -1].argmax().item()

        if next_token_idx == UNK_IDX:
            return tokens[0]
        else:
            return inv_tgt_vocab.get(str(next_token_idx), "<unk>")

    for step in range(max_len):
        tgt_tensor = torch.tensor(output_indices, dtype=torch.long).unsqueeze(0).to(src_tensor.device)
        tgt_embed = model.pos_encoder(model.tgt_embed(tgt_tensor))
        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(tgt_tensor.device)

        with torch.no_grad():
            out = model.transformer.decoder(
                tgt_embed.permute(1, 0, 2),
                memory,
                tgt_mask=tgt_mask
            )
            out = model.fc_out(out.permute(1, 0, 2))
            next_token_idx = out[0, -1].argmax().item()

        if next_token_idx == EOS_IDX:
            break

        input_pos = step + 1
        if input_pos < len(input_indices) - 1:
            input_idx = input_indices[input_pos]
            if input_idx == UNK_IDX:
                output_word = tokens[step]
            else:
                output_word = inv_tgt_vocab.get(str(next_token_idx), "<unk>")
        else:
            output_word = inv_tgt_vocab.get(str(next_token_idx), "<unk>")

        translated_tokens.append(output_word)
        output_indices.append(next_token_idx)

    return " ".join(translated_tokens)
