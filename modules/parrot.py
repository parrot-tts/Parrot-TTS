import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

from modules.fft import FFTBlock, SinusoidalPosEmb
from modules.loss import ModelLoss
from modules.duration import DurationPredictor, length_regulator


class Parrot(nn.Module):
    def __init__(self, data_config, model_config, src_vocab_size, src_pad_idx):
        super().__init__()
        self.max_len = model_config["transformer"]["max_len"]
        self.d_model = model_config["transformer"]["d_model"]
        transformer_config = model_config["transformer"]
        duration_p_config = model_config["duration_predictor"]

        self.pos_emb = SinusoidalPosEmb(self.max_len, self.d_model)
        self.tok_emb = nn.Embedding(src_vocab_size, self.d_model, src_pad_idx)
        self.speaker_emb = None

        spk_path = os.path.join(data_config["path"]["root_path"], "speakers.json")
        with open(spk_path, 'r') as f:
            n_speaker = len(json.load(f))
        
        if n_speaker > 1:
            self.speaker_emb = nn.Embedding(
                n_speaker, 
                self.d_model
            )

        self.duration_predictor = DurationPredictor(
            self.d_model,
            duration_p_config["n_filter"],
            duration_p_config["kernel_size"],
            duration_p_config["dropout_p"],
        )

        self.encoder_layers = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model,
                    transformer_config["encoder"]["n_head"],
                    transformer_config["conv_n_filter"],
                    transformer_config["conv_kernel_sizes"],
                    transformer_config["encoder"]["dropout_p"]
                ) for n in range(transformer_config["encoder"]["n_layer"])
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model,
                    transformer_config["decoder"]["n_head"],
                    transformer_config["conv_n_filter"],
                    transformer_config["conv_kernel_sizes"],
                    transformer_config["decoder"]["dropout_p"]
                ) for n in range(transformer_config["decoder"]["n_layer"])
            ]
        )

        self.head = nn.Linear(self.d_model, data_config["preprocess"]["hubert_codes"])
    
    def forward_encoder(self, x, mask):
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, key_padding_mask=mask)
        return x

    def forward_decoder(self, x, mask):
        for dec_layer in self.decoder_layers:
            x = dec_layer(x, key_padding_mask=mask)
        return x
        
    def forward_duration(self, x, src_mask, tgt_mask=None, dur_target=None):
        log_dur_pred = self.duration_predictor(x, src_mask)
        if dur_target is not None:
            x, tgt_mask = length_regulator(x, dur_target, tgt_mask)
        else:
            dur_rounded = torch.clamp(
                (torch.round(torch.exp(log_dur_pred) - 1)),
                min=0,
            )
            x, tgt_mask = length_regulator(x, dur_rounded.long(), tgt_mask)
        return x, tgt_mask, log_dur_pred


    def forward(self, batch, inference=False):
        if inference != True:
            assert "duration" in batch.keys()

        out = self.tok_emb(batch["phones"])
        out = self.pos_emb(out)
         
        out = self.forward_encoder(out, ~batch['src_mask'])
        if self.speaker_emb is not None:
            out = out + self.speaker_emb(batch["speaker"]).unsqueeze(1)

        if inference:
            out, tgt_mask, log_dur_preds = self.forward_duration(out, ~batch['src_mask'])
        else:
            out, tgt_mask, log_dur_preds = self.forward_duration(out, ~batch['src_mask'], batch['tgt_mask'], batch['duration'])

        out = self.pos_emb(out)    
        out = self.forward_decoder(out, ~tgt_mask)
        out = self.head(out)

        return (out, batch['src_mask'], tgt_mask, log_dur_preds)

    def infer(self, batch):
        assert self.training == False
        out, _, tgt_mask, log_dur_preds = self.forward(batch, inference=True)
        out_codes = torch.argmax(out, dim=-1)
        res = []
        for out_code, msk in zip(out_codes, tgt_mask):
            res.append(out_code[msk].cpu().numpy().tolist())
        
        return res

        
