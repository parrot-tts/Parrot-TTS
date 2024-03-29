# api to use fairseq's trained vocoders

import fairseq
import torch
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import json
import numpy as np
import librosa

class FairseqVocoder:
    def __init__(self, model_config, device="cpu"):
        self.cfg = model_config["vocoder"] 
        assert self.cfg["fairseq"] == True
        with open(self.cfg["config_path"]) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(self.cfg["vocoder_ckpt"], vocoder_cfg)
        self.vocoder.to(device)
        self.device = device
        self.multispkr = self.vocoder.model.multispkr

    def code_str_to_int(codes):
        assert isinstance(codes, str), "Codes must be a space separated string"
        return list(map(int, codes.strip().split()))

    def infer(self, codes, speaker_id=None):
        x = {
            "code": torch.LongTensor(codes).view(1,-1),
        }
        if self.multispkr:
            assert speaker_id is not None
            spk = speaker_id
            x["spkr"] = torch.LongTensor([spk]).view(1, 1)
        
        x = fairseq.utils.move_to_cuda(x, self.device)
        with torch.no_grad():
            wav = self.vocoder(x)
        wav = wav.numpy().astype(np.float32)
        wav = librosa.util.normalize(wav)
        return wav
