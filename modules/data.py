from pathlib import Path
import json 
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from utils import ArpabetTokenizer

def get_mask_from_lengths(lengths, max_len=None, device=None):
    lengths = torch.tensor(lengths)
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    if device:
        lengths = lengths.to(device)
        ids = ids.to(device)
    mask = ids <= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def get_mask_from_batch(batch, pad_idx):
    return (batch != pad_idx)

def convert_arr_to_tensor(arr, dtype):
    return torch.tensor(arr, dtype=dtype)
    

class ParrotDataset(Dataset):
    def __init__(self, split, data_config):
        assert split in ["train", "val"]
        self.root_dir = Path(data_config["path"]["root_path"])
        self.data_file = self.root_dir / f"{split}.txt"
        self.data_list = list()
        self.tokenizer = ArpabetTokenizer()
        self.src_vocab_size = len(self.tokenizer)
        self.src_pad_idx = self.tokenizer.pad_idx
        self.code_pad_idx = data_config["preprocess"]["hubert_codes"]
        
        with open(self.data_file) as f:
            data_lines = f.readlines()
            for l in data_lines:
                l = l.strip().replace("'","\"")
                aa = json.loads(l)
                self.data_list.append(aa)
        
        with open(self.root_dir / "speakers.json") as f:
            self.speaker_map = json.load(f) 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        basename = Path(data_dict['audio']).stem
        speaker_id = self.speaker_map[data_dict['speaker']]
        phones = self.tokenizer.tokenize(data_dict['phones'].split(' '))
        codes = [int(i) for i in data_dict['hubert'].split(' ')]
        durations = [int(i) for i in data_dict['duration'].split(' ')]
        
        return {
            'id': basename,
            'speaker': speaker_id,
            'phones': phones,
            'codes': codes,
            'duration': durations
        }

    def collate_fn(self, data_list):
        # pad 
        ids = [d['id'] for d in data_list]
        speaker = convert_arr_to_tensor([d['speaker'] for d in data_list], dtype=torch.long)
        phones = [convert_arr_to_tensor(d['phones'], dtype=torch.long) for d in data_list]
        codes = [convert_arr_to_tensor(d['codes'], dtype=torch.long) for d in data_list]
        duration = [convert_arr_to_tensor(d['duration'], dtype=torch.long) for d in data_list]
        
        data = {}

        data['ids'] = ids
        data['speaker'] = speaker 
        data['phones'] = torch.nn.utils.rnn.pad_sequence(phones, batch_first=True, padding_value=self.src_pad_idx)
        data['codes'] = torch.nn.utils.rnn.pad_sequence(codes, batch_first=True, padding_value=self.code_pad_idx)
        data['duration'] = torch.nn.utils.rnn.pad_sequence(duration, batch_first=True)
        data['src_mask'] = get_mask_from_batch(data['phones'], self.tokenizer.pad_idx)
        data['tgt_mask'] = get_mask_from_batch(data['codes'], self.code_pad_idx)

        return data

        
        
