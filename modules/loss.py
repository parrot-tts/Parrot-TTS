import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelLoss(nn.Module):
    def __init__(self, data_config):
        super().__init__()

        self.num_codes = data_config["preprocess"]["hubert_codes"]
        self.code_loss = nn.CrossEntropyLoss(ignore_index=self.num_codes)
        self.dur_loss = nn.MSELoss()
        
    def forward(self, out, log_dur_preds, batch):
        log_dur_preds = log_dur_preds.masked_select(batch['src_mask'])
        log_dur_targets = torch.log(batch['duration'].float() + 1).masked_select(batch['src_mask'])

        code_loss = self.code_loss(out.reshape(-1,self.num_codes), batch['codes'].reshape(-1))
        dur_loss = self.dur_loss(log_dur_preds, log_dur_targets)

        loss = code_loss + dur_loss
        return loss, code_loss, dur_loss
    

