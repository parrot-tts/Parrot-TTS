import torch
from torch.utils.data import DataLoader
from modules import ParrotDataset, Parrot
import lightning as L
import yaml
import argparse
import os
import librosa

class LitParrot(L.LightningModule):
    
    # define model architecture
    def __init__(
        self, data_config, src_vocab_size, src_pad_idx
    ):
        super().__init__()
        self.save_hyperparameters()
        self.parrot = Parrot(data_config, src_vocab_size, src_pad_idx)
    
    def infer(self, batch):
        self.eval()
        res = self.parrot.infer(batch)
        return res
    
def main(args):
    data_config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    
    audio_dir = data_config["path"]["wav_path"]
    
    # setup datasets
    val_dataset = ParrotDataset("val", data_config=data_config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=val_dataset.collate_fn,
        num_workers=4,
    )
    
    # load checkpoint
    checkpoint = args.checkpoint_pth
    
    # init the model
    model = LitParrot.load_from_checkpoint(checkpoint,weights_only=True)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
    
    # Move model to the correct device
    model = model.to(device)

    processed_lines = list()
    with torch.no_grad():  # Disable gradient calculations for inference
        for batch in val_loader:
                    
            # Move batch to the same device as the model
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

            speaker = '_'.join(batch['ids'][0].split('_')[:2])
            audio_path = os.path.join(audio_dir,speaker,"wavs",batch['ids'][0]+'.wav')
            codes = ' '.join(map(str, model.infer(batch)[0]))
            y, sr = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)

            data_dict = {}
            data_dict['audio'] = audio_path
            data_dict['hubert'] = codes
            data_dict['duration'] = duration
            processed_lines.append(data_dict)          
        
        with open(data_config["path"]["root_path"]+"/predictions.txt", 'w') as f:
            for line in processed_lines:
                f.write(str(line) + "\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint_pth", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the inference")

    args = parser.parse_args()
    L.seed_everything(42, workers=True)

    main(args)