import argparse
import torch
from torch import optim

from model import Aligner
from paths import Paths
from utils import read_config, unpickle_binary
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for Aligner.')
    parser.add_argument('--config', '-c', default='utils/aligner/aligner_train_config.yaml', help='Points to the config file.')
    parser.add_argument('--checkpoint', '-cp', default=None, help='Points to the a model file to restore.')
    args = parser.parse_args()

    config = read_config(args.config)
    paths = Paths.from_config(config['paths'])
    symbols = unpickle_binary(config['paths']['symbols_path'])
    
    # Declare the device based on the config
    device_id = config['device']['id']
    device = torch.device(device_id) if device_id.startswith('cuda') else torch.device('cpu')

    if args.checkpoint:
        print(f'Restoring model from checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model = Aligner.from_checkpoint(checkpoint)
        assert checkpoint['symbols'] == symbols, 'Symbols from data do not match symbols from model!'
        print(f'Restored model with step {model.get_step()}')
    else:
        model_path = paths.checkpoint_dir / 'latest_model.pt'
        if model_path.exists():
            print(f'Restoring model from checkpoint: {model_path}')
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model = Aligner.from_checkpoint(checkpoint)
            assert checkpoint['symbols'] == symbols, 'Symbols from data do not match symbols from model!'
            print(f'Restored model with step {model.get_step()}')
        else:
            print(f'Initializing new model from config {args.config}')
            model = Aligner(n_mels=config['audio']['n_mels'],
                            num_symbols=len(symbols)+1,
                            **config['model'])
            optim = optim.Adam(model.parameters(), lr=1e-4)
            checkpoint = {'model': model.state_dict(), 'optim': optim.state_dict(),
                          'config': config, 'symbols': symbols}

    trainer = Trainer(paths=paths, device=device)
    trainer.train(checkpoint, train_params=config['training'])

