from pathlib import Path


class Paths:
    
    def __init__(self, data_dir: str, checkpoint_dir: str, dataset_dir: str, precomputed_mels: str, metadata_path: str, actual_dur_path):
        self.data_dir = Path(data_dir)
        self.dataset_dir = dataset_dir
        self.metadata_path = Path(metadata_path)
        self.mel_dir = self.data_dir / 'mels'
        self.token_dir = self.data_dir / 'tokens'
        self.precomputed_mels = precomputed_mels
        self.actual_dur_path = actual_dur_path
        if self.precomputed_mels is not None:
            self.precomputed_mels = Path(precomputed_mels)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.create_dirs()
    
    def create_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mel_dir.mkdir(parents=True, exist_ok=True)
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_config(cls, config):
        return cls(
            # if specified, use specific data path instead of appending to metadata path
            data_dir=Path(config["data_dir"]) if config["data_dir"] is not None else Path(config['metadata_path']).stem+'_data',
            metadata_path=config['metadata_path'],
            checkpoint_dir=Path(config["data_dir"] + "/checkpoints") if config["data_dir"] is not None else Path(config['metadata_path']).stem+'_checkpoints',
            dataset_dir=config['dataset_dir'],
            precomputed_mels=None,
            actual_dur_path=config['dur_path'],
        )
