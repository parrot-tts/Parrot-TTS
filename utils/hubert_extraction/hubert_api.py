import sys
sys.path.append("/media/newhd/Neil/fairseq")

import fairseq
import librosa 
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import joblib
import random
import tqdm
import gc
import numpy as np

class HubertFeatureReader:
    def __init__(self, checkpoint_path, layer, max_chunk=1600000, use_cuda=True):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = model[0].eval()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

    def read_audio(self, path, ref_len=None, channel_id=None):
        wav, sr = librosa.load(path, sr=self.task.cfg.sample_rate)
        if channel_id is not None:
            assert wav.ndim == 2, \
                f"Expected stereo input when channel_id is given ({path})"
            assert channel_id in [1, 2], \
                "channel_id is expected to be in [1, 2]"
            wav = wav[:, channel_id-1]
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
#         assert sr == self.task.cfg.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path, ref_len=None, channel_id=None):
        x = self.read_audio(file_path, ref_len, channel_id)
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            if self.use_cuda:
                x = x.cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)  
    
    def get_feats_from_wav(self, wav_file):
        with torch.no_grad():
            x = torch.from_numpy(wav_file).float()
            if self.use_cuda:
                x = x.cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)   
            
def get_feature_iterator(checkpoint_path, layer, file_path_list, sample_pct=1, channel_id=None):
    if sample_pct < 1.0:
        file_path_list = random.sample(
            file_path_list, int(sample_pct * len(file_path_list))
        )
    num_files = len(file_path_list)
    reader = HubertFeatureReader(
            checkpoint_path=checkpoint_path, layer=layer
        )

    def iterate():
        for file_path in file_path_list:
            feats = reader.get_feats(file_path, channel_id=channel_id)
            yield feats.cpu().numpy()
    
    return iterate, num_files


def main():
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser(
        description="Get Hubert Codes from Audio"
    )
    parser.add_argument("--hubert_model_path", type=str)
    parser.add_argument("--kmeans_model_path", type=str)
    parser.add_argument("--layer", default=6, type=int)

    parser.add_argument("--src_dir", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--channel_id", type=int)
    args = parser.parse_args()
    
    args.device = 'cuda'
    fnames = glob(os.path.join(args.src_dir, "*.wav"))
    generator, num_files = get_feature_iterator(
        checkpoint_path=args.hubert_model_path,
        file_path_list=fnames,
        layer=args.layer,
        channel_id=None
    )
    iterator = generator()
    
    kmeans_model = joblib.load(open(args.kmeans_model_path, 'rb'))
    kmeans_model.verbose = False
    
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    print(f"Writing predictions to {args.out_path}")

    i = 0
    with open(args.out_path, "w") as fout:
        for features in tqdm.tqdm(iterator, total=num_files):
            out_dict = {}
            pred = kmeans_model.predict(features)
            pred_str = " ".join(str(p) for p in pred)
            base_fname = Path(fnames[i]).name
            if args.channel_id is not None:
                base_fname = base_fname+f'-channel{args.channel_id}'
            out_dict['audio'] = base_fname
            out_dict['hubert'] = pred_str 
            # print(out_dict['audio'])
            fout.write(str(out_dict) + "\n")
            i+=1         
         

if __name__ == "__main__":
    main()
