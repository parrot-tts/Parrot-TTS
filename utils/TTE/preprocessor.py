from pathlib import Path
import json
import os
from tqdm import tqdm
import random
import argparse
import yaml
import numpy as np
import pickle

def parse_speaker(path, method):
    # parse the hubert code path
    if type(path) == str:
        path = Path(path)

    if method == "_":
        return '_'.join(path.name.split("_")[:2])
    elif method == "single":
        return "A"
    else:
        raise NotImplementedError()


def adjust_duration(total_codes, durations):
    """
    Adjusts the durations list so that the sum of its elements equals the provided total_codes.
    If the adjustment is not possible or the difference is greater than 2, returns None.

    Args:
        total_codes (int): The desired sum of the durations list.
        durations (list[int]): The list of durations to be adjusted.

    Returns:
        list[int] or None: The adjusted list of durations or None if adjustment is not possible.
    """
    total_duration = sum(durations)
    difference = total_duration - total_codes
    if difference == 0:
        return durations

    # If the difference is greater than 2, adjustment is not possible
    if abs(difference) > 2:
        print("Unable to adjust durations. The difference is greater than 2.")
        return None

    if difference < 0:
        durations[-1] += abs(difference)
        return durations
    
    if difference > 0:
        # Attempt to adjust the last element first
        if durations[-1] > difference:  # Ensure duration doesn't become 0 or negative
            durations[-1] -= difference
            return durations
        # If adjusting the last element is not possible, try the first element

        if durations[0] > difference:  # Ensure duration doesn't become 0 or negative
            durations[0] -= difference
            return durations

    if len(durations) >= 2:
        if difference == 2:
            if durations[0] > 1 and durations[-1] > 1:
                durations[0] -= 1
                durations[-1] -= 1
                return durations
            
    print("Unable to adjust durations by modifying the first or last element.")
    return None

class Preprocessor:
    # take as input the list of hubert dictionaries and extract durations
    # we write two files: train.txt and val.txt
    # each file is a similar list of dictionaries with keys as follows:
    # audio: path to original wav
    # hubert: space separated string of hubert tokens
    # speaker: speaker id
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(config["path"]["root_path"])
        self.hubert_path = config["path"]["hubert_path"]
        self.speaker_method = config["preprocess"]["speaker"]
        self.val_size = config["preprocess"]["val_size"]
        
        self.alignment_dir = Path(config["path"]["alignment_path"])

    def build_from_path(self):
        print("Processing data...")

        with open(self.hubert_path) as f:
            hubert_lines = f.readlines()
        speaker_set = set()
        random.shuffle(hubert_lines)
        processed_lines = list()
        skipped_lines = 0
        
        # load all characters found in the dataset
        with open(self.alignment_dir / "symbols.pkl", "rb") as f: 
            symbols = pickle.load(f)
                
        for l in tqdm(hubert_lines):
            
            # load dict containing audio path, hubert units and total duration
            data_dict = json.loads(l.strip().replace("'", '"'))
            
            # get basename
            basename = Path(data_dict["audio"]).stem
            
            # get speaker
            speaker = parse_speaker(data_dict["audio"], method=self.speaker_method)
            speaker_set.add(speaker)
            data_dict['speaker'] = speaker

            # get tokens obtained from DFA training
            if not os.path.exists(self.alignment_dir / "{}/tokens/{}.npy".format(speaker,basename)):
                continue
            tokens = np.load(self.alignment_dir / "{}/tokens/{}.npy".format(speaker,basename))
        
            # get characters from tokens. replace ' ' with 'sil' character 
            characters = ['sil' if symbols[i-1] == ' ' else symbols[i-1] for i in tokens]
                
            # get durations obtained from DFA training
            if not os.path.exists(self.alignment_dir / "{}/outputs/durations/{}.npy".format(speaker,basename)):
                continue
            durations = np.load(self.alignment_dir / "{}/outputs/durations/{}.npy".format(speaker,basename))
            
            # adjust durations with HuBERT units
            durations = adjust_duration(len(data_dict['hubert'].split()), durations)
            if durations is None:
                skipped_lines+=1
                continue
    
            assert sum(durations) == len(data_dict['hubert'].split()), f"{sum(durations)}, {len(data_dict['hubert'].split())}"
            data_dict['characters'] = " ".join(characters) 
            data_dict['duration'] = " ".join([str(i) for i in durations])
            
            processed_lines.append(data_dict)
            
        print("Total skipped lines: ", skipped_lines)
        
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
        # save speakers dictionary
        with open(self.root_dir / "speakers.json", 'w') as f:
            speaker_dict = {s:i for i,s in enumerate(speaker_set)}
            print("saving speakers.json")
            json.dump(speaker_dict, f)

        with open(self.root_dir / "train.txt", 'w') as f:
            for line in processed_lines[self.val_size:]:
                f.write(str(line) + "\n")

        with open(self.root_dir / "val.txt", 'w') as f:
            for line in processed_lines[:self.val_size]:
                f.write(str(line) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="utils/TTE/TTE_config.yaml", help="path to config.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    Prep = Preprocessor(config)
    Prep.build_from_path()