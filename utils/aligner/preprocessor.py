import argparse
from pathlib import Path
import yaml
import os
import pickle
import random
from typing import List
from langdetect import detect
from cleaners import english_cleaners, nonenglish_cleaners

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(config["path"]["root_dir"])
        self.dataset_dir = Path(config["path"]["dataset_dir"])
        
    @staticmethod
    def get_files(path: str, extension='.txt') -> List[Path]:
        return list(Path(path).expanduser().resolve().rglob(f'*{extension}'))
    
    def is_english(self, text: str) -> bool:
        try:
            return detect(text) == 'en'
        except Exception as e:
            print(f"Language detection error: {e}")
            return False
    
    def read_metafile(self, dataset_dir: str):
        text_dict = {}
        txt_files = self.get_files(dataset_dir, ".txt")

        if not txt_files:
            print(f"No .txt files found in {dataset_dir}")
            return text_dict

        for textfile in txt_files:
            with open(textfile, 'r') as f:
                line = f.read()
            if line=='':
                print(f"Ignoring file {textfile}, since its empty")
                continue
            text_dict[textfile.stem] = line

        return text_dict
        
    def build_from_path(self):
        print("Preprocessing data...")
        
        speakers = list(self.dataset_dir.glob('*'))
        if not speakers:
            print(f"No speakers found in {self.dataset_dir}")
            return
        
        print(f"There are {len(speakers)} speakers to process \n")
        
        symbols = []
        for speaker in speakers:
            print(f"Processing speaker {speaker}...")

            # Path to text files
            speaker_dir = speaker / 'txt'
            
            # Load all text files in a dictionary
            text_dict = self.read_metafile(speaker_dir)
            if not text_dict:
                print(f"    No text files found for speaker {speaker}")
                continue
            
            # Select a random text file for language detection
            selected_file = random.choice(list(text_dict.values()))
            if self.is_english(selected_file):
                use_englishcleaners = True
                print(f"    Detected English language for speaker {speaker}.")
            else:
                use_englishcleaners = False
                print(f"    Detected non-English language for speaker {speaker}.")
            
            # Declare path to store cleaned text files
            clean_speaker_dir = speaker / 'clean_txt'
            clean_speaker_dir.mkdir(parents=True, exist_ok=True)
    
            # Clean each text and store it at clean_speaker_dir
            print(f"    Cleaning text files for speaker {speaker}")
            for id, text in text_dict.items():
                cleaned_text = english_cleaners(text) if use_englishcleaners else nonenglish_cleaners(text)
                with open(clean_speaker_dir / f"{id}.txt", 'w') as f:
                    f.write(cleaned_text)
                    
            # Find unique characters/symbols from cleaned text files
            print(f"    Extracting characters/symbols from speaker {speaker}")
            cleaned_text_dict = self.read_metafile(clean_speaker_dir)
            for text in cleaned_text_dict.values():
                for char in text: 
                    if char not in symbols:
                        symbols.append(char)
                        
            print("\n")
                        
        symbols = sorted(symbols)
        print("\n")
        print(f"Unique symbols: {symbols}")
        print(f"Total symbols across all speakers: {len(symbols)}")
        
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with open(self.root_dir / "symbols.pkl", 'wb') as f:
            pickle.dump(symbols, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="utils/aligner/aligner_preprocessor_config.yaml", help="Path to DFA data.yaml")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found.")
        exit(1)

    Prep = Preprocessor(config)
    Prep.build_from_path()