# Parrot-TTS

Parrot-TTS is a text-to-speech (TTS) system that utilizes a Transformer based sequence-to-sequence model to map character tokens to HuBERT quantized units and a modified HiFi-GAN vocoder for speech synthesis. This repository is an official impplementation of our EACL 2024 paper available at https://aclanthology.org/2024.findings-eacl.6/. This repository provides instructions for installation, demo execution, and training the TTS model on your own data.

## Libraries Installation

1. **Create and activate a new Conda environment:**
    ```bash
    conda create --name parrottts python=3.8.19
    conda activate parrottts
    ```

2. **Install the required libraries:**
    ```bash
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu125
    ```

## Running a Demo

Run a demo using the provided Jupyter notebook, `demo.ipynb`

- The notebook will automatically download the following files from Google Drive:
    - `runs/aligner/symbol.pkl`: A dictionary to map characters to tokens.
    - `runs/TTE/ckpt`: Model to convert character text tokens to HuBERT units.
    - `runs/vocoder/checkpoints`: Model to predict speech from HuBERT units.

## Training Parrot-TTS on Your Data

To train Parrot-TTS on your dataset, follow these steps (1-10):

### Step 1: Compute Unique Symbols/Characters

- Update the `dataset_dir` folder in `utils/aligner/aligner_preprocessor_config.yaml`. The `dataset_dir` contains individual speakers and within it contains their `wavs` and `txt` files. The code cleans text files per speaker, stores them separately, and computes unique characters across all speakers.
    ```bash
    python utils/aligner/preprocessor.py utils/aligner/aligner_preprocessor_config.yaml
    ```

### Step 2: Train Aligner for Each Speaker

- Update `base_dataset_dir` in `train.sh`. `base_dataset_dir` is the same as `dataset_dir` used in Step 1.
    ```bash
    bash utils/aligner/train.sh
    ```

### Step 3: Extract HuBERT Units

- Download the HuBERT checkpoint and quantizer from [this link](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/textless_s2st_real_data.md) and store them in `utils/hubert_extraction`.
- Run the following command to extract HuBERT units:
    ```bash
    python utils/hubert_extraction/extractor.py utils/hubert_extraction/hubert_config.yaml
    ```
- Note: HuBERT units have already been extracted and are available at [this Google Drive link](https://drive.google.com/file/d/1kMPqObD9QlVmN3JzaUZ0jUJGBbFtEyrG/view?usp=drive_link).

### Step 4: Create Files for TTE Training

- Prepare the necessary files for training the TTE module:
    ```bash
    python utils/TTE/preprocessor.py utils/TTE/TTE_config.yaml
    ```

### Step 5: Train the TTE Module

- Train the TTE module using the following command:
    ```bash
    python train.py --config utils/TTE/TTE_config.yaml --num_gpus 1
    ```

### Step 6: Infer HuBERT Prediction

- Run inference to predict HuBERT from the trained TTE module:
    ```bash
    python inference.py --config utils/TTE/TTE_config.yaml --checkpoint_pth runs/TTE/ckpt/parrot_model-step=11000-val_total_loss_step=0.00.ckpt --device cuda:2
    ```

### Step 7: Create Training and Validation Files for Vocoder

- Generate training and validation files for the vocoder:
    ```bash
    python utils/vocoder/preprocessor.py --input_file runs/hubert_extraction/hubert.txt --root_path runs/vocoder
    ```

### Step 8: Train HiFi-GAN Vocoder

- Set the number of GPUs in the `nproc_per_node` variable and run the following command:
    ```bash
    CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.run --nproc_per_node=3 utils/vocoder/train.py --checkpoint_path runs/vocoder/checkpoints --config utils/vocoder/config.json
    ```

### Step 9: Infer Vocoder on Validation File

- Infer the vocoder on the validation file:
    ```bash
    python utils/vocoder/inference.py --checkpoint_file runs/vocoder/checkpoints -n 100 --vc --input_code_file runs/vocoder/val.txt --output_dir runs/vocoder/generations_vocoder
    ```

### Step 10: Infer Vocoder on Actual Predictions

- Infer the vocoder on predictions from the TTE module:
    ```bash
    python utils/vocoder/inference.py --checkpoint_file runs/vocoder/checkpoints -n 100 --vc --input_code_file runs/TTE/predictions.txt --output_dir runs/vocoder/generations_tte
    ```

## Acknowledgements

This repository is developed using insights from:
- [Speech Resynthesis by Facebook Research](https://github.com/facebookresearch/speech-resynthesis)
- [FastSpeech2 by Ming024](https://github.com/ming024/FastSpeech2)