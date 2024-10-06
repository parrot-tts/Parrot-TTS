# Parrot-TTS

# Libraries installation
    ```
    conda create --name parrottts python=3.8.19
    conda activate parrottts
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu125
    ```

# To run a demo using pre-trained models available at runs, go to "demo.ipynb". This repo has pre-trained models available at:
    ```
    runs/TTE/ckpt : Model to convert text tokens to HuBERT units
    runs/vocoder/checkpoints : Model to predict speech from HuBERT units
    ```

# To train Parrot-TTS on your data. Please refer below steps (1-10)
# Step 1 :
- Compute unique symbols/character across all the speaker. Change data folder in "utils/aligner/aligner_preprocessor_config.yaml". The code first cleans text files per speaker. Stores it seperately and then computes unique characters found across all speakers
    ```
    python utils/aligner/preprocessor.py utils/aligner/aligner_preprocessor_config.yaml
    ```

# Step 2 :
- Train aligner for every speaker. Change base_dataset_dir in train.sh.
    ```
    bash utils/aligner/train.sh
    ```

# Step 3 :
- Extract hubert units for every speaker audio files. creates a single hubert file for all speakers. The mhubert checkpoint and quantizer can be downloaded from "https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/textless_s2st_real_data.md" and stored at "utils/hubert_extraction" that is once downloaded the folder should look like: "utils/hubert_extraction/mhubert_base_vp_en_es_fr_it3.pt" and "utils/hubert_extraction/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin". Once downloaded, below command can be run. For easiness, the hubert units are already extracted using the above model and stored at "runs/hubert_extraction/hubert.txt"
    ```
    python utils/hubert_extraction/extractor.py utils/hubert_extraction/hubert_config.yaml
    ```

# Step 4: 
- Create necessary files for training TTE module
    ```
    python utils/TTE/preprocessor.py utils/TTE/TTE_config.yaml
    ```

# Step 5: 
- Train TTE module
    ```
    python train.py --config utils/TTE/TTE_config.yaml --num_gpus 1
    ```

# Step 6: 
- Infer hubert prediction from trained TTE module
    ```
    python inference.py --config utils/TTE/TTE_config.yaml --checkpoint_pth runs/TTE/ckpt/parrot_model-step=11000-val_total_loss_step=0.00.ckpt --device cuda:2
    ```

# Step 7: 
- create train and validation files for training vocoder
    ```
    python utils/vocoder/preprocessor.py --input_file runs/hubert_extraction/hubert.txt --root_path runs/vocoder
    ```

# Step 8: 
- Train HifiGAN speech vocoder. Mention number of gpus want to set in nproc_per_node variable
    ```
    CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.run --nproc_per_node=3 utils/vocoder/train.py --checkpoint_path runs/vocoder/checkpoints --config utils/vocoder/config.json

    ```

# Step 9: 
- infer vocoder on validation file
    ```
    python utils/vocoder/inference.py --checkpoint_file runs/vocoder/checkpoints -n 100 --vc --input_code_file runs/vocoder/val.txt --output_dir runs/vocoder/generations_vocoder
    ```

# Step 10: 
- infer vocoder on actual prediction from TTE trained step obtained from step 6
    ```
    python utils/vocoder/inference.py --checkpoint_file runs/vocoder/checkpoints -n 100 --vc --input_code_file runs/TTE/predictions.txt --output_dir runs/vocoder/generations_tte
    ```