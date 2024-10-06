import random
import argparse
import os
import shutil

# Argument parser to handle input and output file paths
parser = argparse.ArgumentParser(description="Split data into train and val files.")
parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
parser.add_argument('--root_path', type=str, required=True, help='Path to save the train.txt and val.txt file')

# Parse the arguments
args = parser.parse_args()

# Read the input file
with open(args.input_file, 'r') as f:
    lines = f.readlines()

# Shuffle the lines for randomness
random.shuffle(lines)

# Split the data into 98% train and 10% val
split_index = int(0.98 * len(lines))
train_lines = lines[:split_index]
val_lines = lines[split_index:]

if os.path.exists(args.root_path):
    shutil.rmtree(args.root_path)
os.makedirs(args.root_path)
    
# Write to train.txt
with open(args.root_path+'/train.txt', 'w') as f:
    f.writelines(train_lines)

# Write to val.txt
with open(args.root_path+'/val.txt', 'w') as f:
    f.writelines(val_lines)

print(f"Data split into train.txt and val.txt successfully.")