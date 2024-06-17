#!/usr/bin/python3

import keras.callbacks
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from keras import layers
import wandb

import argparse
import wandb
import wandb.integration
import wandb.integration.keras
import datetime


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#==================PARSING COMMAND LINE ARGS=========================

def parse_args():
    parser = argparse.ArgumentParser(description="Set hyperparameters for the model training.")

    # Adding the hyperparameter arguments with their default values
    parser.add_argument('--seq_length', type=int, default=36,
                        help='Input sequence length')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train for')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Dimensionality of the embedding layer')
    parser.add_argument('--ff_dim', type=int, default=256,
                        help='Dimensionality of the feedforward network model')
    parser.add_argument('--enc_heads', type=int, default=2,
                        help='Number of heads in the encoder multi-head attention mechanism')
    parser.add_argument('--dec_heads', type=int, default=4,
                        help='Number of heads in the decoder multi-head attention mechanism')
    parser.add_argument('--artifact_dir', type=str, default="./local_runs/default_run",
                        help='Directory to save artifacts')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training')

    args = parser.parse_args()
    return args

args = parse_args()

#==================CONFIG=========================

#Hardcoded for now
DATA_PATH = "data/flickr30k_images/"
IMAGES_PATH = "data/flickr30k_images/flickr30k_images/"
IMAGE_SIZE=(224, 224)
VAL_FRACTION=0.05
VOCAB_SIZE=10000
AUTOTUNE=tf.data.AUTOTUNE
STRIP_CHARS = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~"

# Assigning values from args directly
SEQ_LENGTH = args.seq_length
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
EMBED_DIM = args.embed_dim
FF_DIM = args.ff_dim
ENC_HEADS = args.enc_heads
DEC_HEADS = args.dec_heads
ARTIFACT_DIR = args.artifact_dir
CNN_MODEL = "efficientnetb1"
LR = args.lr
TIMESTAMP = datetime.datetime.now().strftime("%m-%d-%H:%M")

current_config = {
        "TIMESTAMP" : TIMESTAMP,
        "DATA_PATH": DATA_PATH,
        "IMAGES_PATH": IMAGES_PATH,
        "IMAGE_SIZE": IMAGE_SIZE,
        "VAL_FRACTION": VAL_FRACTION,
        "SEQ_LENGTH": SEQ_LENGTH,
        "VOCAB_SIZE": VOCAB_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "STRIP_CHARS": STRIP_CHARS,
        "EPOCHS": EPOCHS,
        "EMBED_DIM" : EMBED_DIM,
        "FF_DIM" : FF_DIM,
        "ENC_HEADS" : ENC_HEADS,
        "DEC_HEADS" : DEC_HEADS,
        "ARTIFACT_DIR" : ARTIFACT_DIR,
        "CNN_MODEL" : CNN_MODEL,
        "LR" : LR,
    }

from training_utils import *
from data_processing import *


save_trial_config(current_config)


#WANDB INIT
RUNNAME=f"{TIMESTAMP}_model_emd{EMBED_DIM}_DH{DEC_HEADS}_EH{ENC_HEADS}_EPS{EPOCHS}_LR{LR}"
run = wandb.init(project="image_captioning",
            config=current_config,
            name=RUNNAME)



#create artifact dir
if not os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR)

save_trial_config(current_config)


#==================DATA PREPROCESSING=========================

###Creating datasets
captionings_df = pd.read_csv(os.path.join(DATA_PATH, "results.csv"), sep="|").dropna()
captionings_df.columns = ["image_name", "comment_number", "comment"]
captionings_df["image_name"] = IMAGES_PATH + "/" + captionings_df["image_name"] 

#ADDING START AND END special tokens
captionings_df["comment"] = "<START> " + captionings_df["comment"] + " <END>"
captionings_df = captionings_df.sample(frac=1,
                                       random_state=42,
                                       replace=False,
                                       )

n_train_examples = int(len(captionings_df) * (1 - VAL_FRACTION))

train_captionings_df = captionings_df[ : n_train_examples]
val_captionings_df = captionings_df[n_train_examples : ]

print("Train image-text examples: ", train_captionings_df.shape[0])
print("Validation image-text examples: ", val_captionings_df.shape[0])


##Prepare tokinzer
tokenizer = build_tokenizer(vocab_size=VOCAB_SIZE,
                            seq_len=SEQ_LENGTH)
tokenizer.adapt(train_captionings_df["comment"].tolist())
print("Tokenizer is ready")

save_tokenizer(tokenizer, os.path.join(ARTIFACT_DIR, "tokenizer.keras"))

#Create TF-datasets
def process_input(img_path, captions):
    return decode_and_resize(img_path, IMAGE_SIZE), tf.reshape(tokenizer(captions), shape=(1, SEQ_LENGTH))

def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset

train_dataset = make_dataset(train_captionings_df["image_name"].tolist(),
                             train_captionings_df["comment"].tolist())

val_dataset = make_dataset(train_captionings_df["image_name"].tolist(),
                             train_captionings_df["comment"].tolist())

print("TF-Datasets are created")





#==================BUILDING MODEL=========================
base_model = tf.keras.applications.EfficientNetB1(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

cnn = get_cnn_model(base_model)
print("\n\n\n==============FEATURE_EXTRACTOR==========\n\n")
print(cnn.summary())

encoder = TransformerEncoderBlock(
    embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=ENC_HEADS
)
decoder = TransformerDecoderBlock(
    embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=DEC_HEADS, 
    seq_len=SEQ_LENGTH,
    vocab_size=VOCAB_SIZE
)

caption_model = ImageCaptioningModel(
    cnn_model=cnn,
    encoder=encoder, 
    decoder=decoder
)
print("Model is built")

cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction="none"
)




#==================CALLBACKS=========================


checkpoint_saver = keras.callbacks.ModelCheckpoint(os.path.join(ARTIFACT_DIR, "checkpoints/weights_checkpoint.h5"),
                                                  verbose=1,
                                                 save_best_only=True,
                                                  save_weights_only=True)

early_stopping = keras.callbacks.EarlyStopping(patience=2,
                                               verbose=1)

wandb_logger = wandb.integration.keras.WandbCallback(verbose=1,
                                                     save_model=False
                                                     )


#==================TRAININING=========================

try:
    caption_model.compile(optimizer=keras.optimizers.Adam(LR), loss=cross_entropy)
    history = caption_model.fit(train_dataset,
                                validation_data=val_dataset,
                                epochs=EPOCHS,
                                callbacks=[early_stopping, wandb_logger, checkpoint_saver])
except KeyboardInterrupt:
    print("\n\nTraining is manually interupted\n")

#==================SAVING_ARTIFACTS=========================
try:
    save_training_history(history, os.path.join(ARTIFACT_DIR, "train_history.csv"))
except:
    print("Failed to save history")
caption_model.save_weights(os.path.join(ARTIFACT_DIR, RUNNAME + ".h5"))
print("Weights saved: ", RUNNAME + ".h5")

log_artifact_to_wandb(run, ARTIFACT_DIR, RUNNAME)


wandb.finish()