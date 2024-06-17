import gradio as gr
import numpy as np
import tensorflow as tf
from data_processing import *
from training_utils import *
import json

IMAGE_SIZE=(224, 224)

MODEL_WEIGHTS="assets/s_model_weights.h5"
MODEL_CONFIG="assets/small_model_config.json"
TOKENIZER="assets/s_tokenizer.keras"

with open(MODEL_CONFIG, "r") as conf_f:
    CONFIG = json.load(conf_f)

tokenizer = load_tokenizer(TOKENIZER)
caption_model = load_trained_model_weights(MODEL_WEIGHTS, CONFIG)
vocab = tokenizer.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = CONFIG["SEQ_LENGTH"]

def predict_caption(input_img : np.ndarray): 

    #preprocess
    img = tf.image.resize(input_img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img.numpy().clip(0, 255).astype(np.uint8)

    # Pass the image to the CNN
    img = tf.expand_dims(img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = tokenizer([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "end":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace("end", "").strip()
    print("Predicted Caption: ", decoded_caption)

    return decoded_caption


demo = gr.Interface(fn=predict_caption, inputs=gr.Image(), outputs="textbox")
    
demo.launch(share=True)

