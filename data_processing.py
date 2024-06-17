import tensorflow as tf
import numpy as np
import re
import keras



@tf.keras.saving.register_keras_serializable()
def text_standrardization(input_str):
    strip_chars = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~"

    lowercase = tf.strings.lower(input_str)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def build_tokenizer(vocab_size, seq_len):

    vectorization = tf.keras.layers.TextVectorization(
                        max_tokens=vocab_size,
                        output_mode="int",
                        output_sequence_length=seq_len,
                        #standardize=text_standrardization,
                        
                                                    )
    return vectorization

def build_image_augmenter(rotation_rate=0.2, contrast=0.3):
    image_augmentation = tf.keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(rotation_rate),
        keras.layers.RandomContrast(contrast),
    ]
    )
    return image_augmentation

def decode_and_resize(img_path, image_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


