import os
import datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import RMSprop

# ================================================================
# Data loading
# ================================================================
features = pickle.load(open("featsV-1.pkl", "rb"))
tokenizer = Tokenizer(filters='', split=" ", lower=False)

# Read XSL training data
all_filenames = os.listdir('xsl/')
all_filenames.sort()
raw_xsl_code = [open('xsl/' + fname).read() for fname in all_filenames]

# Fit tokenizer and encode sequences
tokenizer.fit_on_texts(raw_xsl_code)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(raw_xsl_code)
max_length = max(len(s) for s in sequences)

# Prepare training pairs (image, partial sequence -> next word)
raw_xsl_code, raw_sequence_data, image_data = [], [], []
for img_no, seq in enumerate(sequences):
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        image_data.append(features[img_no])
        raw_xsl_code.append(in_seq[-200:])
        raw_sequence_data.append(out_seq)

raw_xsl_code = np.array(raw_xsl_code)
raw_sequence_data = np.array(raw_sequence_data)
image_data = np.array(image_data)

# ================================================================
# Model building blocks
# ================================================================
class TokenAndPositionEmbedding(layers.Layer):
    """Embedding layer that adds positional encodings."""
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(maxlen, embed_dim)
        self.maxlen = maxlen

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# ================================================================
# Image encoder
# ================================================================
image_model = Sequential([
    layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(8, 8, 1536,)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.RepeatVector(200),
])

visual_input = layers.Input(shape=(8, 8, 1536,))
encoded_image = image_model(visual_input)

# ================================================================
# Transformer-based language model with cross attention
# ================================================================
embed_dim = 256
num_heads = 8
ff_dim = 512

language_input = layers.Input(shape=(200,))
x = TokenAndPositionEmbedding(200, vocab_size, embed_dim)(language_input)
x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

# Cross attention with image features
cross_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
x2 = cross_attn(x, encoded_image)
x = layers.LayerNormalization(epsilon=1e-6)(x + x2)

x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(vocab_size, activation='softmax')(x)

model = Model(inputs=[visual_input, language_input], outputs=outputs)
optimizer = RMSprop(learning_rate=1e-4, clipvalue=1.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

os.makedirs("futureweights", exist_ok=True)
filepath = "futureweights/transformer-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')

log_dir = "tuber/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, update_freq='epoch')
callbacks_list = [tensorboard, checkpoint]

model.fit([image_data, raw_xsl_code], raw_sequence_data,
          validation_split=0.33,
          batch_size=128,
          shuffle=False,
          epochs=200,
          callbacks=callbacks_list)

