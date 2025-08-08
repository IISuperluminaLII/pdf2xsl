import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam


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
    """Basic Transformer block used by LayoutTransformer and similar models."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
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


def build_model(maxlen: int, vocab_size: int, pdf_len: int) -> Model:
    """Create a style‑conditioned Transformer that emits raw PDF bytes.

    The network encodes an XSL token sequence, fuses it with a CNN style
    embedding of a reference PDF page, and directly decodes a sequence of
    byte values (0‑255) representing the final PDF file.  No external
    rendering libraries are used – the model's predictions are written
    verbatim to disk.
    """

    # Encode XSL tokens
    xsl_input = layers.Input(shape=(maxlen,), name="xsl_tokens")
    x = TokenAndPositionEmbedding(maxlen, vocab_size, 256)(xsl_input)
    x = TransformerBlock(256, 4, 512)(x)

    # Encode style image using ResNet50 to capture document appearance
    style_input = layers.Input(shape=(224, 224, 3), name="style_image")
    base_cnn = ResNet50(include_top=False, weights="imagenet", pooling="avg")
    base_cnn.trainable = False
    style_feat = base_cnn(style_input)
    style_feat = layers.Dense(256, activation="relu")(style_feat)
    style_feat = layers.Reshape((1, 256))(style_feat)

    # Cross‑attention to inject style context into the token sequence
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=256)
    style_context = attn(query=x, value=style_feat, key=style_feat)

    # Combine style context with token embeddings then decode bytes
    combined = layers.Concatenate()([x, style_context])
    combined = TransformerBlock(512, 8, 1024)(combined)
    combined = layers.GlobalAveragePooling1D()(combined)
    byte_logits = layers.Dense(pdf_len * 256)(combined)
    byte_logits = layers.Reshape((pdf_len, 256))(byte_logits)
    pdf_bytes = layers.Softmax(axis=-1, name="pdf_bytes")(byte_logits)

    model = Model(inputs=[xsl_input, style_input], outputs=pdf_bytes)
    model.compile(optimizer=Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def preds_to_pdf(pred_seq, pdf_path: str):
    """Write model predictions directly to a PDF file.

    ``pred_seq`` should be a sequence of probability distributions over
    256 byte values produced by ``build_model``. The argmax byte is taken
    at each position and written verbatim to ``pdf_path``.
    """

    byte_ids = tf.argmax(pred_seq, axis=-1)
    byte_arr = tf.cast(byte_ids, tf.uint8).numpy().tobytes()
    with open(pdf_path, "wb") as f:
        f.write(byte_arr)



if __name__ == "__main__":
    # Example workflow operating on multiple XSL files
    xsl_files = sorted(glob.glob("xsl/*.xsl"))

    xsl_texts = [open(f).read() for f in xsl_files]
    tokenizer = Tokenizer(filters="", split=" ", lower=False)
    tokenizer.fit_on_texts(xsl_texts)
    sequences = tokenizer.texts_to_sequences(xsl_texts)
    maxlen = 200
    sequences = pad_sequences(sequences, maxlen=maxlen)

    pdf_len = 256  # length of PDF byte sequence to model
    dummy_pdf = np.zeros((len(sequences), pdf_len), dtype="int32")
    template = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
    dummy_pdf[:, : len(template)] = np.frombuffer(template, dtype=np.uint8)
    dummy_images = np.zeros((len(sequences), 224, 224, 3), dtype="float32")

    model = build_model(maxlen, len(tokenizer.word_index) + 1, pdf_len)
    model.fit([sequences, dummy_images], dummy_pdf, epochs=1)

    preds = model.predict([sequences, dummy_images])
    preds_to_pdf(preds[0], "sample_output.pdf")
