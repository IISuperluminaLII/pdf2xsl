import os
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


def build_model(maxlen: int, vocab_size: int, layout_vocab: int) -> Model:
    """Create a style‑conditioned Transformer for XSL → PDF layout."""

    # Encode XSL tokens
    xsl_input = layers.Input(shape=(maxlen,), name="xsl_tokens")
    x = TokenAndPositionEmbedding(maxlen, vocab_size, 256)(xsl_input)
    x = TransformerBlock(256, 4, 512)(x)

    # Encode style from a reference PDF page using a CNN as in ResNet50
    style_input = layers.Input(shape=(224, 224, 3), name="style_image")
    base_cnn = ResNet50(include_top=False, weights="imagenet", pooling="avg")
    base_cnn.trainable = False
    style_feat = base_cnn(style_input)
    style_feat = layers.Dense(256, activation="relu")(style_feat)
    style_feat = layers.RepeatVector(maxlen)(style_feat)

    # Combine style and content, then decode into layout tokens
    combined = layers.Concatenate()([x, style_feat])
    combined = TransformerBlock(512, 8, 1024)(combined)
    combined = layers.GlobalAveragePooling1D()(combined)
    layout_output = layers.Dense(layout_vocab, activation="softmax", name="layout_token")(combined)

    model = Model(inputs=[xsl_input, style_input], outputs=layout_output)
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def tokens_to_pdf(layout_tokens, pdf_path: str):
    """Render layout tokens into a styled PDF using fpdf2."""
    try:
        from fpdf import FPDF
    except Exception:  # pragma: no cover - library is optional for compilation
        return
    pdf = FPDF()
    pdf.add_page()
    for token in layout_tokens:
        text = token.get("text", "")
        x = token.get("x", 10)
        y = token.get("y", 10)
        pdf.text(x, y, text)
    pdf.output(pdf_path)


if __name__ == "__main__":
    # Example workflow operating on multiple XSL files and style images
    xsl_files = sorted(glob.glob("xsl/*.xsl"))
    style_images = sorted(glob.glob("style_pdfs/*.png"))

    xsl_texts = [open(f).read() for f in xsl_files]
    tokenizer = Tokenizer(filters="", split=" ", lower=False)
    tokenizer.fit_on_texts(xsl_texts)
    sequences = tokenizer.texts_to_sequences(xsl_texts)
    maxlen = 200
    sequences = pad_sequences(sequences, maxlen=maxlen)

    layout_vocab = 500  # vocabulary of layout tokens (e.g., bounding boxes, fonts)
    dummy_layout = np.zeros((len(sequences), layout_vocab), dtype="float32")
    dummy_images = np.zeros((len(sequences), 224, 224, 3), dtype="float32")

    model = build_model(maxlen, len(tokenizer.word_index) + 1, layout_vocab)
    model.fit([sequences, dummy_images], dummy_layout, epochs=1)
