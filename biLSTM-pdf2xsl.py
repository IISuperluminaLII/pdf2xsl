import os
import datetime
import numpy as np

# Use TensorFlow's bundled Keras implementation to avoid relying on the
# deprecated standalone ``keras`` package.  This improves compatibility with
# modern CPU-only environments where ``CuDNNLSTM`` is unavailable.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Embedding,
    RepeatVector,
    concatenate,
    Input,
    Bidirectional,
    Dense,
    Dropout,
    Flatten,
    Conv2D,
)
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from feature_utils import load_or_generate_features

#set_session(session)
features = load_or_generate_features()
print(len(features))
tokenizer = Tokenizer(filters='', split=" ", lower=False)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_doc(fn):
    file = open(fn, 'r')
    text = file.read()
    file.close()
    return text

raw_xsl_code = []
all_filenames = os.listdir('xsl/')
all_filenames.sort()
for filename in all_filenames:
    raw_xsl_code.append(load_doc('xsl/' + filename))
tokenizer.fit_on_texts(raw_xsl_code)

vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(raw_xsl_code)
max_length = max(len(s) for s in sequences)

raw_xsl_code, raw_sequence_data, image_data = list(), list(), list()
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

# raw_xsl_code = pickle.load(open("rawxc.pkl", "rb"))
# raw_sequence_data = pickle.load(open("rsd.pkl", "rb"))
# image_data =  pickle.load(open("idcg.pkl", "rb"))

#Create the encoder
image_model = Sequential()
image_model.add(Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(8, 8, 1536,)))
image_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

image_model.add(Flatten())
image_model.add(Dense(1024, activation='relu'))
image_model.add(Dropout(0.3))
image_model.add(Dense(1024, activation='relu'))
image_model.add(Dropout(0.3))

image_model.add(RepeatVector(200))

visual_input = Input(shape=(8, 8, 1536,))
encoded_image = image_model(visual_input)

language_input = Input(shape=(200,))
language_model = Embedding(vocab_size, 100, input_length=200)(language_input)
language_model = Bidirectional(CuDNNLSTM(512, return_sequences=True))(language_model)
language_model = Bidirectional(CuDNNLSTM(512, return_sequences=True))(language_model)

#Create the decoder
decoder = concatenate([encoded_image, language_model])
decoder = Bidirectional(CuDNNLSTM(1024, return_sequences=True))(decoder)
decoder = Bidirectional(CuDNNLSTM(1024, return_sequences=False))(decoder)
decoder = Dense(vocab_size, activation='softmax')(decoder)

# Compile the model
model = Model(inputs=[visual_input, language_input], outputs=decoder)
optimizer = RMSprop(learning_rate=0.0001, clipvalue=1.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

os.makedirs("futureweights", exist_ok=True)
filepath = "futureweights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')

tensorboard = TensorBoard(log_dir=("tuber\/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"),
                          write_graph=True, update_freq='epoch')
callbacks_list = [tensorboard, checkpoint]
# Let the learning begin

model.fit([image_data, raw_xsl_code], raw_sequence_data, validation_split=0.33, batch_size=128, shuffle=False,
          epochs=200, callbacks=callbacks_list)


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


#print(word_for_id(17, tokenizer))


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'START'
    # iterate over the whole length of the sequence
    for i in range(900):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0][-200:]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # Print the prediction
        print(' ' + word, end='')
        # stop if we predict the end of the sequence
        if word == 'END':
            break
    return


# Load and image, preprocess it for IR2, extract features and generate the HTML
test_image = img_to_array(load_img('pdf_images/3.jpg', target_size=(300, 300)))
test_image = np.array(test_image, dtype=float)
test_image = preprocess_input(test_image)
test_features = InceptionResNetV2(weights='imagenet', include_top=False).predict(np.array([test_image]))
generate_desc(model, tokenizer, np.array(test_features), 200)
