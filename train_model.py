
import pandas as pd
import numpy as np

# change working directory
english = pd.read_csv('machine_learning_certification/Challenge 7/en.csv')
french = pd.read_csv('machine_learning_certification/Challenge 7/fr.csv')

df = pd.concat([english, french],  axis=1)

df.columns = ['English', 'French']

import regex as re

def remove_punctuation(text):
    return re.sub(r'[.!?,;]', '', text)

df['English'] = df['English'].apply(lambda x: remove_punctuation(x))
df['French'] = df['French'].apply(lambda x: remove_punctuation(x))

# Make sure that the punctuation is removed by printing the example that you printed earlier.
print("English:", df['English'][2])
print("French:", df['French'][2])

df["ENG Length"] = df["English"].apply(lambda x: len(x.split()))
df.head()

df["FR Length"] = df["French"].apply(lambda x: len(x.split()))
df.head()

max_eng = max(df["ENG Length"])
max_fr = max(df["FR Length"])
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(df['English'])

fr_tokenizer = Tokenizer()
fr_tokenizer.fit_on_texts(df['French'])

fr_tokenized = fr_tokenizer.texts_to_sequences(df['French'])
eng_tokenized = eng_tokenizer.texts_to_sequences(df['English'])

eng_vocab_size = len(eng_tokenizer.word_index)
fr_vocab_size = len(fr_tokenizer.word_index)

from keras.preprocessing.sequence import pad_sequences

fr_tokenized_padded = pad_sequences(fr_tokenized, maxlen=max_fr, padding='post')
eng_tokenized_padded = pad_sequences(eng_tokenized, maxlen=max_eng, padding='post')

from keras.layers import Dense, LSTM, GRU, Embedding, RepeatVector, Input, TimeDistributed, Bidirectional
from keras.models import Sequential

model2 = Sequential()

model2.add(Embedding(eng_vocab_size+1, 100, input_length=max_eng))
model2.add(Bidirectional(GRU(20)))
model2.add(RepeatVector(max_fr))
model2.add(Bidirectional(GRU(20, return_sequences=True)))
model2.add(TimeDistributed(Dense(fr_vocab_size+1, activation="softmax")))

model2.summary()

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model2.fit(x=eng_tokenized_padded, y=fr_tokenized_padded, validation_split=0.2, epochs=5, batch_size=64)
