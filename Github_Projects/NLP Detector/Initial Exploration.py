import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

sentances = [
    "I love my dog",
    "I love my cat"
]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentances)
word_index = tokenizer.word_index
print(word_index)
