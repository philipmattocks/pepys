from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from keras.models import Model
from keras import Input
from keras.layers import Dense

def clean_and_tokenise(raw_text):
    raw_text = raw_text.lower()
    raw_text = word_tokenize(raw_text)
    raw_text = [word for word in raw_text if word not in (['the', 'a', 'and', 'is', 'be', 'will'])]
    return raw_text


def get_focus_context_and_unique(words, window_size):
    word_pairs = []
    unique_words = set()
    for i, word in enumerate(words):
        unique_words.add(word)
        for w in range(window_size):
            if i + w + 1 < len(words):
                word_pairs.append((word, words[i + w + 1]))
        for w in range(window_size):
            if i - w - 1 >= 0:
                word_pairs.append((word,words[i - w - 1]))
    return word_pairs, unique_words

def create_index(unique):
    return {word:i for i, word in enumerate(unique)}



def create_one_hot(unique, word_pairs, word_index):
    X = []
    Y = []
    for x,y in word_pairs:
        X_arr = np.zeros(len(unique))
        Y_arr = np.zeros(len(unique))
        X_arr[word_index[x]] = 1
        Y_arr[word_index[y]] = 1
        X.append(X_arr)
        Y.append(Y_arr)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y

window = 1
embed_size = 2
text = 'The future king is the prince Daughter is the princess Son is the prince Only a man can be a king Only a woman ' \
       'can be a queen The princess will be a queen Queen and king rule the realm The prince is a strong man The princess ' \
       'is a beautiful woman The royal family is the king and queen and their children Prince is only a boy now A boy will be a man'

text  = clean_and_tokenise(text)

word_pairs, unique_words = get_focus_context_and_unique(text, window)
            
print(word_pairs)
print(unique_words)
word_index = create_index(unique_words)
X,Y = create_one_hot(unique_words,word_pairs, word_index)
# Defining the neural network
inp = Input(shape=(X.shape[1],))
x = Dense(units=embed_size, activation='linear')(inp)
x = Dense(units=Y.shape[1], activation='softmax')(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# Optimizing the network weights
model.fit(
    x=X,
    y=Y,
    batch_size=256,
    epochs=1000
)

embeddings = model.get_weights()[0]
word_embedding_dict = {word:embeddings[i] for i,word in enumerate(word_index)}

plt.figure(figsize=(10, 10))
for word,coords in word_embedding_dict.items():
    plt.scatter(coords[0], coords[1])
    plt.annotate(word, (coords[0], coords[1]))
plt.show()

