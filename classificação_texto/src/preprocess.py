import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Mapeando os rótulos para inteiros
    label_mapping = {'spam': 1, 'ham': 0}  
    df['label'] = df['label'].map(label_mapping)

    return df['text'], df['label']

def preprocess_data(texts, max_words=5000, max_len=150):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences, tokenizer

def prepare_data(file_path):
    texts, labels = load_data(file_path)
    X, tokenizer = preprocess_data(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

    # Verifica e trata valores não finitos em y_train e y_test
    if y_train.isnull().any() or np.isinf(y_train).any():
        print("Valores não finitos encontrados em y_train. Tratando...")
        y_train = y_train.fillna(0) 
    if y_test.isnull().any() or np.isinf(y_test).any():
        print("Valores não finitos encontrados em y_test. Tratando...")
        y_test = y_test.fillna(0)  

    return X_train, X_test, y_train, y_test, tokenizer

