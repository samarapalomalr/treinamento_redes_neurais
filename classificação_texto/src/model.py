#model.py
# Definição do modelo de rede neural 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

def build_model(vocab_size, embedding_dim=128, input_length=150):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))  # Dropout para regularização
    model.add(LSTM(64, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

