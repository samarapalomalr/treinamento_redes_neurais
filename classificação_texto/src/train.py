import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import prepare_data
from sklearn.model_selection import StratifiedKFold

def create_model(input_dim, output_dim=1):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128))  
    model.add(LSTM(128, return_sequences=True))  
    model.add(LSTM(64))  # Outra camada LSTM
    model.add(Dense(64, activation='relu'))  # Adicionar uma camada densa intermediária
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Função de validação cruzada k-fold
def k_fold_cross_validation(file_path, n_splits=5):
    X_train, X_test, y_train, y_test, tokenizer = prepare_data(file_path)
    
    # Convertendo y_train e y_test para arrays NumPy
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Criando k-folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_num = 1
    for train_index, val_index in skf.split(X_train, y_train):
        print(f"Treinando fold {fold_num}/{n_splits}...")
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = create_model(len(tokenizer.word_index) + 1)

        # Early stopping para evitar overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        # Treinando o modelo
        model.fit(X_train_fold, y_train_fold, 
                  validation_data=(X_val_fold, y_val_fold), 
                  epochs=5, 
                  batch_size=50,
                  callbacks=[early_stopping])
        
        # Avaliando o modelo com o conjunto de teste
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
        
        fold_num += 1

if __name__ == '__main__':
    k_fold_cross_validation('data/dataset.csv', n_splits=5)






