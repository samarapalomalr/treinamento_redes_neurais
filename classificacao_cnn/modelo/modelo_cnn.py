#Armazena o código-fonte do modelo de CNN.
#Ativando ambiente virtual: venv/bin/activate
import tensorflow as tf

def criar_modelo():
    model = tf.keras.Sequential([
        #2 camadas convolucionais com 32 e 64 filtros
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        #2 camadas de pooling (reduz dimensionalidade)
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        #1 camada densa com 128 neuronios
        tf.keras.layers.Dense(128, activation='relu'),
        #1 camada com 5 saidas(classificaçoes)
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    #compilando o modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = criar_modelo()
    #Modelo sendo exibido
    model.summary()



