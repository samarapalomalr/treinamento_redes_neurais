#Script para treinar o modelo de CNN com os dados pre-processados
import os
import tensorflow as tf
from preprocessamento_dados import load_and_preprocess_data
from modelo.modelo_cnn import criar_modelo

def treinar_modelo():
    #carrega e pre-processa dos dados
    train_ds, val_ds, _ = load_and_preprocess_data()
    #cria o modelo
    model = criar_modelo()

    #salva os checks points durante o treinamento 
    checkpoint_dir = 'scripts/modelo/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    #o script usa o callbacks do keras para salvar o melhor modelo
    #durante o treinamento
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'modelo_{epoch:02d}.keras'),
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[checkpoint_callback])

    # Salvando o modelo no formato nativo do Keras
    #o modelo final é salvo em final_modelo.keras
    model.save('scripts/modelo/final_modelo.keras')

if __name__ == "__main__":
    treinar_modelo()
    print("Modelo treinado e salvo com sucesso.")