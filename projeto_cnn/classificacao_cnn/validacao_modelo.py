import tensorflow as tf
from preprocessamento_dados import load_and_preprocess_data

def validar_modelo():
    _, val_ds, _ = load_and_preprocess_data()
    
    # Carrega o modelo salvo
    model = tf.keras.models.load_model('scripts/modelo/final_modelo.keras')
    
    # Avalia o modelo nos dados de validação
    #exibindo a perda e a acuracia
    loss, accuracy = model.evaluate(val_ds)
    print(f"Acurácia na validação: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    validar_modelo()


