import tensorflow as tf
import numpy as np
from PIL import Image
import os
from modelo.modelo_cnn import criar_modelo

#carrega uma imagem usando a biblioteca PIL
#e a redimensiona
def carregar_e_processar_imagem(caminho_imagem):
    try:
        imagem = Image.open(caminho_imagem)
        imagem = imagem.resize((224, 224))
        imagem = np.array(imagem) / 255.0
        return np.expand_dims(imagem, axis=0)
    except Exception as e:
        print(f"Erro ao carregar ou processar a imagem: {e}")
        return None

#carrega o modelo salvo e faz a previsao da classe da imagem
def prever_imagem(caminho_imagem):
    if not os.path.exists(caminho_imagem):
        print(f"O caminho da imagem fornecido não existe: {caminho_imagem}")
        return None
    
    imagem = carregar_e_processar_imagem(caminho_imagem)
    if imagem is None:
        return None

    model = tf.keras.models.load_model('scripts/modelo/final_modelo.keras')
    
    previsao = model.predict(imagem)
    classe_prevista = np.argmax(previsao, axis=1)
    return classe_prevista

if __name__ == "__main__":
    caminho_imagem = 'scripts/download.jpeg' 
    classe = prever_imagem(caminho_imagem)
    if classe is not None:
        print(f'Classe prevista: {classe}')

