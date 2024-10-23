#redimensionar_imagens.py
#usa a biblioteca PIL para manipulação das imagens 
from PIL import Image
import os

def redimensionar_imagem(caminho_imagem, caminho_saida, tamanho=(224, 224)):
    try:
        imagem = Image.open(caminho_imagem)
        imagem = imagem.resize(tamanho, Image.Resampling.LANCZOS)  
        imagem.save(caminho_saida)
        print(f"Imagem salva em {caminho_saida}")
    except Exception as e:
        print(f"Erro ao redimensionar a imagem {caminho_imagem}: {e}")

def redimensionar_imagens_diretorio(diretorio_entrada, diretorio_saida, tamanho=(224, 224)):
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
    
    for nome_arquivo in os.listdir(diretorio_entrada):
        caminho_imagem = os.path.join(diretorio_entrada, nome_arquivo)
        caminho_saida = os.path.join(diretorio_saida, nome_arquivo)
        
        if os.path.isfile(caminho_imagem):
            redimensionar_imagem(caminho_imagem, caminho_saida, tamanho)

if __name__ == "__main__":
    diretorio_entrada = 'scripts/diretorio_imagens'
    diretorio_saida = 'scripts'
    tamanho_imagem = (224, 224)  # Tamanho para redimensionar as imagens
    
    redimensionar_imagens_diretorio(diretorio_entrada, diretorio_saida, tamanho_imagem)
