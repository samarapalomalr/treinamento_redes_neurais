# treinamento_redes_neurais
Este repositório contém dois projetos distintos que implementam redes neurais para classificação: um para **classificação de imagens** usando Redes Neurais Convolucionais (CNN) 
e outro para **classificação de texto** utilizando redes neurais.

Classificação de Imagens
O projeto de classificação de imagens utiliza uma Rede Neural Convolucional (CNN) implementada com TensorFlow e Keras. O objetivo é classificar imagens em cinco categorias usando o conjunto de dados tf_flowers.
O projeto inclui scripts para:

    Pré-processamento de Dados: Carrega e prepara os dados para treinamento.
    Treinamento: Treina o modelo, salvando checkpoints ao longo do processo.
    Validação: Avalia o modelo em um conjunto de validação.
    Teste: Realiza inferências com novas imagens.



Classificação de texto
Esse projeto é um classificador de texto para identificar mensagens de spam e não spam utilizando uma rede neural recorrente com LSTMs (Long Short-Term Memory). 
A ideia principal é treinar um modelo para prever se uma mensagem de texto é spam ou não com base em seu conteúdo.
O projeto inclui scripts para:
    
    Preprocessamento: As mensagens são tokenizadas e convertidas em sequências de números. Os dados são então divididos em treinamento e teste.
    Construção do Modelo: Um modelo de rede neural com camadas LSTM é criado para aprender as sequências de texto.
    Treinamento: O modelo é treinado utilizando cross-validation para garantir que o desempenho seja consistente entre diferentes subconjuntos dos dados.
    Avaliação: O modelo é avaliado em termos de perda (loss) e precisão (accuracy).

Esse projeto é uma aplicação prática de Processamento de Linguagem Natural (NLP) e Redes Neurais, voltado para a detecção de spam, e pode ser expandido para outros tipos de classificação de texto.
