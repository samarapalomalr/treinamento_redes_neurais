import tensorflow_datasets as tfds
import tensorflow as tf

# Carrega o conjunto de dados ImageNet v2
dataset, info = tfds.load("imagenet_v2", split="test", as_supervised=True, with_info=True)

# Exibe algumas informações sobre o conjunto de dados
print(info)

# Exibe algumas imagens do conjunto de dados
import matplotlib.pyplot as plt

for image, label in dataset.take(5):
    plt.imshow(image.numpy())
    plt.title(f"Label: {label.numpy()}")
    plt.show()


