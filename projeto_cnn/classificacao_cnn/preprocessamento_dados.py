#Esse script irá carregar os dados e prepará-los para o treinamento.
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))  # Redimensiona a imagem para 224x224 pixels
    image = image / 255.0  # Normaliza os valores dos pixels
    return image, label

#carrega o conjunto de dados tf_flowers 
#e divide-o em treinamento(80%) e validação(20%)
def load_and_preprocess_data():
    (train_ds, val_ds), ds_info = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:]'],
        as_supervised=True,
        with_info=True,
    )

    #aplica o pre-processamento as imagens
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, ds_info

if __name__ == "__main__":
    train_ds, val_ds, ds_info = load_and_preprocess_data()
    print("Dados carregados e preprocessados com sucesso.")



