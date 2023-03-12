import tensorflow as tf
from I2CS_layer import *
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

num_classes = 10
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(I2CS_layer(num_classes))

model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=I2CS_loss, metrics=I2CS_accuracy)


from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(model.summary())
epochs=5
history=model.fit(train_images, train_labels, epochs=epochs, batch_size=128,validation_data=(test_images,test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)


plt.plot(history.history['val_I2CS_accuracy'])
plt.title('Accuracy on test-data')
plt.ylabel('Accuracy')
plt.xlabel('ephocs')
plt.show()

