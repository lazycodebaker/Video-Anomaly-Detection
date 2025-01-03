 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

SIZE = 128
BATCH_SIZE = 64 

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode='input'
)

validation_generator = datagen.flow_from_directory(
    'data/validation',
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode='input'
)

anomaly_generator = datagen.flow_from_directory(
    'data/anomaly',
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode='input'
)

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=10
)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

data_batch = []
img_num = 0

while img_num < train_generator.batch_index:
    data = train_generator.next()
    data_batch.append(data[0])
    img_num += 1
    
predicted = model.predict(data_batch[0])

image_number = random.randint(0,predicted.shape[0])
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(data_batch[0][image_number])
plt.subplot(1, 2, 2)
plt.imshow(predicted[image_number])
plt.show()

validation_error = model.evaluate_generator(validation_generator)
anomaly_error = model.evaluate_generator(anomaly_generator)

print(validation_error)
print(anomaly_error)

encoder_model = Sequential()
encoder_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
encoder_model.add(MaxPooling2D((2, 2), padding='same'))
encoder_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
encoder_model.add(MaxPooling2D((2, 2), padding='same'))
encoder_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
encoder_model.add(MaxPooling2D((2, 2), padding='same'))
encoder_model.summary()

from sklearn.neighbors import KernelDensity

encoded_image = encoder_model.predict_generator(train_generator)
encoder_ouput_shape = encoder_model.output_shape

out_vector_shape = encoder_model[1] * encoder_ouput_shape[2] * encoder_ouput_shape[3]

encoded_images_vector = [np.reshape(img,(out_vector_shape,)) for img in encoded_image]

kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(encoded_images_vector)

def calc_anomaly_score(image):