# %%
import numpy as np
from tensorflow import keras
import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


# %%
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)


# %%
# Normalize the images.
train_images = (train_images / 255)
test_images = (test_images / 255)

# Type of Data is float64
train_images.dtype


# %%
rows = 2
columns = 3
for i in range(1, rows * columns + 1):
    plt.subplot(rows,columns,i)
    plt.imshow(train_images[i - 1], cmap='gray')

plt.show()


# %%
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)



# %%
# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# %%
num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(10, activation='softmax'),
])

# %%
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# %%
# Train the model.
model.fit(
  train_images,
  train_labels_cat,
  epochs=3,
  validation_data=(test_images, test_labels_cat),
)
# %%
# Predict on the first 5 test images.
predictions = model.predict(test_images[:6])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:6]) # [7, 2, 1, 0, 4]

rows = 3
columns = 3
for i in range(1, rows * columns + 1):
    plt.subplot(rows,columns,i)
    plt.imshow(test_images[i - 1], cmap='gray')

plt.show()

# %%
