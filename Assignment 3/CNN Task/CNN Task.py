# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)


# %%
train_labels_one_hot[0]


# %%




