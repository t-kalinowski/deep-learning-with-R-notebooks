#!/usr/bin/env Rscript

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.7.

# # Advanced deep learning for computer vision

# ## Three essential computer vision tasks

# ## An image segmentation example

# In[ ]:


# get_ipython().system('wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz')
# get_ipython().system('wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz')
# get_ipython().system('tar -xf images.tar.gz')
# get_ipython().system('tar -xf annotations.tar.gz')
if(FALSE) {
system('wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz')
system('wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz')
system('tar -xf images.tar.gz')
system('tar -xf annotations.tar.gz')
}

tensorflow::tf$config$gpu.set_per_process_memory_growth(TRUE)


# In[ ]:


# import os
#
# input_dir = "images/"
# target_dir = "annotations/trimaps/"
#
# input_img_paths = sorted(
#     [os.path.join(input_dir, fname)
#      for fname in os.listdir(input_dir)
#      if fname.endswith(".jpg")])
# target_paths = sorted(
#     [os.path.join(target_dir, fname)
#      for fname in os.listdir(target_dir)
#      if fname.endswith(".png") and not fname.startswith(".")])
#

input_dir <- "images/"
target_dir <- "annotations/trimaps/"

input_img_paths <- fs::dir_ls(input_dir, glob = "*.jpg")
target_paths <- fs::dir_ls(target_dir, glob = "*.png")


# In[ ]:

#
# import matplotlib.pyplot as plt
# from tensorflow.keras.utils import load_img, img_to_array
#
# plt.axis("off")
# plt.imshow(load_img(input_img_paths[9]))
#

library(keras)
library(tensorflow)

img <- image_load(input_img_paths[10]) %>%
  image_to_array() %>%
  as.raster(max = 255) %>%
  plot()

# In[ ]:


# def display_target(target_array):
#     normalized_array = (target_array.astype("uint8") - 1) * 127
#     plt.axis("off")
#     plt.imshow(normalized_array[:, :, 0])
#
# img = img_to_array(load_img(target_paths[9], color_mode="grayscale"))
# display_target(img)

display_target <- function(target_array) {
  normalized_array <- (target_array - 1) * 127
  normalized_array <- tf$image$grayscale_to_rgb(as_tensor(normalized_array))
  normalized_array <- as.raster(as.array(normalized_array), max = 255)
  plot(normalized_array)
}

img <- image_to_array(image_load(target_paths[10], grayscale = TRUE))
display_target(img)

# In[ ]:

#
# import numpy as np
# import random
#
# img_size = (200, 200)
# num_imgs = len(input_img_paths)
#
# random.Random(1337).shuffle(input_img_paths)
# random.Random(1337).shuffle(target_paths)
#
# def path_to_input_image(path):
#     return img_to_array(load_img(path, target_size=img_size))
#
# def path_to_target(path):
#     img = img_to_array(
#         load_img(path, target_size=img_size, color_mode="grayscale"))
#     img = img.astype("uint8") - 1
#     return img
#
# input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
# targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
# for i in range(num_imgs):
#     input_imgs[i] = path_to_input_image(input_img_paths[i])
#     targets[i] = path_to_target(target_paths[i])
#
# num_val_samples = 1000
# train_input_imgs = input_imgs[:-num_val_samples]
# train_targets = targets[:-num_val_samples]
# val_input_imgs = input_imgs[-num_val_samples:]
# val_targets = targets[-num_val_samples:]

img_size <- c(200, 200)
num_imgs <- length(input_img_paths)

input_img_paths <- sample(input_img_paths)
target_paths <- sample(target_paths)

path_to_input_image <- function(path) {
  path %>%
    image_load(target_size = img_size) %>%
    image_to_array() %>%
    as_tensor()
}

path_to_target <- function(path) {
  im <- path %>%
    image_load(grayscale = TRUE, target_size = img_size) %>%
    image_to_array() %>%
    as_tensor()
  (im - 1)
}

input_imgs <- input_img_paths %>%
  lapply(path_to_input_image) %>%
  do.call(what = tuple, .) %>%
  tf$stack()

targets <- target_paths %>%
  lapply(path_to_target) %>%
  do.call(what = tuple) %>%
  tf$stack() %>%
  tf$cast(tf$uint8)

num_val_samples <- 1000

train_input_imgs <- input_imgs[`:-num_val_samples`,all_dims()]
train_targets = targets[`:-num_val_samples`, all_dims()]
val_input_imgs = input_imgs[`-num_val_samples:`, all_dims()]
val_targets = targets[`-num_val_samples:`, all_dims()]


# In[ ]:

#
# from tensorflow import keras
# from tensorflow.keras import layers
#
# def get_model(img_size, num_classes):
#     inputs = keras.Input(shape=img_size + (3,))
#     x = layers.Rescaling(1./255)(inputs)
#
#     x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
#     x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
#     x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
#     x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
#     x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
#
#     x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
#     x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", strides=2)(x)
#     x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
#     x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
#     x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
#     x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)
#
#     outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
#
#     model = keras.Model(inputs, outputs)
#     return model
#
# model = get_model(img_size=img_size, num_classes=3)
# model.summary()


get_model <- function(img_size, num_classes) {
  input <- layer_input(shape = c(img_size, 3))
  output <- input %>%
    layer_rescaling(scale = 1/255) %>%
    layer_conv_2d(filters = 64, kernel_size = 3, strides = 2, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = 128, kernel_size = 3, strides = 2, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = 256, kernel_size = 3, strides = 2, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu",
                  padding = "same") %>%


    layer_conv_2d_transpose(filters = 256, kernel_size = 3, activation = "relu",
                            padding = "same") %>%
    layer_conv_2d_transpose(filters = 256, kernel_size = 3, activation = "relu",
                            padding = "same", strides = 2) %>%
    layer_conv_2d_transpose(filters = 128, kernel_size = 3, activation = "relu",
                            padding = "same") %>%
    layer_conv_2d_transpose(filters = 128, kernel_size = 3, activation = "relu",
                            padding = "same", strides = 2) %>%
    layer_conv_2d_transpose(filters = 64, kernel_size = 3, activation = "relu",
                            padding = "same") %>%
    layer_conv_2d_transpose(filters = 64, kernel_size = 3, activation = "relu",
                            padding = "same", strides = 2) %>%

    layer_conv_2d(num_classes, 3, activation="softmax", padding="same")


  keras_model(input, output)
}

model <- get_model(img_size=img_size, num_classes=3)
model


# In[ ]:

#
# model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("oxford_segmentation.keras",
#                                     save_best_only=True)
# ]
#
# history = model.fit(train_input_imgs, train_targets,
#                     epochs=50,
#                     callbacks=callbacks,
#                     batch_size=64,
#                     validation_data=(val_input_imgs, val_targets))

model %>%
  compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks <- list(
    callback_model_checkpoint("oxford_segmentation.keras", save_best_only=TRUE)
)

history <- model %>% fit(
  train_input_imgs, train_targets,
  epochs=50,
  callbacks=callbacks,
  batch_size=64,
  validation_data=list(val_input_imgs, val_targets)
)


# In[ ]:

#
# epochs = range(1, len(history.history["loss"]) + 1)
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# plt.figure()
# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()

epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()


# In[ ]:

#
# from tensorflow.keras.utils import array_to_img
#
# model = keras.models.load_model("oxford_segmentation.keras")
#
# i = 4
# test_image = val_input_imgs[i]
# plt.axis("off")
# plt.imshow(array_to_img(test_image))
#
# mask = model.predict(np.expand_dims(test_image, 0))[0]
#
# def display_mask(pred):
#     mask = np.argmax(pred, axis=-1)
#     mask *= 127
#     plt.axis("off")
#     plt.imshow(mask)
#
# display_mask(mask)

from tensorflow.keras.utils import array_to_img

model = keras.models.load_model("oxford_segmentation.keras")

i = 4
test_image = val_input_imgs[i]
plt.axis("off")
plt.imshow(array_to_img(test_image))

mask = model.predict(np.expand_dims(test_image, 0))[0]

def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)

display_mask(mask)

