#!/usr/bin/env Rscript
# coding: utf-8

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition?a_aid=keras&a_bid=76564dff). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.6.

# ## Modern convnet architecture patterns

# ### Modularity, hierarchy, and reuse

# ### Residual connections

# **Residual block where the number of filters changes**

# In[ ]:


# from tensorflow import keras
# from tensorflow.keras import layers
#
# inputs = keras.Input(shape=(32, 32, 3))
# x = layers.Conv2D(32, 3, activation="relu")(inputs)
# residual = x
# x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
# residual = layers.Conv2D(64, 1)(residual)
# x = layers.add([x, residual])


library(keras)

inputs <- layer_input(shape = c(32, 32, 3))
x <- layer_conv_2d(inputs, filters = 32, kernel_size = 3, activation = "relu")
residual <- x
x <- layer_conv_2d(x, filters = 64, kernel_size = 3, activation = "relu",
                   padding = "same")
residual <- layer_conv_2d(residual, filters = 64, kernel_size = 1)
x <- layer_add(list(x, residual))

# **Case where target block includes a max pooling layer**

# In[ ]:


# inputs = keras.Input(shape=(32, 32, 3))
# x = layers.Conv2D(32, 3, activation="relu")(inputs)
# residual = x
# x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
# x = layers.MaxPooling2D(2, padding="same")(x)
# residual = layers.Conv2D(64, 1, strides=2)(residual)
# x = layers.add([x, residual])


inputs <- layer_input(shape = c(32, 32, 3))
x <- layer_conv_2d(inputs, filters = 32, kernel_size = 3, activation = "relu")
residual <- x
x <- layer_conv_2d(x, filters = 64, kernel_size = 3, activation = "relu",
                   padding = "same")
x <- layer_max_pooling_2d(x, pool_size = 2, padding = "same")
residual <- layer_conv_2d(residual, filters = 64, kernel_size = 1, strides = 2)
x <- layer_add(list(x, residual))

# In[ ]:


# inputs = keras.Input(shape=(32, 32, 3))
# x = layers.Rescaling(1./255)(inputs)
#
# def residual_block(x, filters, pooling=False):
#     residual = x
#     x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
#     x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
#     if pooling:
#         x = layers.MaxPooling2D(2, padding="same")(x)
#         residual = layers.Conv2D(filters, 1, strides=2)(residual)
#     elif filters != residual.shape[-1]:
#         residual = layers.Conv2D(filters, 1)(residual)
#     x = layers.add([x, residual])
#     return x
#
# x = residual_block(x, filters=32, pooling=True)
# x = residual_block(x, filters=64, pooling=True)
# x = residual_block(x, filters=128, pooling=False)
#
# x = layers.GlobalAveragePooling2D()(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
# model.summary()

inputs <- layer_input(shape = c(32, 32, 3))
x <- layer_rescaling(inputs, scale = 1/255)

residual_block <- function(x, filters, pooling = FALSE) {
  residual <- x
  x <- x %>%
    layer_conv_2d(filters = filters, kernel_size = 3, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = filters, kernel_size = 3, activation = "relu",
                  padding = "same")

  if (pooling) {
    x <- layer_max_pooling_2d(x, pool_size = 2, padding = "same")
    residual <- layer_conv_2d(residual, filters = filters, kernel_size = 1,
                              strides = 2)
  } else if (filters != residual$shape[[4]]) {
    residual <- layer_conv_2d(residual, filters = filters, kernel_size = 1)
  }

  layer_add(list(x, residual))
}

x <- residual_block(x, filters=32, pooling=TRUE)
x <- residual_block(x, filters=64, pooling=TRUE)
x <- residual_block(x, filters=128, pooling=FALSE)

x <- layer_global_average_pooling_2d(x)
outputs <- layer_dense(x, units = 1, activation="sigmoid")

model <- keras_model(inputs = inputs, outputs = outputs)
model

# ### Batch normalization

# ### Depthwise separable convolutions

# ### Putting it together: A mini Xception-like model

# In[ ]:


# from google.colab import files
# files.upload()

# In[ ]:


# get_ipython().system('mkdir ~/.kaggle')
# get_ipython().system('cp kaggle.json ~/.kaggle/')
# get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')
# get_ipython().system('kaggle competitions download -c dogs-vs-cats')
# get_ipython().system('unzip -qq train.zip')

if (FALSE) {
  system('kaggle competitions download -c dogs-vs-cats')
  system('unzip -qq dogs-vs-cats.zip')
  system('unzip -qq train.zip')
}

# In[ ]:


# import os, shutil, pathlib
# from tensorflow.keras.utils import image_dataset_from_directory
#
# original_dir = pathlib.Path("train")
# new_base_dir = pathlib.Path("cats_vs_dogs_small")
#
# def make_subset(subset_name, start_index, end_index):
#     for category in ("cat", "dog"):
#         dir = new_base_dir / subset_name / category
#         os.makedirs(dir)
#         fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
#         for fname in fnames:
#             shutil.copyfile(src=original_dir / fname,
#                             dst=dir / fname)
#
# make_subset("train", start_index=0, end_index=1000)
# make_subset("validation", start_index=1000, end_index=1500)
# make_subset("test", start_index=1500, end_index=2500)
#
# train_dataset = image_dataset_from_directory(
#     new_base_dir / "train",
#     image_size=(180, 180),
#     batch_size=32)
# validation_dataset = image_dataset_from_directory(
#     new_base_dir / "validation",
#     image_size=(180, 180),
#     batch_size=32)
# test_dataset = image_dataset_from_directory(
#     new_base_dir / "test",
#     image_size=(180, 180),
#     batch_size=32)


original_dir <- "train"
new_base_dir <- "cats_vs_dogs_small"

if (FALSE) {

  make_subset <- function(subset_name, start_index, end_index) {
    for (category in c("cat", "dog")) {
      dir <- fs::path(new_base_dir, subset_name, category)
      fs::dir_create(dir)
      fnames <- glue::glue("{category}.{start_index:(end_index-1)}.jpg")
      fs::file_copy(fs::path(original_dir, fnames), fs::path(dir, fnames))
    }
  }

  make_subset("train", start_index=0, end_index=1000)
  make_subset("validation", start_index=1000, end_index=1500)
  make_subset("test", start_index=1500, end_index=2500)

}

train_dataset <- image_dataset_from_directory(
  fs::path(new_base_dir, "train"),
  image_size=c(180, 180),
  batch_size=32)
validation_dataset <- image_dataset_from_directory(
  fs::path(new_base_dir, "validation"),
  image_size=c(180, 180),
  batch_size=32)
test_dataset <- image_dataset_from_directory(
  fs::path(new_base_dir, "test"),
  image_size=c(180, 180),
  batch_size=32)


# In[ ]:


# data_augmentation = keras.Sequential(
#     [
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(0.1),
#         layers.RandomZoom(0.2),
#     ]
# )


data_augmentation <- keras_model_sequential() %>%
  layer_random_flip(mode = "horizontal") %>%
  layer_random_rotation(factor = 0.1) %>%
  layer_random_zoom(height_factor = 0.2)

# In[ ]:


# inputs = keras.Input(shape=(180, 180, 3))
# x = data_augmentation(inputs)
#
# x = layers.Rescaling(1./255)(x)
# x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)
#
# for size in [32, 64, 128, 256, 512]:
#     residual = x
#
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)
#
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)
#
#     x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
#
#     residual = layers.Conv2D(
#         size, 1, strides=2, padding="same", use_bias=False)(residual)
#     x = layers.add([x, residual])
#
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)

inputs <- layer_input(shape = c(180, 180, 3))
x <- data_augmentation(inputs)

x <- layer_rescaling(x, scale = 1/255)
x <- layer_conv_2d(x, filters = 32, kernel_size = 5, use_bias = FALSE)

for (size in c(32, 64, 128, 256, 512)) {

  residual <- x

  x <- x %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_separable_conv_2d(filters = size, kernel_size = 3, padding = "same",
                            use_bias = FALSE) %>%

    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_separable_conv_2d(filters = size, kernel_size = 3, padding = "same",
                            use_bias = FALSE) %>%

    layer_max_pooling_2d(pool_size = 3, strides = 2, padding = "same")

  residual <- layer_conv_2d(residual, filters = size, kernel_size = 1, strides = 2,
                            padding = "same", use_bias = FALSE)

  x <- layer_add(list(x, residual))
}

outputs <- x %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)

# In[ ]:


# model.compile(loss="binary_crossentropy",
#               optimizer="rmsprop",
#               metrics=["accuracy"])
# history = model.fit(
#     train_dataset,
#     epochs=100,
#     validation_data=validation_dataset)

model %>%
  compile(
    loss="binary_crossentropy",
    optimizer="rmsprop",
    metrics="accuracy"
  )

history <- model %>%
  fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset)
