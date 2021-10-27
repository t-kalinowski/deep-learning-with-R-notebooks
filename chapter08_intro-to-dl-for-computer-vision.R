#!/usr/bin/env Rscript

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.7.

# # Introduction to deep learning for computer vision

# ## Introduction to convnets

# **Instantiating a small convnet**

# In[ ]:


# from tensorflow import keras
# from tensorflow.keras import layers
# inputs = keras.Input(shape=(28, 28, 1))
# x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
# x = layers.Flatten()(x)
# outputs = layers.Dense(10, activation="softmax")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)

library(keras)
inputs <- layer_input(shape = c(28, 28, 1))
outputs <- inputs %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(10, activation = "softmax")
model <- keras_model(inputs, outputs)

# **Displaying the model's summary**

# In[ ]:


# model.summary()
model

# **Training the convnet on MNIST images**

# In[ ]:


# from tensorflow.keras.datasets import mnist
#
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images = train_images.reshape((60000, 28, 28, 1))
# train_images = train_images.astype("float32") / 255
# test_images = test_images.reshape((10000, 28, 28, 1))
# test_images = test_images.astype("float32") / 255
# model.compile(optimizer="rmsprop",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"])
# model.fit(train_images, train_labels, epochs=5, batch_size=64)

mnist <- dataset_mnist()
train_images <- array_reshape(mnist$train$x, c(60000, 28, 28, 1)) / 255
test_images <- array_reshape(mnist$test$x, c(10000, 28, 28, 1)) / 255
train_labels <- mnist$train$y
test_labels <- mnist$test$y
model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = c("accuracy"))
model %>% fit(train_images, train_labels, epochs = 5, batch_size = 64)
# **Evaluating the convnet**

# In[ ]:


# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f"Test accuracy: {test_acc:.3f}")
result <- evaluate(model, test_images, test_labels)
cat("Test accuracy:", result['accuracy'], "\n")

# ### The convolution operation

# #### Understanding border effects and padding

# #### Understanding convolution strides

# ### The max-pooling operation

# **An incorrectly structured convnet missing its max-pooling layers**

# In[ ]:


# inputs = keras.Input(shape=(28, 28, 1))
# x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
# x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
# x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
# x = layers.Flatten()(x)
# outputs = layers.Dense(10, activation="softmax")(x)
# model_no_max_pool = keras.Model(inputs=inputs, outputs=outputs)

inputs <- layer_input(shape=c(28, 28, 1))
outputs <- inputs %>%
  layer_conv_2d(filters=32, kernel_size=3, activation="relu") %>%
  layer_conv_2d(filters=64, kernel_size=3, activation="relu") %>%
  layer_conv_2d(filters=128, kernel_size=3, activation="relu") %>%
  layer_flatten() %>%
  layer_dense(10, activation = "softmax")
model_no_max_pool <- keras_model(inputs=inputs, outputs=outputs)

# In[ ]:


model_no_max_pool


# ## Training a convnet from scratch on a small dataset

# ### The relevance of deep learning for small-data problems

# ### Downloading the data

# In[ ]:


# from google.colab import files
# files.upload()


# In[ ]:


# get_ipython().system('mkdir ~/.kaggle')
# get_ipython().system('cp kaggle.json ~/.kaggle/')
# get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')

## R example assumes Linux/macOS
if(FALSE) {

system('mkdir ~/.kaggle')
system('cp kaggle.json ~/.kaggle/')
system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


# get_ipython().system('kaggle competitions download -c dogs-vs-cats')
system('kaggle competitions download -c dogs-vs-cats')

# In[ ]:


}
# get_ipython().system('unzip -qq train.zip')
unzip('dogs-vs-cats.zip', exdir = "dogs-vs-cats")

zip::unzip('dogs-vs-cats.zip', exdir = "dogs-vs-cats")
zip::unzip("dogs-vs-cats/train.zip", exdir = "dogs-vs-cats")
# unzip('dogs-vs-cats.zip')
# unzip('dogs-vs-cats.zip', exdir = "dogs-vs-cats")

# **Copying images to training, validation, and test directories**

# In[ ]:


# import os, shutil, pathlib
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

# library(fs)

original_dir <- fs::path("dogs-vs-cats/train")
new_base_dir <- fs::path("cats_vs_dogs_small")


make_subset <- function(subset_name, start_index, end_index) {
  categories <- c("dog", "cat")
  df <- tidyr::expand_grid(category = categories,
                           id = start_index:end_index) %>%
    dplyr::mutate(file_name = glue::glue("{category}.{id}.jpg"))
  fs::dir_create(new_base_dir / subset_name / categories)
  fs::file_copy(original_dir / df$file_name,
                new_base_dir / subset_name / df$category / df$file_name)
}

make_subset("train", start_index=0, end_index=999)
make_subset("validation", start_index=1000, end_index=1499)
make_subset("test", start_index=1500, end_index=2499)



# ### Building the model

# **Instantiating a small convnet for dogs vs. cats classification**

# In[ ]:


# from tensorflow import keras
# from tensorflow.keras import layers
#
# inputs = keras.Input(shape=(180, 180, 3))
# x = layers.Rescaling(1./255)(inputs)
# x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
# x = layers.Flatten()(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)

library(keras)

inputs <- layer_input(shape = c(180, 180, 3))
outputs <- inputs %>%
  layer_rescaling(1 / 255) %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)

# In[ ]:


# model.summary()
model

# **Configuring the model for training**

# In[ ]:


# model.compile(loss="binary_crossentropy",
#               optimizer="rmsprop",
#               metrics=["accuracy"])
model %>% compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = "accuracy")

# ### Data preprocessing

# **Using `image_dataset_from_directory` to read images**

# In[ ]:


# from tensorflow.keras.utils import image_dataset_from_directory
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


train_dataset <- image_dataset_from_directory(
  new_base_dir / "train",
  image_size = c(180, 180),
  batch_size = 32)
validation_dataset <- image_dataset_from_directory(
  new_base_dir / "validation",
  image_size = c(180, 180),
  batch_size = 32)
test_dataset <- image_dataset_from_directory(
  new_base_dir / "test",
  image_size = c(180, 180),
  batch_size = 32)

# In[ ]:


# import numpy as np
# import tensorflow as tf
# random_numbers = np.random.normal(size=(1000, 16))
# dataset = tf.data.Dataset.from_tensor_slices(random_numbers)

library(tfdatasets)
random_numbers  <- k_random_normal(shape=c(1000, 16))
dataset <- tensor_slices_dataset(random_numbers)

# In[ ]:


# for i, element in enumerate(dataset):
#     print(element.shape)
#     if i >= 2:
#         break

dataset_iterator <- as_iterator(dataset)
for(i in 1:3) {
  element <- iter_next(dataset_iterator)
  print(element)
}


# In[ ]:


# batched_dataset = dataset.batch(32)
# for i, element in enumerate(batched_dataset):
#     print(element.shape)
#     if i >= 2:
#         break

batched_dataset <- dataset %>%
  dataset_batch(32)
batched_dataset_iterator <- as_iterator(batched_dataset)
for(i in 1:3) {
  element <- iter_next(batched_dataset_iterator)
  print(dim(element))
}

# In[ ]:


# reshaped_dataset = dataset.map(lambda x: tf.reshape(x, (4, 4)))
# for i, element in enumerate(reshaped_dataset):
#     print(element.shape)
#     if i >= 2:
#         break

reshaped_dataset <- dataset %>%
  dataset_map(~ tf$reshape(.x, as_tensor(shape(4, 4))))
reshaped_dataset_iterator <- as_iterator(reshaped_dataset)
for(i in 1:3) {
  element <- iter_next(reshaped_dataset_iterator)
  print(dim(element))
}

# **Displaying the shapes of the data and labels yielded by the `Dataset`**

# In[ ]:


# for data_batch, labels_batch in train_dataset:
#     print("data batch shape:", data_batch.shape)
#     print("labels batch shape:", labels_batch.shape)
#     break

c(data_batch, labels_batch) %<-% iter_next(as_iterator(train_dataset))
data_batch$shape
labels_batch$shape
# cat("data batch shape: ", as.integer(data_batch$shape), "\n")
# cat("labels batch shape: ", as.integer(labels_batch$shape), "\n")

# **Fitting the model using a `Dataset`**

# In[ ]:


# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         filepath="convnet_from_scratch.keras",
#         save_best_only=True,
#         monitor="val_loss")
# ]
# history = model.fit(
#     train_dataset,
#     epochs=30,
#     validation_data=validation_dataset,
#     callbacks=callbacks)
callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_from_scratch.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>%
  fit(
    train_dataset,
    epochs = 30,
    validation_data = validation_dataset,
    callbacks = callbacks
  )

# **Displaying curves of loss and accuracy during training**

# In[ ]:


# import matplotlib.pyplot as plt
# accuracy = history.history["accuracy"]
# val_accuracy = history.history["val_accuracy"]
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(1, len(accuracy) + 1)
# plt.plot(epochs, accuracy, "bo", label="Training accuracy")
# plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()
# plt.show()

plot(history)

# **Evaluating the model on the test set**

# In[ ]:


# test_model = keras.models.load_model("convnet_from_scratch.keras")
# test_loss, test_acc = test_model.evaluate(test_dataset)
# print(f"Test accuracy: {test_acc:.3f}")
test_model <- load_model_tf("convnet_from_scratch.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))

# ### Using data augmentation

# **Define a data augmentation stage to add to an image model**

# In[ ]:


# data_augmentation = keras.Sequential(
#     [
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(0.1),
#         layers.RandomZoom(0.2),
#     ]
# )
data_augmentation <- keras_model_sequential() %>%
  layer_random_flip("horizontal") %>%
  layer_random_rotation(0.1) %>%
  layer_random_zoom(0.2)

# **Displaying some randomly augmented training images**

# In[ ]:


# plt.figure(figsize=(10, 10))
# for images, _ in train_dataset.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")
library(tfdatasets)

batch <- train_dataset %>%
  dataset_take(1) %>%
  as_array_iterator() %>%
  iter_next()
c(images, labels) %<-% batch
par(mfrow = c(3, 3), mar = c(1,0,1.5,0))
for(i in 1:9) {
  image <- images[i, , , ]
  label <- labels[i]
  plot(as.raster(image, max = 255))
  title(sprintf("label: %s", label))
}


# **Defining a new convnet that includes image augmentation and dropout**

# In[ ]:


# inputs = keras.Input(shape=(180, 180, 3))
# x = data_augmentation(inputs)
# x = layers.Rescaling(1./255)(x)
# x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
# x = layers.Flatten()(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
#
# model.compile(loss="binary_crossentropy",
#               optimizer="rmsprop",
#               metrics=["accuracy"])

inputs = layer_input(shape=c(180, 180, 3))
outputs <- inputs %>%
  data_augmentation() %>%
  layer_rescaling(1 / 255) %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs=inputs, outputs=outputs)

model %>% compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = "accuracy")


# **Training the regularized convnet**

# In[ ]:


# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         filepath="convnet_from_scratch_with_augmentation.keras",
#         save_best_only=True,
#         monitor="val_loss")
# ]
# history = model.fit(
#     train_dataset,
#     epochs=100,
#     validation_data=validation_dataset,
#     callbacks=callbacks)
callbacks = list(
  callback_model_checkpoint(
    filepath = "convnet_from_scratch_with_augmentation.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>% fit(
  train_dataset,
  epochs = 100,
  validation_data = validation_dataset,
  callbacks = callbacks
)

# **Evaluating the model on the test set**

# In[ ]:


# test_model = keras.models.load_model(
#     "convnet_from_scratch_with_augmentation.keras")
# test_loss, test_acc = test_model.evaluate(test_dataset)
# print(f"Test accuracy: {test_acc:.3f}")
test_model <- load_model_tf("convnet_from_scratch_with_augmentation.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))

# ## Leveraging a pretrained model

# ### Feature extraction with a pretrained model

# **Instantiating the VGG16 convolutional base**

# In[ ]:


# conv_base = keras.applications.vgg16.VGG16(
#     weights="imagenet",
#     include_top=False,
#     input_shape=(180, 180, 3))

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(180, 180, 3)
)

# In[ ]:


# conv_base.summary()
conv_base

# #### Fast feature extraction without data augmentation

# **Extracting the VGG16 features and corresponding labels**

# In[ ]:


# import numpy as np
#
# def get_features_and_labels(dataset):
#     all_features = []
#     all_labels = []
#     for images, labels in dataset:
#         preprocessed_images = keras.applications.vgg16.preprocess_input(images)
#         features = conv_base.predict(preprocessed_images)
#         all_features.append(features)
#         all_labels.append(labels)
#     return np.concatenate(all_features), np.concatenate(all_labels)
#
# train_features, train_labels =  get_features_and_labels(train_dataset)
# val_features, val_labels =  get_features_and_labels(validation_dataset)
# test_features, test_labels =  get_features_and_labels(test_dataset)


library(purrr)
# xyz <- function(...) .mapply(c, list(...), NULL)
get_features_and_labels <- function(dataset) {
  dataset %>%
    iterate(function(batch) {
      c(images, labels) %<-% batch
      preprocessed_images <-
        keras$applications$vgg16$preprocess_input(images)
      # TODO: complete exports keras::*_preprocess_input? or preprocess_application_input(x, model = "vgg16") ?
      features <- conv_base(preprocessed_images)
      list(features = features, labels = labels)
    }) %>%
    purrr::transpose() %>%
    map( ~ k_concatenate(.x, axis = 1))
}

c(train_features, train_labels) %<-% get_features_and_labels(train_dataset)
c(val_features, val_labels) %<-% get_features_and_labels(validation_dataset)
c(test_features, test_labels) %<-% get_features_and_labels(test_dataset)

# In[ ]:


# train_features.shape
train_features$shape


# **Defining and training the densely connected classifier**

# In[ ]:


# inputs = keras.Input(shape=(5, 5, 512))
# x = layers.Flatten()(inputs)
# x = layers.Dense(256)(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(loss="binary_crossentropy",
#               optimizer="rmsprop",
#               metrics=["accuracy"])
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#       filepath="feature_extraction.keras",
#       save_best_only=True,
#       monitor="val_loss")
# ]
# history = model.fit(
#     train_features, train_labels,
#     epochs=20,
#     validation_data=(val_features, val_labels),
#     callbacks=callbacks)



inputs = layer_input(shape = c(5, 5, 512))
outputs <- inputs %>%
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_dropout(.5) %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)

model %>% compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = "accuracy")

callbacks <- list(
  callback_model_checkpoint(
    filepath = "feature_extraction.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>% fit(
  train_features, train_labels,
  epochs = 20,
  validation_data = list(val_features, val_labels),
  callbacks = callbacks
)

# **Plotting the results**

# In[ ]:


# import matplotlib.pyplot as plt
# acc = history.history["accuracy"]
# val_acc = history.history["val_accuracy"]
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, "bo", label="Training accuracy")
# plt.plot(epochs, val_acc, "b", label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()
# plt.show()

plot(history)

# #### Feature extraction together with data augmentation

# **Instantiating and freezing the VGG16 convolutional base**

# In[ ]:


# conv_base  = keras.applications.vgg16.VGG16(
#     weights="imagenet",
#     include_top=False)
# conv_base.trainable = False

conv_base  = application_vgg16(
    weights="imagenet",
    include_top=FALSE)
# conv_base$trainable <-  FALSE
freeze_weights(conv_base)

# **Printing the list of trainable weights before and after freezing**

# In[ ]:


# conv_base.trainable = True
# print("This is the number of trainable weights "
#       "before freezing the conv base:", len(conv_base.trainable_weights))
unfreeze_weights(conv_base)
cat("This is the number of trainable weights",
    "before freezing the conv base:",
    length(conv_base$trainable_weights), "\n")

# In[ ]:


# conv_base.trainable = False
# print("This is the number of trainable weights "
#       "after freezing the conv base:", len(conv_base.trainable_weights))
freeze_weights(conv_base)
cat("This is the number of trainable weights",
    "after freezing the conv base:",
    length(conv_base$trainable_weights), "\n")

# **Adding a data augmentation stage and a classifier to the convolutional base**

# In[ ]:


# data_augmentation = keras.Sequential(
#     [
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(0.1),
#         layers.RandomZoom(0.2),
#     ]
# )
#
# inputs = keras.Input(shape=(180, 180, 3))
# x = data_augmentation(inputs)
# x = keras.applications.vgg16.preprocess_input(x)
# x = conv_base(x)
# x = layers.Flatten()(x)
# x = layers.Dense(256)(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(loss="binary_crossentropy",
#               optimizer="rmsprop",
#               metrics=["accuracy"])


data_augmentation <- keras_model_sequential() %>%
  layer_random_flip("horizontal") %>%
  layer_random_rotation(0.1) %>%
  layer_random_zoom(0.2)

inputs <- layer_input(shape = c(180, 180, 3))
outputs <- inputs %>%
  data_augmentation() %>%
  {keras$applications$vgg16$preprocess_input(.)} %>% # TODO: new preprocessor
  conv_base() %>%
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = "accuracy")

# In[ ]:


# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         filepath="feature_extraction_with_data_augmentation.keras",
#         save_best_only=True,
#         monitor="val_loss")
# ]
# history = model.fit(
#     train_dataset,
#     epochs=50,
#     validation_data=validation_dataset,
#     callbacks=callbacks)
callbacks <- list(
  callback_model_checkpoint(
    filepath = "feature_extraction_with_data_augmentation.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>%
  fit(
    train_dataset,
    epochs = 50,
    validation_data = validation_dataset,
    callbacks = callbacks
  )

# **Evaluating the model on the test set**

# In[ ]:


# test_model = keras.models.load_model(
#     "feature_extraction_with_data_augmentation.keras")
# test_loss, test_acc = test_model.evaluate(test_dataset)
# print(f"Test accuracy: {test_acc:.3f}")

test_model <- load_model_tf("feature_extraction_with_data_augmentation.keras")
result <- evaluate(test_model, test_dataset)
sprintf("Test accuracy: %.3f", result["accuracy"])

# ### Fine-tuning a pretrained model

# In[ ]:


# conv_base.summary()

conv_base


# **Freezing all layers until the fourth from the last**

# In[ ]:


# conv_base.trainable = True
# for layer in conv_base.layers[:-4]:
#     layer.trainable = False

conv_base$trainable <- TRUE
for (layer in head(conv_base$layers, -4))
    layer$trainable <- FALSE

# **Fine-tuning the model**

# In[ ]:


# model.compile(loss="binary_crossentropy",
#               optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
#               metrics=["accuracy"])
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         filepath="fine_tuning.keras",
#         save_best_only=True,
#         monitor="val_loss")
# ]
# history = model.fit(
#     train_dataset,
#     epochs=30,
#     validation_data=validation_dataset,
#     callbacks=callbacks)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 1e-5),
  metrics = "accuracy"
)

callbacks <- list(
  callback_model_checkpoint(
    filepath = "fine_tuning.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>% fit(
  train_dataset,
  epochs = 30,
  validation_data = validation_dataset,
  callbacks = callbacks
)

# In[ ]:


model <- load_model_tf("fine_tuning.keras")
result <-  evaluate(model, test_dataset)
sprintf("Test accuracy: %.3f", result["accuracy"])


# ## Summary
