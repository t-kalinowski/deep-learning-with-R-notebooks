#!/usr/bin/env Rscript

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.7.

# # Fundamentals of machine learning

# ## Generalization: The goal of machine learning

# ### Underfitting and overfitting

# #### Noisy training data

# #### Ambiguous features

# #### Rare features and spurious correlations

# **Adding white-noise channels or all-zeros channels to MNIST**

# In[ ]:

#
# from tensorflow.keras.datasets import mnist
# import numpy as np
#
# (train_images, train_labels), _ = mnist.load_data()
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype("float32") / 255
#
# train_images_with_noise_channels = np.concatenate(
#     [train_images, np.random.random((len(train_images), 784))], axis=1)
#
# train_images_with_zeros_channels = np.concatenate(
#     [train_images, np.zeros((len(train_images), 784))], axis=1)

library(keras)

mnist <- dataset_mnist()
train_labels <- mnist$train$y
train_images <- mnist$train$x %>%
  array_reshape(c(60000, 28*28))
train_images <- train_images / 255

noise_channels <- array(runif(nrow(train_images) * 784),
                        dim = c(nrow(train_images), 784))
train_images_with_noise_channels <- cbind(train_images, noise_channels)

zeros_channels <- array(0, dim = c(nrow(train_images), 784))
train_images_with_zeros_channels <- cbind(train_images, zeros_channels)


# **Training the same model on MNIST data with noise channels or all-zero channels**

# In[ ]:


# from tensorflow import keras
# from tensorflow.keras import layers
#
# def get_model():
#     model = keras.Sequential([
#         layers.Dense(512, activation="relu"),
#         layers.Dense(10, activation="softmax")
#     ])
#     model.compile(optimizer="rmsprop",
#                   loss="sparse_categorical_crossentropy",
#                   metrics=["accuracy"])
#     return model
#
# model = get_model()
# history_noise = model.fit(
#     train_images_with_noise_channels, train_labels,
#     epochs=10,
#     batch_size=128,
#     validation_split=0.2)
#
# model = get_model()
# history_zeros = model.fit(
#     train_images_with_zeros_channels, train_labels,
#     epochs=10,
#     batch_size=128,
#     validation_split=0.2)


library(keras)

get_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(10, activation = "softmax")

  model %>% compile(
    optimizer = "rmsprop",
    loss = "sparse_categorical_crossentropy",
    metrics = "accuracy")

  model
}

model <- get_model()
history_noise <- model %>% fit(
    train_images_with_noise_channels, train_labels,
    epochs=10,
    batch_size=128,
    validation_split=0.2)

model <- get_model()
history_zeros <- model %>% fit(
    train_images_with_zeros_channels, train_labels,
    epochs=10,
    batch_size=128,
    validation_split=0.2)


# **Plotting a validation accuracy comparison**

# In[ ]:

#
# import matplotlib.pyplot as plt
# val_acc_noise = history_noise.history["val_accuracy"]
# val_acc_zeros = history_zeros.history["val_accuracy"]
# epochs = range(1, 11)
# plt.plot(epochs, val_acc_noise, "b-",
#          label="Validation accuracy with noise channels")
# plt.plot(epochs, val_acc_zeros, "b--",
#          label="Validation accuracy with zeros channels")
# plt.title("Effect of noise channels on validation accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()

library(patchwork)
# TODO: plot() should always draw to the graphics device,
# currently when method==ggplot it only returns a plot object
# which should be the work of autoplot()
# data.frame(
#   "Validation accuracy with noise_channels" = history_noise$metrics$val_accuracy,
#   "Validation accuracy with zeros channels" = history_zeros$metrics$val_accuracy
# ) %>% plot(type = 'l')
plot(NULL, NULL,
     main = "Effect of Noise Channels on Validation Accuracy",
     xlab = "Epochs", xlim = c(1, history_noise$params$epochs),
     ylab = "Validation Accuracy", ylim = c(0.9, 1))
lines(history_zeros$metrics$val_accuracy, lty = 1, type = 'o')
lines(history_noise$metrics$val_accuracy, lty = 2, type = 'o')
legend("bottomright", legend = c(
  "Validation accuracy with zeros channels",
  "Validation accuracy with noise channels"
  ), lty = 1:2
)



# ### The nature of generalization in deep learning

# **Fitting a MNIST model with randomly shuffled labels**

# In[ ]:

#
# (train_images, train_labels), _ = mnist.load_data()
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype("float32") / 255
#
# random_train_labels = train_labels[:]
# np.random.shuffle(random_train_labels)
#
# model = keras.Sequential([
#     layers.Dense(512, activation="relu"),
#     layers.Dense(10, activation="softmax")
# ])
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_images, random_train_labels,
#           epochs=100,
#           batch_size=128,
#           validation_split=0.2)

c(c(train_images, train_labels), .) %<-% dataset_mnist()
train_images <- train_images %>%
  array_reshape(c(60000, 28 * 28)) %>%
  `/`(255) #Q: do we teach this?

random_train_labels <- sample(train_labels)

model <- keras_model_sequential() %>%
  layer_dense(512, activation = "relu") %>%
  layer_dense(10, activation = "softmax")

model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")
history <- model %>% fit(train_images, random_train_labels,
              epochs = 100, batch_size = 128,
              validation_split = 0.2)
plot(history)
# #### The manifold hypothesis

# #### Interpolation as a source of generalization

# #### Why deep learning works

# #### Training data is paramount

# ## Evaluating machine-learning models

# ### Training, validation, and test sets

# #### Simple hold-out validation

# #### K-fold validation

# #### Iterated K-fold validation with shuffling

# ### Beating a common-sense baseline

# ### Things to keep in mind about model evaluation

# ## Improving model fit

# ### Tuning key gradient descent parameters

# **Training a MNIST model with an incorrectly high learning rate**

# In[ ]:

#
# (train_images, train_labels), _ = mnist.load_data()
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype("float32") / 255
#
# model = keras.Sequential([
#     layers.Dense(512, activation="relu"),
#     layers.Dense(10, activation="softmax")
# ])
# model.compile(optimizer=keras.optimizers.RMSprop(1.),
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_images, train_labels,
#           epochs=10,
#           batch_size=128,
#           validation_split=0.2)

c(c(train_images, train_labels), .) %<-% dataset_mnist()
train_images <- train_images %>%
  array_reshape(c(60000, 28 * 28)) %>% `/`(255)

model <- keras_model_sequential(list(
  layer_dense(units = 512, activation = "relu"),
  layer_dense(units = 10, activation = "softmax")
))
model %>% compile(optimizer=optimizer_rmsprop(1.),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")
model %>% fit(train_images, train_labels,
              epochs=10, batch_size=128,
              validation_split=0.2)


# **The same model with a more appropriate learning rate**

# In[ ]:

#
# model = keras.Sequential([
#     layers.Dense(512, activation="relu"),
#     layers.Dense(10, activation="softmax")
# ])
# model.compile(optimizer=keras.optimizers.RMSprop(1e-2),
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_images, train_labels,
#           epochs=10,
#           batch_size=128,
#           validation_split=0.2)

model <- keras_model_sequential(list(
  layer_dense(units = 512, activation = "relu"),
  layer_dense(units = 10, activation = "softmax")
))
model %>% compile(optimizer=optimizer_rmsprop(1e-2),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")
model %>% fit(train_images, train_labels,
              epochs=10, batch_size=128,
              validation_split=0.2)


# ### Leveraging better architecture priors

# ### Increasing model capacity

# **A simple logistic regression on MNIST**

# In[ ]:

#
# model = keras.Sequential([layers.Dense(10, activation="softmax")])
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# history_small_model = model.fit(
#     train_images, train_labels,
#     epochs=20,
#     batch_size=128,
#     validation_split=0.2)

model <- keras_model_sequential() %>%
  layer_dense(10, activation="softmax")
model %>% compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")
history_small_model <- model %>% fit(
    train_images, train_labels,
    epochs=20,
    batch_size=128,
    validation_split=0.2)


# In[ ]:

#
# import matplotlib.pyplot as plt
# val_loss = history_small_model.history["val_loss"]
# epochs = range(1, 21)
# plt.plot(epochs, val_loss, "b--",
#          label="Validation loss")
# plt.title("Effect of insufficient model capacity on validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

dplyr::as_tibble(history_small_model) %>%
  dplyr::filter(metric == "loss", data == "validation") %>%
  plot(value ~ epoch, data = ., type = "o",
       main = "Effect of Insufficient Model Capacity on Validation Loss",
       xlab = "Epochs", ylab = "Validation Loss")

# w/o going through data.frame
plot(history_small_model$metrics$val_loss, type = 'o',
     main = "Effect of Insufficient Model Capacity on Validation Loss",
     xlab = "Epochs", ylab = "Validation Loss")


# In[ ]:

# model = keras.Sequential([
#     layers.Dense(96, activation="relu"),
#     layers.Dense(96, activation="relu"),
#     layers.Dense(10, activation="softmax"),
# ])
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# history_large_model = model.fit(
#     train_images, train_labels,
#     epochs=20,
#     batch_size=128,
#     validation_split=0.2)

model <- keras_model_sequential() %>%
    layer_dense(96, activation="relu") %>%
    layer_dense(96, activation="relu") %>%
    layer_dense(10, activation="softmax")

model %>% compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")
history_large_model = model %>% fit(
    train_images, train_labels,
    epochs=20,
    batch_size=128,
    validation_split=0.2)


# ## Improving generalization

# ### Dataset curation

# ### Feature engineering

# ### Using early stopping

# ### Regularizing your model

# #### Reducing the network's size

# **Original model**

# In[ ]:

#
# from tensorflow.keras.datasets import imdb
# (train_data, train_labels), _ = imdb.load_data(num_words=10000)
#
# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.
#     return results
# train_data = vectorize_sequences(train_data)
#
# model = keras.Sequential([
#     layers.Dense(16, activation="relu"),
#     layers.Dense(16, activation="relu"),
#     layers.Dense(1, activation="sigmoid")
# ])
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# history_original = model.fit(train_data, train_labels,
#                              epochs=20, batch_size=512, validation_split=0.4)

c(c(train_data, train_labels), .) %<-% dataset_imdb(num_words = 10000)

vectorize_sequences <- function(sequences, dimension=10000) {

    results <- matrix(0, nrow = length(sequences), ncol = dimension)
    # results[cbind(seq_along(sequences), sequences)] <- 1
    for(i in seq_along(sequences))
        results[i, sequences[[i]]] <- 1.
    results
}
train_data <- vectorize_sequences(train_data)

model <- keras_model_sequential() %>%
    layer_dense(16, activation="relu") %>%
    layer_dense(16, activation="relu") %>%
    layer_dense(1, activation="sigmoid")

model %>% compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics="accuracy")

history_original <- model %>% fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512,
  validation_split = 0.4
)
plot(history_original)


# **Version of the model with lower capacity**

# In[ ]:

#
# model = keras.Sequential([
#     layers.Dense(4, activation="relu"),
#     layers.Dense(4, activation="relu"),
#     layers.Dense(1, activation="sigmoid")
# ])
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# history_smaller_model = model.fit(
#     train_data, train_labels,
#     epochs=20, batch_size=512, validation_split=0.4)

model = keras_model_sequential() %>%
    layer_dense(4, activation="relu") %>%
    layer_dense(4, activation="relu") %>%
    layer_dense(1, activation="sigmoid")

model %>% compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics="accuracy")

history_smaller_model <- model %>% fit(
  train_data, train_labels,
  epochs=20, batch_size=512, validation_split=0.4)

plot(history_smaller_model)

# **Version of the model with higher capacity**

# In[ ]:

#
# model = keras.Sequential([
#     layers.Dense(512, activation="relu"),
#     layers.Dense(512, activation="relu"),
#     layers.Dense(1, activation="sigmoid")
# ])
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# history_larger_model = model.fit(
#     train_data, train_labels,
#     epochs=20, batch_size=512, validation_split=0.4)

model = keras_model_sequential() %>%
    layer_dense(512, activation="relu") %>%
    layer_dense(512, activation="relu") %>%
    layer_dense(1, activation="sigmoid")

model %>% compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics="accuracy")
history_larger_model <- model %>% fit(
    train_data, train_labels,
    epochs=20, batch_size=512, validation_split=0.4)

plot(history_larger_model)

# #### Adding weight regularization

# **Adding L2 weight regularization to the model**

# In[ ]:

#
# from tensorflow.keras import regularizers
# model = keras.Sequential([
#     layers.Dense(16,
#                  kernel_regularizer=regularizers.l2(0.002),
#                  activation="relu"),
#     layers.Dense(16,
#                  kernel_regularizer=regularizers.l2(0.002),
#                  activation="relu"),
#     layers.Dense(1, activation="sigmoid")
# ])
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# history_l2_reg = model.fit(
#     train_data, train_labels,
#     epochs=20, batch_size=512, validation_split=0.4)


model <- keras_model_sequential() %>%
  layer_dense(16,
              kernel_regularizer = regularizer_l2(0.002),
              activation = "relu") %>%
  layer_dense(16,
              kernel_regularizer = regularizer_l2(0.002),
              activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics="accuracy")
history_l2_reg <- model %>% fit(
  train_data, train_labels,
  epochs=20, batch_size=512, validation_split=0.4)

plot(history_l2_reg)

# #### Adding weight regulariza

# **Different weight regularizers available in Keras**

# In[ ]:

#
# from tensorflow.keras import regularizers
# regularizers.l1(0.001)
# regularizers.l1_l2(l1=0.001, l2=0.001)

regularizer_l1(0.001)
regularizer_l1_l2(l1=0.001, l2=0.001)


# #### Adding dropout

# **Adding dropout to the IMDB model**

# In[ ]:

#
# model = keras.Sequential([
#     layers.Dense(16, activation="relu"),
#     layers.Dropout(0.5),
#     layers.Dense(16, activation="relu"),
#     layers.Dropout(0.5),
#     layers.Dense(1, activation="sigmoid")
# ])
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# history_dropout = model.fit(
#     train_data, train_labels,
#     epochs=20, batch_size=512, validation_split=0.4)

model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(16, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics="accuracy")
history_dropout <- model %>% fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512,
  validation_split = 0.4
)

plot(history_dropout)


# ## Summary
