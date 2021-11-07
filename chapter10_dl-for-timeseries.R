#!/usr/bin/env R
# coding: utf-8

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.7.

# # Deep learning for timeseries

# ## Different kinds of timeseries tasks

# ## A temperature-forecasting example

# In[ ]:


# get_ipython().system('wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip')
# get_ipython().system('unzip jena_climate_2009_2016.csv.zip')

if(FALSE) {
  system('wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip')
  system('unzip jena_climate_2009_2016.csv.zip')
}


# **Inspecting the data of the Jena weather dataset**

# In[ ]:


# import os
# fname = os.path.join("jena_climate_2009_2016.csv")
#
# with open(fname) as f:
#     data = f.read()
#
# lines = data.split("\n")
# header = lines[0].split(",")
# lines = lines[1:]
# print(header)
# print(len(lines))
library(keras)

# data = readr::read_csv("jena_climate_2009_2016.csv")
df <- readr::read_csv("jena_climate_2009_2016.csv.zip")
df

# **Parsing the data**

# In[ ]:


# import numpy as np
# temperature = np.zeros((len(lines),))
# raw_data = np.zeros((len(lines), len(header) - 1))
# for i, line in enumerate(lines):
#     values = [float(x) for x in line.split(",")[1:]]
#     temperature[i] = values[1]
#     raw_data[i, :] = values[:]

temperature <- df$`T (degC)`
raw_data <- df %>% dplyr::select(-`Date Time`) %>% as.matrix()

# **Plotting the temperature timeseries**

# In[ ]:


# from matplotlib import pyplot as plt
# plt.plot(range(len(temperature)), temperature)
plot(temperature, pch = 19, cex = .5)


# **Plotting the first 10 days of the temperature timeseries**

# In[ ]:


# plt.plot(range(1440), temperature[:1440])
plot(temperature[1:1440])

# **Computing the number of samples we'll use for each data split**

# In[ ]:


# num_train_samples = int(0.5 * len(raw_data))
# num_val_samples = int(0.25 * len(raw_data))
# num_test_samples = len(raw_data) - num_train_samples - num_val_samples
# print("num_train_samples:", num_train_samples)
# print("num_val_samples:", num_val_samples)
# print("num_test_samples:", num_test_samples)
num_train_samples <- round(nrow(raw_data) * .5)
num_val_samples <- round(nrow(raw_data) * 0.25)
num_test_samples <- nrow(raw_data) - num_train_samples - num_val_samples
cat("num_train_samples:", num_train_samples, "\n")
cat("num_val_samples:", num_val_samples, "\n")
cat("num_test_samples:", num_test_samples, "\n")


# ### Preparing the data

# **Normalizing the data**

# In[ ]:

library(purrr)
library(dplyr)

# mean = raw_data[:num_train_samples].mean(axis=0)
# raw_data -= mean
# std = raw_data[:num_train_samples].std(axis=0)
# raw_data /= std

# train_df <- df %>% slice(seq(1, num_train_samples))
# val_df <- df %>% slice(seq(num_train_samples + 1,
#                            length.out = num_val_samples))
# test_df <- df %>% slice(seq(num_train_samples + num_val_samples + 1,
#                             length.out = num_val_samples))
# train_df <- train_df %>%
#   mutate(across(is.numeric, mean))
# mean <- apply(raw_data[1:num_train_samples,], 1, mean)

#TODO: this whole sequence should probably be reworked to split early,
# define a standardization function, then apply it to each dataset split.

# mean <- map_dbl(1:num_train_samples, ~mean(raw_data[.x,]))
# raw_data <- raw_data - mean

# std <- map_dbl(1:num_train_samples, ~sd(raw_data[.x,]))
# raw_data <- raw_data / std

# Note: decided to go full explicit here, but
# also should consider if using listarrays::map_along_rows() or apply()
# or map_dfc?
# is more readable.
# Also, not sure if working with a matrix is the best idea here, maybe we should
# work with a data.frame here?
# for(col_idx in 1:ncol(raw_data)) {
#   col <- raw_data[, col_idx]
#   train_col <- col[1:num_train_samples]
#   col <- col - mean(train_col)
#   col <- col / sd(train_col)
#   raw_data[, col_idx] <- col
# }
library(purrr)
# need to save a reference to the normalization_values for unnormalizeing preds later
normalization_values <-
  map(set_names(colnames(raw_data)), function(col_nm) {
    train_col <- raw_data[1:num_train_samples, col_nm]
    list(mean = mean(train_col), sd = sd(train_col))
  })

for(col_nm in colnames(raw_data)) {
  col <- raw_data[,col_nm]
  nv <- normalization_values[[col_nm]]
  col <- (col - nv$mean) / nv$sd
  raw_data[ ,col_nm] <- col
}



# In[ ]:

#
# import numpy as np
# from tensorflow import keras
# int_sequence = np.arange(10)
# dummy_dataset = keras.utils.timeseries_dataset_from_array(
#     data=int_sequence[:-3],
#     targets=int_sequence[3:],
#     sequence_length=3,
#     batch_size=2,
# )
#
# for inputs, targets in dummy_dataset:
#     for i in range(inputs.shape[0]):
#         print([int(x) for x in inputs[i]], int(targets[i]))
#
# import numpy as np
# from tensorflow import keras
# int_sequence = np.arange(10)
# dummy_dataset = keras.utils.timeseries_dataset_from_array(
#     data=int_sequence[:-3],
#     targets=int_sequence[3:],
#     sequence_length=3,
#     batch_size=2,
# )
#
# for inputs, targets in dummy_dataset:
#     for i in range(inputs.shape[0]):
#         print([int(x) for x in inputs[i]], int(targets[i]))
#

library(keras)
int_sequence <- seq(10)
dummy_dataset <- timeseries_dataset_from_array(
    data=head(int_sequence, -3), # drop last 3
    targets= tail(int_sequence, -3), # drop first 3
    sequence_length=3,
    batch_size=2
)

library(tfdatasets)
dummy_dataset %>%
  as_array_iterator() %>%
  iterate(function(element) {
    c(inputs, targets) %<-% element
    for (r in 1:nrow(inputs))
      cat(sprintf("input: [ %s ]  target: %s\n",
                  paste(inputs[r, ], collapse = " "),
                  targets[r]))
  })




# **Instantiating datasets for training, validation, and testing**

# In[ ]:

#
# sampling_rate = 6
# sequence_length = 120
# delay = sampling_rate * (sequence_length + 24 - 1)
# batch_size = 256
#
# train_dataset = keras.utils.timeseries_dataset_from_array(
#     raw_data[:-delay],
#     targets=temperature[delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=True,
#     batch_size=batch_size,
#     start_index=0,
#     end_index=num_train_samples)
#
# val_dataset = keras.utils.timeseries_dataset_from_array(
#     raw_data[:-delay],
#     targets=temperature[delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=True,
#     batch_size=batch_size,
#     start_index=num_train_samples,
#     end_index=num_train_samples + num_val_samples)
#
# test_dataset = keras.utils.timeseries_dataset_from_array(
#     raw_data[:-delay],
#     targets=temperature[delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=True,
#     batch_size=batch_size,
#     start_index=num_train_samples + num_val_samples)
#

sampling_rate <- 6
sequence_length <- 120
delay <- sampling_rate * (sequence_length + 24 - 1)
batch_size <- 256

#Q: consider making raw_data a tensor here so you can index into it using
#negative indices like raw_data[:-delay], targets=temperature[delay:],


train_dataset <- timeseries_dataset_from_array(
  head(raw_data, -delay),
  # head slices on rows
  targets = tail(temperature, -delay),
  sampling_rate = sampling_rate,
  sequence_length = sequence_length,
  shuffle = TRUE,
  batch_size = batch_size,
  start_index = 0,
  end_index = num_train_samples
)

val_dataset <- timeseries_dataset_from_array(
  head(raw_data, -delay),
  targets = tail(temperature, -delay),
  sampling_rate = sampling_rate,
  sequence_length = sequence_length,
  shuffle = TRUE,
  batch_size = batch_size,
  start_index = num_train_samples,
  end_index = num_train_samples + num_val_samples
)

test_dataset <- timeseries_dataset_from_array(
  head(raw_data, -delay),
  targets = tail(temperature, -delay),
  sampling_rate = sampling_rate,
  sequence_length = sequence_length,
  shuffle = TRUE,
  batch_size = batch_size,
  start_index = num_train_samples + num_val_samples
)

# **Inspecting the output of one of our datasets**

# In[ ]:

#
# for samples, targets in train_dataset:
#     print("samples shape:", samples.shape)
#     print("targets shape:", targets.shape)
#     break

c(samples, targets) %<-% iter_next(as_iterator(train_dataset))
samples$shape
targets$shape



# ### A common-sense, non-machine-learning baseline

# **Computing the common-sense baseline MAE**

# In[ ]:

#
# def evaluate_naive_method(dataset):
#     total_abs_err = 0.
#     samples_seen = 0
#     for samples, targets in dataset:
#         preds = samples[:, -1, 1] * std[1] + mean[1]
#         total_abs_err += np.sum(np.abs(preds - targets))
#         samples_seen += samples.shape[0]
#     return total_abs_err / samples_seen
#
# print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
# print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}")

evaluate_naive_method <- function(dataset) {
  total_abs_err <- 0
  samples_seen <- 0L
  ds_iterator <- as_array_iterator(dataset)
  temp_nv <- normalization_values[["T (degC)"]]
  unnormalize <- function(x) (x * temp_nv$sd) + temp_nv$mean
  repeat {
    batch <- iter_next(ds_iterator) %||% break
    c(samples, targets) %<-% batch
    preds <- samples[, dim(samples)[2], 2] # col 2 of raw_data is Temperature
    preds <- unnormalize(preds)
    total_abs_err <- total_abs_err + sum(abs(preds - targets))
    samples_seen <- samples_seen + dim(samples)[1]
  }

  total_abs_err / samples_seen
}

sprintf("Validation MAE: %.2f", evaluate_naive_method(val_dataset))
sprintf("Test MAE: %.2f", evaluate_naive_method(test_dataset))


# ### Let's try a basic machine-learning model

# **Training and evaluating a densely connected model**

# In[ ]:

#
# from tensorflow import keras
# from tensorflow.keras import layers
#
# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.Flatten()(inputs)
# x = layers.Dense(16, activation="relu")(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("jena_dense.keras",
#                                     save_best_only=True)
# ]
# model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
# history = model.fit(train_dataset,
#                     epochs=10,
#                     validation_data=val_dataset,
#                     callbacks=callbacks)
#
# model = keras.models.load_model("jena_dense.keras")
# print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")


inputs <- layer_input(shape = c(sequence_length, ncol(raw_data)))
outputs <- inputs %>%
  layer_flatten() %>%
  layer_dense(16, activation="relu") %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks = list(
    callback_model_checkpoint("jena_dense.keras",
                              save_best_only=TRUE)
)

model %>%
  compile(optimizer = "rmsprop",
          loss = "mse",
          metrics = "mae")

history <- model %>%
  fit(
    train_dataset,
    epochs = 10,
    validation_data = val_dataset,
    callbacks = callbacks
  )


model <- load_model_tf("jena_dense.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)["mae"])


# **Plotting results**

# In[ ]:

#
# import matplotlib.pyplot as plt
# loss = history.history["mae"]
# val_loss = history.history["val_mae"]
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, "bo", label="Training MAE")
# plt.plot(epochs, val_loss, "b", label="Validation MAE")
# plt.title("Training and validation MAE")
# plt.legend()
# plt.show()


plot(history, metrics = "mae")


# ### Let's try a 1D convolutional model

# In[ ]:

#
# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.Conv1D(8, 24, activation="relu")(inputs)
# x = layers.MaxPooling1D(2)(x)
# x = layers.Conv1D(8, 12, activation="relu")(x)
# x = layers.MaxPooling1D(2)(x)
# x = layers.Conv1D(8, 6, activation="relu")(x)
# x = layers.GlobalAveragePooling1D()(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("jena_conv.keras",
#                                     save_best_only=True)
# ]
# model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
# history = model.fit(train_dataset,
#                     epochs=10,
#                     validation_data=val_dataset,
#                     callbacks=callbacks)
#
# model = keras.models.load_model("jena_conv.keras")
# print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

inputs <- layer_input(shape = c(sequence_length, ncol(raw_data)))
outputs <- inputs %>%
  layer_conv_1d(8, 24, activation="relu") %>%
  layer_max_pooling_1d(2) %>%
  layer_conv_1d(8, 12, activation="relu") %>%
  layer_max_pooling_1d(2) %>%
  layer_conv_1d(8, 6, activation="relu") %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks <- list(
    callback_model_checkpoint("jena_conv.keras", save_best_only=TRUE)
)

model %>% compile(optimizer="rmsprop", loss="mse", metrics="mae")
history <- model %>% fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)

model <- load_model_tf("jena_conv.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)["mae"])


# ### A first recurrent baseline

# **A simple LSTM-based model**

# In[ ]:

#
# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.LSTM(16)(inputs)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("jena_lstm.keras",
#                                     save_best_only=True)
# ]
# model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
# history = model.fit(train_dataset,
#                     epochs=10,
#                     validation_data=val_dataset,
#                     callbacks=callbacks)
#
# model = keras.models.load_model("jena_lstm.keras")
# print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

inputs <- layer_input(shape = c(sequence_length, ncol(raw_data)))
outputs <- inputs %>%
  layer_lstm(16) %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks = list(callback_model_checkpoint("jena_lstm.keras",
                                           save_best_only = TRUE))

model %>% compile(optimizer = "rmsprop",
                  loss = "mse",
                  metrics = "mae")

history <- model %>% fit(
  train_dataset,
  epochs = 10,
  validation_data = val_dataset,
  callbacks = callbacks
)

model <- load_model_tf("jena_lstm.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)["mae"])


# ## Understanding recurrent neural networks

# **NumPy implementation of a simple RNN**

# In[ ]:

#
# import numpy as np
# timesteps = 100
# input_features = 32
# output_features = 64
# inputs = np.random.random((timesteps, input_features))
# state_t = np.zeros((output_features,))
# W = np.random.random((output_features, input_features))
# U = np.random.random((output_features, output_features))
# b = np.random.random((output_features,))
# successive_outputs = []
# for input_t in inputs:
#     output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
#     successive_outputs.append(output_t)
#     state_t = output_t
# final_output_sequence = np.stack(successive_outputs, axis=0)

## In edition 1, there was some pseudo code preceding this that was quite nice
## and should be here too: something like
## state_t <- 0
## for (input_t in input_sequence) {
##   output_t <- activation(dot(W, input_t) + dot(U, state_t) + b)
##   state_t <- output_t
## }

runif_array <- function(...) array(runif(prod(...)), dim = c(...))

timesteps <- 100
input_features <- 32
output_features <- 64
inputs <- runif_array(timesteps, input_features)
state_t <- array(0, dim = output_features)
W <- runif_array(output_features, input_features)
U <- runif_array(output_features, output_features)
b <- runif_array(output_features, 1)
# successive_outputs <- vector("list", length = timesteps)
successive_outputs <- array(0, dim = c(timesteps, output_features))
for(ts in 1:timesteps) {
  input_t <- inputs[ts, ]
  output_t <- tanh((W %*% input_t) + (U %*% state_t) + b)
  # successive_outputs[[ts]] <- output_t
  successive_outputs[ts, ] <- output_t
  state_t <- output_t
}
# final_output_sequence <- do.call(rbind, successive_outputs)
final_output_sequence <- successive_outputs


# ### A recurrent layer in Keras

# **An RNN layer that can process sequences of any length**

# In[ ]:

#
# num_features = 14
# inputs = keras.Input(shape=(None, num_features))
# outputs = layers.SimpleRNN(16)(inputs)

num_features <- 14
inputs <- layer_input(shape = c(NA, num_features))
outputs <- inputs %>% layer_simple_rnn(16)


# **An RNN layer that returns only its last output step**

# In[ ]:

#
# num_features = 14
# steps = 120
# inputs = keras.Input(shape=(steps, num_features))
# outputs = layers.SimpleRNN(16, return_sequences=False)(inputs)
# print(outputs.shape)

num_features <- 14
steps <- 120
inputs <- layer_input(shape=c(steps, num_features))
outputs <- inputs %>% layer_simple_rnn(16, return_sequences=FALSE)
print(outputs$shape)


# **An RNN layer that returns its full output sequence**

# In[ ]:

#
# num_features = 14
# steps = 120
# inputs = keras.Input(shape=(steps, num_features))
# outputs = layers.SimpleRNN(16, return_sequences=True)(inputs)
# print(outputs.shape)

num_features <- 14
steps <- 120
inputs <- layer_input(shape=c(steps, num_features))
outputs <- inputs %>% layer_simple_rnn(16, return_sequences=TRUE)
print(outputs$shape)


# **Stacking RNN layers**

# In[ ]:

#
# inputs = keras.Input(shape=(steps, num_features))
# x = layers.SimpleRNN(16, return_sequences=True)(inputs)
# x = layers.SimpleRNN(16, return_sequences=True)(x)
# outputs = layers.SimpleRNN(16)(x)

inputs <- layer_input(shape = c(steps, num_features))
outputs <- inputs %>%
  layer_simple_rnn(16, return_sequences = TRUE) %>%
  layer_simple_rnn(16, return_sequences = TRUE) %>%
  layer_simple_rnn(16)


# ## Advanced use of recurrent neural networks

# ### Using recurrent dropout to fight overfitting

# **Training and evaluating a dropout-regularized LSTM**

# In[ ]:

#
# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("jena_lstm_dropout.keras",
#                                     save_best_only=True)
# ]
# model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
# history = model.fit(train_dataset,
#                     epochs=50,
#                     validation_data=val_dataset,
#                     callbacks=callbacks)


inputs <- layer_input(shape = c(sequence_length, ncol(raw_data)))
outputs <- inputs %>%
  layer_lstm(32, recurrent_dropout = 0.25) %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks = list(callback_model_checkpoint("jena_lstm_dropout.keras",
                                           save_best_only = TRUE))

model %>% compile(optimizer = "rmsprop",
                  loss = "mse",
                  metrics = "mae")

history <- model %>% fit(
  train_dataset,
  epochs = 50,
  validation_data = val_dataset,
  callbacks = callbacks
)


# In[ ]:

#
# inputs = keras.Input(shape=(sequence_length, num_features))
# x = layers.LSTM(32, recurrent_dropout=0.2, unroll=True)(inputs)

inputs <- layer_input(shape=c(sequence_length, num_features))
x <- inputs %>% layer_lstm(32, recurrent_dropout=0.2, unroll=TRUE)


# ### Stacking recurrent layers

# **Training and evaluating a dropout-regularized, stacked GRU model**

# In[ ]:

#
# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.GRU(32, recurrent_dropout=0.5, return_sequences=True)(inputs)
# x = layers.GRU(32, recurrent_dropout=0.5)(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("jena_stacked_gru_dropout.keras",
#                                     save_best_only=True)
# ]
# model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
# history = model.fit(train_dataset,
#                     epochs=50,
#                     validation_data=val_dataset,
#                     callbacks=callbacks)
# model = keras.models.load_model("jena_stacked_gru_dropout.keras")
# print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
#

inputs <- layer_input(shape = c(sequence_length, ncol(raw_data)))
outputs <- inputs %>%
  layer_gru(32, recurrent_dropout = 0.5, return_sequences = TRUE) %>%
  layer_gru(32, recurrent_dropout = 0.5) %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
model <- keras_model(inputs, outputs)

callbacks = list(
  callback_model_checkpoint("jena_stacked_gru_dropout.keras",
                            save_best_only = TRUE)
)

model %>% compile(optimizer = "rmsprop",
                  loss = "mse",
                  metrics = "mae")

history <- model %>% fit(
  train_dataset,
  epochs = 50,
  validation_data = val_dataset,
  callbacks = callbacks
)

model <- load_model_tf("jena_stacked_gru_dropout.keras")
sprintf("Test MAE: %.2f", evaluate(model, test_dataset)["mae"])


# ### Using bidirectional RNNs

# **Training and evaluating a bidirectional LSTM**

# In[ ]:


# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.Bidirectional(layers.LSTM(16))(inputs)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)
#
# model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
# history = model.fit(train_dataset,
#                     epochs=10,
#                     validation_data=val_dataset)
#
inputs <- layer_input(shape = c(sequence_length, ncol(raw_data)))
outputs <- inputs %>%
  bidirectional(layer_lstm(units = 16)) %>%
  layer_dense(1)

model <- keras_model(inputs, outputs)

model %>% compile(optimizer = "rmsprop",
                  loss = "mse",
                  metrics = "mae")
history <- model %>% fit(train_dataset,
                         epochs = 10,
                         validation_data = val_dataset)


# ### Going even further

# ## Summary
