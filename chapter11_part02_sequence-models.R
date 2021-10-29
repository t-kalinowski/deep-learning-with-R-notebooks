#!/usr/bin/env R

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.7.

# ### Processing words as a sequence: The sequence model approach

# #### A first practical example

# **Downloading the data**


# In[ ]:


# get_ipython().system('curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
# get_ipython().system('tar -xf aclImdb_v1.tar.gz')
# get_ipython().system('rm -r aclImdb/train/unsup')
if(FALSE) {
system('curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
system('tar -xf aclImdb_v1.tar.gz')
system('rm -r aclImdb/train/unsup')
}



# **Preparing the data**

# In[ ]:


# import os, pathlib, shutil, random
# from tensorflow import keras
# batch_size = 32
# base_dir = pathlib.Path("aclImdb")
# val_dir = base_dir / "val"
# train_dir = base_dir / "train"
# for category in ("neg", "pos"):
#     os.makedirs(val_dir / category)
#     files = os.listdir(train_dir / category)
#     random.Random(1337).shuffle(files)
#     num_val_samples = int(0.2 * len(files))
#     val_files = files[-num_val_samples:]
#     for fname in val_files:
#         shutil.move(train_dir / category / fname,
#                     val_dir / category / fname)
#
# train_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/train", batch_size=batch_size
# )
# val_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/val", batch_size=batch_size
# )
# test_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/test", batch_size=batch_size
# )
# text_only_train_ds = train_ds.map(lambda x, y: x)

library(fs)
library(tfdatasets)
library(keras)

batch_size <- 32 / 2 # decrease this if you encounter out-of-memory errors on a GPU
base_dir <- path("aclImdb")
val_dir <- path(base_dir) / "val"
train_dir <- path(base_dir) / "train"
if(FALSE) {
for (category in c("neg", "pos")) {
    dir_create(val_dir / category)
    files <- basename(dir_ls(train_dir / category))
    files <- sample(files, length(files)) # shuffle
    num_val_samples <- round(0.2 * length(files))
    val_files <- tail(files, num_val_samples)
    file_move(train_dir / category / val_files,
              val_dir / category / val_files)
}
}

train_ds = text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)
val_ds = text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)
test_ds = text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)
text_only_train_ds <- train_ds %>% dataset_map(function(x, y) x)


# **Preparing integer sequence datasets**

# In[ ]:


# from tensorflow.keras import layers
#
# max_length = 600
# max_tokens = 20000
# text_vectorization = layers.TextVectorization(
#     max_tokens=max_tokens,
#     output_mode="int",
#     output_sequence_length=max_length,
# )
# text_vectorization.adapt(text_only_train_ds)
#
# int_train_ds = train_ds.map(
#     lambda x, y: (text_vectorization(x), y)),
#     num_parallel_calls=4)
# int_val_ds = val_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)
# int_test_ds = test_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)


max_length <- 600
max_tokens <- 20000
text_vectorization <- layer_text_vectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length
)

adapt(text_vectorization, text_only_train_ds)

int_train_ds <- train_ds %>%
  dataset_map(~ list(text_vectorization(.x), .y), num_parallel_calls = 4)
int_val_ds <- val_ds %>%
  dataset_map(~ list(text_vectorization(.x), .y), num_parallel_calls = 4)
int_test_ds <- test_ds %>%
  dataset_map(~ list(text_vectorization(.x), .y), num_parallel_calls = 4)


# **A sequence model built on one-hot encoded vector sequences**

# In[ ]:


# import tensorflow as tf
# inputs = keras.Input(shape=(None,), dtype="int64")
# embedded = tf.one_hot(inputs, depth=max_tokens)
# x = layers.Bidirectional(layers.LSTM(32))(embedded)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# model.summary()

library(tensorflow)
inputs  <- layer_input(shape(NULL), dtype = "int64")
embedded <- k_one_hot(inputs, num_classes = max_tokens)
outputs <- embedded %>%
  bidirectional(layer_lstm(units = 32)) %>%
  layer_dropout(.5) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")
model


# **Training a first basic sequence model**

# In[ ]:


# callbacks = [
#     keras.callbacks.ModelCheckpoint("one_hot_bidir_lstm.keras",
#                                     save_best_only=True)
# ]
# model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
# model = keras.models.load_model("one_hot_bidir_lstm.keras")
# print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

callbacks = list(
  callback_model_checkpoint("one_hot_bidir_lstm.keras",
                            save_best_only = TRUE))
model %>% fit(int_train_ds, validation_data = int_val_ds,
              epochs = 10, callbacks = callbacks)


model <- load_model_tf("one_hot_bidir_lstm.keras")
sprintf("Test acc: %.3f", evaluate(model, int_test_ds)["accuracy"])


# #### Understanding word embeddings

# #### Learning word embeddings with the Embedding layer

# **Instantiating an `Embedding` layer**

# In[ ]:


# embedding_layer = layers.Embedding(input_dim=max_tokens, output_dim=256)
embedding_layer = layer_embedding(input_dim = max_tokens, output_dim = 256)


# **Model that uses an `Embedding` layer trained from scratch**

# In[ ]:


# inputs = keras.Input(shape=(None,), dtype="int64")
# embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
# x = layers.Bidirectional(layers.LSTM(32))(embedded)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# model.summary()
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("embeddings_bidir_gru.keras",
#                                     save_best_only=True)
# ]
# model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
# model = keras.models.load_model("embeddings_bidir_gru.keras")
# print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

inputs <- layer_input(shape(NULL), dtype="int64")
embedded <- inputs %>% layer_embedding(input_dim=max_tokens, output_dim=256)
outputs <- embedded %>%
  bidirectional(layer_lstm(units = 32)) %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation="sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")
model

callbacks = list(
  callback_model_checkpoint("embeddings_bidir_gru.keras",
                                           save_best_only = TRUE)
)

model %>% fit(int_train_ds, validation_data = int_val_ds,
              epochs = 10, callbacks = callbacks)
model <- load_model_tf("embeddings_bidir_gru.keras")
sprintf("Test acc: %.3f", evaluate(model, int_test_ds)["accuracy"])


# #### Understanding padding and masking

# **Using an `Embedding` layer with masking enabled**

# In[ ]:


# inputs = keras.Input(shape=(None,), dtype="int64")
# embedded = layers.Embedding(
#     input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
# x = layers.Bidirectional(layers.LSTM(32))(embedded)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# model.summary()
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("embeddings_bidir_gru_with_masking.keras",
#                                     save_best_only=True)
# ]
# model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
# model = keras.models.load_model("embeddings_bidir_gru_with_masking.keras")
# print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

inputs = layer_input(shape(NA), dtype="int64")
embedded <- inputs %>% layer_embedding(
    input_dim=max_tokens, output_dim=256, mask_zero=TRUE)
outputs <- embedded %>%
  bidirectional(layer_lstm(units = 32)) %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation="sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics="accuracy")
model

callbacks = list(
  callback_model_checkpoint("embeddings_bidir_gru_with_masking.keras",
                            save_best_only = TRUE)
)


model %>% fit(int_train_ds, validation_data=int_val_ds,
              epochs=10, callbacks=callbacks)
model <- load_model_tf("embeddings_bidir_gru_with_masking.keras")
sprintf("Test acc: %.3f", evaluate(model, int_test_ds)["accuracy"])


# #### Using pretrained word embeddings

# In[ ]:


# get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')
# get_ipython().system('unzip -q glove.6B.zip')
if(FALSE) {
system('wget http://nlp.stanford.edu/data/glove.6B.zip')
system('unzip -q glove.6B.zip')
}


# **Parsing the GloVe word-embeddings file**

# In[ ]:


# import numpy as np
# path_to_glove_file = "glove.6B.100d.txt"
#
# embeddings_index = {}
# with open(path_to_glove_file) as f:
#     for line in f:
#         word, coefs = line.split(maxsplit=1)
#         coefs = np.fromstring(coefs, "f", sep=" ")
#         embeddings_index[word] = coefs
#
# print(f"Found {len(embeddings_index)} word vectors.")
path_to_glove_file = "glove.6B.100d.txt"

if(FALSE) {
df <- readr::read_table(path_to_glove_file, col_names = FALSE)
embeddings_index <- lapply(1:nrow(df), function(r) as.numeric(df[r, -1]))
names(embeddings_index) <- df[[1]]
}


lines <- readLines(path_to_glove_file) %>% strsplit(" ", fixed = TRUE)
embeddings_index <- map(lines, function(line) as.numeric(line[-1]))
names(embeddings_index) <- map_chr(lines, function(line) line[[1]])
rm(lines) # clear from memory

str(embeddings_index, list.len = 10)


# library(purrr)
# words <- df[[1]]
# coefs <- as.matrix(unname(df)[-1])
# rownames(coefs) <- words
#
# words %>% asplit(coefs)
#   set_names() %>%




# **Preparing the GloVe word-embeddings matrix**

# In[ ]:


# embedding_dim = 100
#
# vocabulary = text_vectorization.get_vocabulary()
# word_index = dict(zip(vocabulary, range(len(vocabulary))))
#
# embedding_matrix = np.zeros((max_tokens, embedding_dim))
# for word, i in word_index.items():
#     if i < max_tokens:
#         embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
embedding_dim <- 100

vocabulary <- text_vectorization %>% get_vocabulary()
embedding_matrix <- array(0, dim = c(max_tokens, embedding_dim))
for(i in seq_along(head(vocabulary, embedding_dim))) {
  word <- vocabulary[[i]]
  embedding_vector <- embeddings_index[[word]]

  if (!is.null(embedding_vector))
    embedding_matrix[i, ] <- embedding_vector
}


# In[ ]:


# embedding_layer = layers.Embedding(
#     max_tokens,
#     embedding_dim,
#     embeddings_initializer=keras.initializers.Constant(embedding_matrix),
#     trainable=False,
#     mask_zero=True,
# )

embedding_layer <- layer_embedding(
  input_dim = max_tokens,
  output_dim = embedding_dim,
  embeddings_initializer = initializer_constant(embedding_matrix),
  trainable = FALSE,
  mask_zero = TRUE
)


# **Model that uses a pretrained Embedding layer**

# In[ ]:


# inputs = keras.Input(shape=(None,), dtype="int64")
# embedded = embedding_layer(inputs)
# x = layers.Bidirectional(layers.LSTM(32))(embedded)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# model.summary()
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("glove_embeddings_sequence_model.keras",
#                                     save_best_only=True)
# ]
# model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
# model = keras.models.load_model("glove_embeddings_sequence_model.keras")
# print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

inputs <- layer_input(shape(NULL), dtype="int64")
embedded <- embedding_layer(inputs)
outputs <- embedded %>%
  bidirectional(layer_lstm(units = 32)) %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)

model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")
model

callbacks = list(
  callback_model_checkpoint("glove_embeddings_sequence_model.keras",
                            save_best_only = TRUE)
)


model %>% fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
model = load_model_tf("glove_embeddings_sequence_model.keras")
sprintf("Test acc: %.3f", evaluate(model, int_test_ds)["accuracy"])

