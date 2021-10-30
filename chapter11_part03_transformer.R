#!/usr/bin/env R

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.7.

# ## The Transformer architecture

# ### Understanding self-attention

# #### Generalized self-attention: the query-key-value model

# ### Multi-head attention

# ### The Transformer encoder

# **Getting the data**

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
library(keras)
batch_size <-  32
base_dir <- path("aclImdb")
val_dir <- base_dir / "val"
train_dir <- base_dir / "train"
if(FALSE)
for (category in c("neg", "pos")) {
    dir_create(val_dir / category)
    files = basename(dir_ls(train_dir / category))
    files <- sample(files, length(files))
    num_val_samples = round(0.2 * length(files))
    val_files = tail(files, num_val_samples)
    file_move(train_dir / category / val_files,
              val_dir / category / val_files)
}

train_ds <- text_dataset_from_directory("aclImdb/train", batch_size = batch_size)
val_ds <- text_dataset_from_directory("aclImdb/val", batch_size = batch_size)
test_ds <- text_dataset_from_directory("aclImdb/test", batch_size = batch_size)
text_only_train_ds <- train_ds %>% dataset_map(function(x, y) x)


# **Vectorizing the data**

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
#     lambda x, y: (text_vectorization(x), y),
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
  max_tokens = max_tokens,
  output_mode = "int",
  output_sequence_length = max_length
)
text_vectorization %>% adapt(text_only_train_ds)

int_train_ds = train_ds %>%
  dataset_map( ~ list(text_vectorization(.x), .y), num_parallel_calls = 4)
int_val_ds = val_ds %>%
  dataset_map( ~ list(text_vectorization(.x), .y), num_parallel_calls = 4)
int_test_ds = test_ds %>%
  dataset_map( ~ list(text_vectorization(.x), .y), num_parallel_calls = 4)


# **Transformer encoder implemented as a subclassed `Layer`**

# In[ ]:

#
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
#
# class TransformerEncoder(layers.Layer):
#     def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.dense_dim = dense_dim
#         self.num_heads = num_heads
#         self.attention = layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=embed_dim)
#         self.dense_proj = keras.Sequential(
#             [layers.Dense(dense_dim, activation="relu"),
#              layers.Dense(embed_dim),]
#         )
#         self.layernorm_1 = layers.LayerNormalization()
#         self.layernorm_2 = layers.LayerNormalization()
#
#     def call(self, inputs, mask=None):
#         if mask is not None:
#             mask = mask[:, tf.newaxis, :]
#         attention_output = self.attention(
#             inputs, inputs, attention_mask=mask)
#         proj_input = self.layernorm_1(inputs + attention_output)
#         proj_output = self.dense_proj(proj_input)
#         return self.layernorm_2(proj_input + proj_output)
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "embed_dim": self.embed_dim,
#             "num_heads": self.num_heads,
#             "dense_dim": self.dense_dim,
#         })
#         return config
library(tensorflow)
library(keras)

# 3 ways of writing the same thing,
# still need to pick which to go with in the book

TransformerEncoder(keras$layers$Layer) %py_class% {
  initialize <- function(embed_dim, dense_dim, num_heads, ...) {
    super$initialize(...)
    self$embed_dim <- embed_dim
    self$dense_dim <- dense_dim
    self$num_heads <- num_heads
    self$attention <-
      layer_multi_head_attention(num_heads = num_heads, key_dim =
                                   embed_dim)
    self$dense_proj  <- keras_model_sequential() %>%
      layer_dense(dense_dim, activation = "relu") %>%
      layer_dense(embed_dim)

    self$layernorm_1 <- layer_layer_normalization()
    self$layernorm_2 <- layer_layer_normalization()
  }

  call <- function(inputs, mask = NULL) {
    if (!is.null(mask))
      mask = mask[, tf$newaxis, ]
    attention_output <-
      self$attention(inputs, inputs, attention_mask = mask)
    proj_input <- self$layernorm_1(inputs + attention_output)
    proj_output <- self$dense_proj(proj_input)
    self$layernorm_2(proj_input + proj_output)
  }

  get_config <- function() {
    super$get_config() %>%
      modifyList(list(
        embed_dim = self$embed_dim,
        num_heads = self$num_heads,
        dense_dim = self$dense_dim
      ))
  }
}


TransformerEncoder <- R6::R6Class(
  classname = "TransformerEncoder",
  inherit = keras$layers$Layer,
  public = list(
    initialize = function(embed_dim, dense_dim, num_heads, ...) {
      super$initialize(...)
      self$embed_dim <- embed_dim
      self$dense_dim <- dense_dim
      self$num_heads <- num_heads
      self$attention <- layer_multi_head_attention(num_heads = num_heads,
                                                   key_dim = embed_dim)
      self$dense_proj  <- keras_model_sequential() %>%
        layer_dense(dense_dim, activation = "relu") %>%
        layer_dense(embed_dim)

      self$layernorm_1 <- layer_layer_normalization()
      self$layernorm_2 <- layer_layer_normalization()
    },

    call = function(inputs, mask = NULL) {
      if (!is.null(mask))
        mask = mask[, tf$newaxis,]
      attention_output <-
        self$attention(inputs, inputs, attention_mask = mask)
      proj_input <- self$layernorm_1(inputs + attention_output)
      proj_output <- self$dense_proj(proj_input)
      self$layernorm_2(proj_input + proj_output)
    },

    get_config = function() {
      super$get_config() %>%
        modifyList(
          list(
            embed_dim = self$embed_dim,
            num_heads = self$num_heads,
            dense_dim = self$dense_dim
          )
        )
    }
  )
)

layer_transformer_encoder <- create_layer_wrapper(TransformerEncoder)



layer_transformer_encoder <- Layer(
  classname = "TransformerEncoder",
  initialize = function(embed_dim, dense_dim, num_heads, ...) {
    super$initialize(...)
    self$embed_dim <- embed_dim
    self$dense_dim <- dense_dim
    self$num_heads <- num_heads
    self$attention <-
      layer_multi_head_attention(num_heads = num_heads,
                                 key_dim = embed_dim)
    self$dense_proj  <- keras_model_sequential() %>%
      layer_dense(dense_dim, activation = "relu") %>%
      layer_dense(embed_dim)

    self$layernorm_1 <- layer_layer_normalization()
    self$layernorm_2 <- layer_layer_normalization()
  },

  call = function(inputs, mask = NULL) {
    if (!is.null(mask))
      mask = mask[, tf$newaxis, ]
    attention_output <-
      self$attention(inputs, inputs, attention_mask = mask)
    proj_input <- self$layernorm_1(inputs + attention_output)
    proj_output <- self$dense_proj(proj_input)
    self$layernorm_2(proj_input + proj_output)
  },

  get_config = function() {
    super$get_config() %>%
      modifyList(
        list(
          embed_dim = self$embed_dim,
          num_heads = self$num_heads,
          dense_dim = self$dense_dim
        )
      )
  }
)



# **Using the Transformer encoder for text classification**

# In[ ]:

#
# vocab_size = 20000
# embed_dim = 256
# num_heads = 2
# dense_dim = 32
#
# inputs = keras.Input(shape=(None,), dtype="int64")
# x = layers.Embedding(vocab_size, embed_dim)(inputs)
# x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
# x = layers.GlobalMaxPooling1D()(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# model.summary()

vocab_size <- 20000
embed_dim <- 256
num_heads <- 2
dense_dim <- 32

inputs <- layer_input(shape(NULL), dtype = "int64")
outputs <- inputs %>%
  layer_embedding(vocab_size, embed_dim) %>%
  layer_transformer_encoder(embed_dim, dense_dim, num_heads) %>%
  layer_global_average_pooling_1d() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")
model <-  keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")
model

# **Training and evaluating the Transformer encoder based model**

# In[ ]:

#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("transformer_encoder.keras",
#                                     save_best_only=True)
# ]
# model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)
# model = keras.models.load_model(
#     "transformer_encoder.keras",
#     custom_objects={"TransformerEncoder": TransformerEncoder})
# print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

callbacks = list(
    callback_model_checkpoint("transformer_encoder.keras",
                                    save_best_only=TRUE)
)
model %>% fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)

model <- load_model_tf(
  "transformer_encoder.keras",
  custom_objects = list("TransformerEncoder" = TransformerEncoder)
)
sprintf("Test acc: %.3f", evaluate(model, int_test_ds)["accuracy"])


# #### Using positional encoding to re-inject order information

# **Implementing positional embedding as a subclassed layer**

# In[ ]:

#
# class PositionalEmbedding(layers.Layer):
#     def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.token_embeddings = layers.Embedding(
#             input_dim=input_dim, output_dim=output_dim)
#         self.position_embeddings = layers.Embedding(
#             input_dim=sequence_length, output_dim=output_dim)
#         self.sequence_length = sequence_length
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#     def call(self, inputs):
#         length = tf.shape(inputs)[-1]
#         positions = tf.range(start=0, limit=length, delta=1)
#         embedded_tokens = self.token_embeddings(inputs)
#         embedded_positions = self.position_embeddings(positions)
#         return embedded_tokens + embedded_positions
#
#     def compute_mask(self, inputs, mask=None):
#         return tf.math.not_equal(inputs, 0)
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "output_dim": self.output_dim,
#             "sequence_length": self.sequence_length,
#             "input_dim": self.input_dim,
#         })
#         return config


PositionalEmbedding(keras$layers$Layer) %py_class% {
  initialize <- function(sequence_length, input_dim, output_dim, ...) {
    super$initialize(...)
    self$token_embeddings <-
      layer_embedding(input_dim = input_dim, output_dim = output_dim)
    self$position_embeddings <-
      layer_embedding(input_dim = sequence_length, output_dim = output_dim)
    self$sequence_length <- sequence_length
    self$input_dim <- input_dim
    self$output_dim <- output_dim
  }

  call <- function(inputs) {
    length <- tail(tf$shape(inputs), 1)
    positions <- tf$range(start = 0, limit = length, delta = 1)
    embedded_tokens <- self$token_embeddings(inputs)
    embedded_positions <- self$position_embeddings(positions)
    embedded_tokens + embedded_positions
  }

  compute_mask <- function(inputs, mask = NULL)
    inputs != 0

  get_config <- function() {
    config <- super$get_config()
    config$output_dim <- self$output_dim
    config$sequence_length <- self$sequence_length
    config$input_dim <- self$input_dim
    config
  }
}

layer_positional_embedding <- create_layer_wrapper(PositionalEmbedding)


# #### Putting it all together: A text-classification Transformer

# **Combining the Transformer encoder with positional embedding**

# In[ ]:

#
# vocab_size = 20000
# sequence_length = 600
# embed_dim = 256
# num_heads = 2
# dense_dim = 32
#
# inputs = keras.Input(shape=(None,), dtype="int64")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
# x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
# x = layers.GlobalMaxPooling1D()(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# model.summary()
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("full_transformer_encoder.keras",
#                                     save_best_only=True)
# ]
# model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)
# model = keras.models.load_model(
#     "full_transformer_encoder.keras",
#     custom_objects={"TransformerEncoder": TransformerEncoder,
#                     "PositionalEmbedding": PositionalEmbedding})
# print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

vocab_size <- 20000
sequence_length <- 600
embed_dim <- 256
num_heads <- 2
dense_dim <- 32

inputs = layer_input(shape(NULL), dtype = "int64")
outputs <- inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  layer_transformer_encoder(embed_dim, dense_dim, num_heads) %>%
  layer_global_average_pooling_1d() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")
model

callbacks = list(
  callback_model_checkpoint("full_transformer_encoder.keras",
                            save_best_only = TRUE)
)

model %>% fit(
  int_train_ds,
  validation_data = int_val_ds,
  epochs = 20,
  callbacks = callbacks
)


model <- load_model_tf(
  "full_transformer_encoder.keras",
  custom_objects = list(
    "TransformerEncoder" = TransformerEncoder,
    "PositionalEmbedding" = PositionalEmbedding
  )
)
sprintf("Test acc: %.3f", evaluate(model, int_test_ds)["accuracy"])


# ### When to use sequence models over bag-of-words models?
