#!/usr/bin/env python
# coding: utf-8

# This is a companion notebook for the book [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.6.

# # Generative deep learning

# ## Text generation

# ### A brief history of generative deep learning for sequence generation

# ### How do you generate sequence data?

# ### The importance of the sampling strategy

# **Reweighting a probability distribution to a different temperature**

# In[ ]:

#
# import numpy as np
# def reweight_distribution(original_distribution, temperature=0.5):
#     distribution = np.log(original_distribution) / temperature
#     distribution = np.exp(distribution)
#     return distribution / np.sum(distribution)

reweight_distribution <-
  function(original_distribution, temperature = 0.5) {
    distribution <- log(original_distribution) / temperature
    distribution <- exp(distribution)
    distribution / sum(distribution)
  }


# ### Implementing text generation with Keras

# #### Preparing the data

# **Downloading and uncompressing the IMDB movie reviews dataset**

# In[ ]:

#
# get_ipython().system('wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
# get_ipython().system('tar -xf aclImdb_v1.tar.gz')

if(FALSE) {
system('wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
system('tar -xf aclImdb_v1.tar.gz')
}


# **Creating a dataset from text files (one file = one sample)**

# In[ ]:

#
# import tensorflow as tf
# from tensorflow import keras
# dataset = keras.utils.text_dataset_from_directory(
#     directory="aclImdb", label_mode=None, batch_size=256)
# dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))

library(tensorflow)
library(tfdatasets)
library(keras)
dataset <- text_dataset_from_directory(directory = "aclImdb",
                                       label_mode = NULL,
                                       batch_size = 256)
dataset <- dataset %>%
  dataset_map( ~ tf$strings$regex_replace(.x, "<br />", " "))


# **Preparing a `TextVectorization` layer**

# In[ ]:

#
# from tensorflow.keras.layers import TextVectorization
#
# sequence_length = 100
# vocab_size = 15000
# text_vectorization = TextVectorization(
#     max_tokens=vocab_size,
#     output_mode="int",
#     output_sequence_length=sequence_length,
# )
# text_vectorization.adapt(dataset)


sequence_length <- 100
vocab_size <- 15000
text_vectorization <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_mode = "int",
  output_sequence_length = sequence_length
)
adapt(text_vectorization, dataset)


# **Setting up a language modeling dataset**

# In[ ]:

#
# def prepare_lm_dataset(text_batch):
#     vectorized_sequences = text_vectorization(text_batch)
#     x = vectorized_sequences[:, :-1]
#     y = vectorized_sequences[:, 1:]
#     return x, y
#
# lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)

prepare_lm_dataset <- function(text_batch) {
  vectorized_sequences <- text_vectorization(text_batch)
  x <- vectorized_sequences[, NULL:-2]
  y <- vectorized_sequences[, 2:NULL]
  list(x, y)
}

lm_dataset <- dataset %>%
  dataset_map(prepare_lm_dataset, num_parallel_calls = 4)


# #### A Transformer-based sequence-to-sequence model

# In[ ]:

#
# import tensorflow as tf
# from tensorflow.keras import layers
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
#         config = super(PositionalEmbedding, self).get_config()
#         config.update({
#             "output_dim": self.output_dim,
#             "sequence_length": self.sequence_length,
#             "input_dim": self.input_dim,
#         })
#         return config

# TODO: consider bringing back KerasLayer, as an alias for keras$layers$Layer
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
    length <- tf$shape(inputs)[-1] # tail(, 1)
    positions <- tf$range(start = 0L, limit = length, delta = 1L)
    embedded_tokens <- self$token_embeddings(inputs)
    embedded_positions <- self$position_embeddings(positions)
    embedded_tokens + embedded_positions
  }

  compute_mask <- function(inputs, mask = NULL) {
    inputs != 0L
  }

  get_config <- function() {
    config <- super$get_config()
    config$output_dim <-  self$output_dim
    config$sequence_length <-  self$sequence_length
    config$input_dim <-  self$input_dim
    config
  }
}

layer_positional_embedding <- create_layer_wrapper(PositionalEmbedding)

#
# class TransformerDecoder(layers.Layer):
#     def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.dense_dim = dense_dim
#         self.num_heads = num_heads
#         self.attention_1 = layers.MultiHeadAttention(
#           num_heads=num_heads, key_dim=embed_dim)
#         self.attention_2 = layers.MultiHeadAttention(
#           num_heads=num_heads, key_dim=embed_dim)
#         self.dense_proj = keras.Sequential(
#             [layers.Dense(dense_dim, activation="relu"),
#              layers.Dense(embed_dim),]
#         )
#         self.layernorm_1 = layers.LayerNormalization()
#         self.layernorm_2 = layers.LayerNormalization()
#         self.layernorm_3 = layers.LayerNormalization()
#         self.supports_masking = True
#
#     def get_config(self):
#         config = super(TransformerDecoder, self).get_config()
#         config.update({
#             "embed_dim": self.embed_dim,
#             "num_heads": self.num_heads,
#             "dense_dim": self.dense_dim,
#         })
#         return config
#
#     def get_causal_attention_mask(self, inputs):
#         input_shape = tf.shape(inputs)
#         batch_size, sequence_length = input_shape[0], input_shape[1]
#         i = tf.range(sequence_length)[:, tf.newaxis]
#         j = tf.range(sequence_length)
#         mask = tf.cast(i >= j, dtype="int32")
#         mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
#         mult = tf.concat(
#             [tf.expand_dims(batch_size, -1),
#              tf.constant([1, 1], dtype=tf.int32)], axis=0)
#         return tf.tile(mask, mult)
#
#     def call(self, inputs, encoder_outputs, mask=None):
#         causal_mask = self.get_causal_attention_mask(inputs)
#         if mask is not None:
#             padding_mask = tf.cast(
#                 mask[:, tf.newaxis, :], dtype="int32")
#             padding_mask = tf.minimum(padding_mask, causal_mask)
#         attention_output_1 = self.attention_1(
#             query=inputs,
#             value=inputs,
#             key=inputs,
#             attention_mask=causal_mask)
#         attention_output_1 = self.layernorm_1(inputs + attention_output_1)
#         attention_output_2 = self.attention_2(
#             query=attention_output_1,
#             value=encoder_outputs,
#             key=encoder_outputs,
#             attention_mask=padding_mask,
#         )
#         attention_output_2 = self.layernorm_2(
#             attention_output_1 + attention_output_2)
#         proj_output = self.dense_proj(attention_output_2)
#         return self.layernorm_3(attention_output_2 + proj_output)

TransformerDecoder(keras$layers$Layer) %py_class% {
  initialize <- function(embed_dim, dense_dim, num_heads, ...) {
    super$initialize(...)
    self$embed_dim <- as.integer(embed_dim)
    self$dense_dim <- as.integer(dense_dim)
    self$num_heads <- as.integer(num_heads)
    self$attention_1 <-
      layer_multi_head_attention(num_heads = num_heads, key_dim = embed_dim)
    self$attention_2 <-
      layer_multi_head_attention(num_heads = num_heads, key_dim = embed_dim)
    self$dense_proj <- keras_model_sequential() %>%
      layer_dense(dense_dim, activation = "relu") %>%
      layer_dense(embed_dim)
    self$layernorm_1 <- layer_layer_normalization()
    self$layernorm_2 <- layer_layer_normalization()
    self$layernorm_3 <- layer_layer_normalization()
    self$supports_masking <- TRUE
  }

  get_config <- function() {
    config <- super$get_config()
    config$embed_dim <- self$embed_dim
    config$num_heads <- self$num_heads
    config$dense_dim <- self$dense_dim
    config
  }

  get_causal_attention_mask <- function(inputs) {
    # browser()
    input_shape <- tf$shape(inputs)
    c(batch_size, sequence_length) %<-% list(input_shape[1], input_shape[2])
    i <- tf$range(sequence_length)[, tf$newaxis]
    j <- tf$range(sequence_length)
    mask = tf$cast(i >= j, dtype = "int32")
    mask = tf$reshape(mask, list(1L, input_shape[2], input_shape[2]))
    mult = tf$concat(list(
      tf$expand_dims(batch_size, -1L),
      tf$constant(c(1L, 1L), dtype = tf$int32)
    ), axis = 0L)
    tf$tile(mask, mult)
  }

  call <- function(inputs, encoder_outputs, mask = NULL) {
    causal_mask <- self$get_causal_attention_mask(inputs)
    if (!is.null(mask)) {
      padding_mask <- tf$cast(mask[, tf$newaxis,], dtype = "int32")
      padding_mask <- tf$minimum(padding_mask, causal_mask)
    }
    attention_output_1 <-
      self$attention_1(
        query = inputs,
        value = inputs,
        key = inputs,
        attention_mask = causal_mask
      )
    attention_output_1 <-
      self$layernorm_1(inputs + attention_output_1)

    ## perhaps this chain is more readable?
    # attention_output_1 <-
    #   self$attention_1(query = inputs, value = inputs,
    #                    key = inputs, attention_mask = causal_mask) %>%
    #   `+`(inputs) %>%
    #   self$layernorm_1()

    attention_output_2 <-
      self$attention_2(
        query = attention_output_1,
        value = encoder_outputs,
        key = encoder_outputs,
        attention_mask = padding_mask
      )
    attention_output_2 <-
      self$layernorm_2(attention_output_1 + attention_output_2)
    proj_output <- self$dense_proj(attention_output_2)
    self$layernorm_3(attention_output_2 + proj_output)
  }
}

layer_transformer_decoder <- create_layer_wrapper(TransformerDecoder)


# **A simple Transformer-based language model**

# In[ ]:

#
# from tensorflow.keras import layers
# embed_dim = 256
# latent_dim = 2048
# num_heads = 2
#
# inputs = keras.Input(shape=(None,), dtype="int64")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
# x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x)
# outputs = layers.Dense(vocab_size, activation="softmax")(x)
# model = keras.Model(inputs, outputs)
# model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")

embed_dim <- 256
latent_dim <- 2048
num_heads <- 2

decoder <- TransformerDecoder(embed_dim, latent_dim, num_heads)

inputs <- layer_input(shape = list(NULL), dtype = "int64")
embedded <- inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim)
# TODO: still need a good way to pass multiple args to a layer's call() method
# in functions that use create_layer()
outputs <- decoder(embedded, embedded) %>%
  layer_dense(vocab_size, activation = "softmax")
model <- keras_model(inputs, outputs)
model %>% compile(loss = "sparse_categorical_crossentropy",
                  optimizer = "rmsprop")


# ### A text-generation callback with variable-temperature sampling

# **The text-generation callback**

# In[ ]:

#
# import numpy as np
#
# tokens_index = dict(enumerate(text_vectorization.get_vocabulary()))
#
# def sample_next(predictions, temperature=1.0):
#     predictions = np.asarray(predictions).astype("float64")
#     predictions = np.log(predictions) / temperature
#     exp_preds = np.exp(predictions)
#     predictions = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, predictions, 1)
#     return np.argmax(probas)
#
# class TextGenerator(keras.callbacks.Callback):
#     def __init__(self,
#                  prompt,
#                  generate_length,
#                  model_input_length,
#                  temperatures=(1.,),
#                  print_freq=1):
#         self.prompt = prompt
#         self.generate_length = generate_length
#         self.model_input_length = model_input_length
#         self.temperatures = temperatures
#         self.print_freq = print_freq
#
#     def on_epoch_end(self, epoch, logs=None):
#         if (epoch + 1) % self.print_freq != 0:
#             return
#         for temperature in self.temperatures:
#             print("== Generating with temperature", temperature)
#             sentence = self.prompt
#             for i in range(self.generate_length):
#                 tokenized_sentence = text_vectorization([sentence])
#                 predictions = self.model(tokenized_sentence)
#                 next_token = sample_next(predictions[0, i, :])
#                 sampled_token = tokens_index[next_token]
#                 sentence += " " + sampled_token
#             print(sentence)
#
# prompt = "This movie"
# text_gen_callback = TextGenerator(
#     prompt,
#     generate_length=50,
#     model_input_length=sequence_length,
#     temperatures=(0.2, 0.5, 0.7, 1., 1.5))



tokens_index <- text_vectorization$get_vocabulary()

sample_next <- function(predictions, temperature = 1.0) {
  predictions <- log(predictions) / temperature
  exp_preds <- exp(predictions)
  predictions <- exp_preds / sum(exp_preds)
  probas <- sample.int(length(predictions), 1, prob = predictions)
  # can also do which.max(rmultinom(1, size = 1, prob = predictions))
  probas
}

TextGenerator(keras$callbacks$Callback) %py_class% {
  initialize <- function(prompt,
                         generate_length,
                         model_input_length,
                         temperatures = c(1.),
                         print_freq = 1) {
    self$prompt <- prompt
    self$generate_length <- generate_length
    self$model_input_length <- model_input_length
    self$temperatures <- temperatures
    self$print_freq <- as.integer(print_freq)
  }

  on_epoch_end <- function(epoch, logs = NULL) {
    # if (!((epoch + 1) %% self$print_freq))
    if ( ((epoch + 1) %% self$print_freq) != 0 )
      return()
    for (temperature in self$temperatures) {
      cat("== Generating with temperature", temperature, "\n")
      sentence <- self$prompt
      for (i in seq(self$generate_length)) {
        tokenized_sentence <-
          text_vectorization(array(sentence, dim = c(1, 1)))
        predictions <- self$model(tokenized_sentence)
        next_token <- sample_next(as.array(predictions[1, i, ]))
        sampled_token <- tokens_index[next_token]
        # consider using stringi::`%s+%`? or stringr::str_c()?
        sentence <- paste(sentence, sampled_token)
      }
      cat(sentence, "\n")
    }
  }
}



prompt <- "This movie"
text_gen_callback <- TextGenerator(
  prompt,
  generate_length = 50,
  model_input_length = sequence_length,
  temperatures = c(0.2, 0.5, 0.7, 1., 1.5)
)

# text_gen_callback$on_epoch_end(1) # test outside of fit
# ## may need to add `model` to self if calling the callback before
# ## it's been initialized in `fit()`
# text_gen_callback$model <- model # otherwise available after calling fit()

# **Fitting the language model**

# In[ ]:


# model.fit(lm_dataset, epochs=200, callbacks=[text_gen_callback])


model %>%
  fit(lm_dataset, epochs = 200, callbacks = list(text_gen_callback))

# ### Wrapping up
