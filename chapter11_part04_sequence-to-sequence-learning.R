#!/usr/bin/env R
# coding: utf-8

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.7.

library(keras)
# Sys.setenv("CUDA_VISIBLE_DEVICES"="")


# ## Beyond text classification: Sequence-to-sequence learning

# ### A machine translation example

# In[ ]:

#
# get_ipython().system('wget http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip')
# get_ipython().system('unzip -q spa-eng.zip')


if(FALSE)
{
system('wget http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip')
system('unzip -q spa-eng.zip')
}


# In[ ]:

#
# text_file = "spa-eng/spa.txt"
# with open(text_file) as f:
#     lines = f.read().split("\n")[:-1]
# text_pairs = []
# for line in lines:
#     english, spanish = line.split("\t")
#     spanish = "[start] " + spanish + " [end]"
#     text_pairs.append((english, spanish))

library(dplyr, warn.conflicts = FALSE)

text_file <- "spa-eng/spa.txt"
text_pairs <- tibble(lines = readLines(text_file)) %>%
  tidyr::separate(lines, into = c("english", "spanish"), sep = "\t") %>%
  mutate(spanish = paste("[start]", spanish, "[end]"))
# text_pairs <- lapply(1:nrow(text_pairs_df), function(r) unlist(text_pairs_df[r,]))
text_pairs

# In[ ]:


# import random
# print(random.choice(text_pairs))
text_pairs %>% slice_sample(n = 5)


# In[ ]:

#
# import random
# random.shuffle(text_pairs)
# num_val_samples = int(0.15 * len(text_pairs))
# num_train_samples = len(text_pairs) - 2 * num_val_samples
# train_pairs = text_pairs[:num_train_samples]
# val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
# test_pairs = text_pairs[num_train_samples + num_val_samples:]

text_pairs <- text_pairs %>% slice(sample(1:n())) # shuffle

num_val_samples <- round(0.15 * nrow(text_pairs))
num_train_samples <- nrow(text_pairs) - 2 * num_val_samples
train_pairs <- text_pairs[1:num_train_samples,]
val_pairs <- text_pairs[num_train_samples:(num_train_samples + num_val_samples),]
test_pairs <- text_pairs[(num_train_samples + num_val_samples):nrow(text_pairs),]


# **Vectorizing the English and Spanish text pairs**

# In[ ]:


# import tensorflow as tf
# import string
# import re
#
# strip_chars = string.punctuation + "¿"
# strip_chars = strip_chars.replace("[", "")
# strip_chars = strip_chars.replace("]", "")
#
# def custom_standardization(input_string):
#     lowercase = tf.strings.lower(input_string)
#     return tf.strings.regex_replace(
#         lowercase, f"[{re.escape(strip_chars)}]", "")
#
# vocab_size = 15000
# sequence_length = 20
#
# source_vectorization = layers.TextVectorization(
#     max_tokens=vocab_size,
#     output_mode="int",
#     output_sequence_length=sequence_length,
# )
# target_vectorization = layers.TextVectorization(
#     max_tokens=vocab_size,
#     output_mode="int",
#     output_sequence_length=sequence_length + 1,
#     standardize=custom_standardization,
# )
# train_english_texts = [pair[0] for pair in train_pairs]
# train_spanish_texts = [pair[1] for pair in train_pairs]
# source_vectorization.adapt(train_english_texts)
# target_vectorization.adapt(train_spanish_texts)


# [[:punct:]] except omits "[" and "]" and adds "¿"
strip_chars <- r"(!"#$%&'()*+,-./:;<=>?@\^_`{|}~)"

library(tensorflow)
custom_standardization <- function(input_string) {
  # Note, this time we're using tensor operations. This allows the function
  # to be traced into a tensorflow graph, which can typically run much faster
  lowercase = tf$strings$lower(input_string)
  tf$strings$regex_replace(lowercase, rex::escape(strip_chars), "")
}
## Note, tensorflow regex has subtle differences from R regex engine.
## Consult the source documentation liberally:
## https://github.com/google/re2/wiki/Syntax
library(keras)
vocab_size <- 15000
sequence_length <- 20

source_vectorization <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_mode = "int",
  output_sequence_length = sequence_length
)

target_vectorization <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_mode = "int",
  output_sequence_length = sequence_length + 1,
  standardize = custom_standardization
)

adapt(source_vectorization, train_pairs$english)
adapt(target_vectorization, train_pairs$spanish)


# **Preparing datasets for the translation task**

# In[ ]:


# batch_size = 64
#
# def format_dataset(eng, spa):
#     eng = source_vectorization(eng)
#     spa = target_vectorization(spa)
#     return ({
#         "english": eng,
#         "spanish": spa[:, :-1],
#     }, spa[:, 1:])
#
# def make_dataset(pairs):
#     eng_texts, spa_texts = zip(*pairs)
#     eng_texts = list(eng_texts)
#     spa_texts = list(spa_texts)
#     dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.map(format_dataset, num_parallel_calls=4)
#     return dataset.shuffle(2048).prefetch(16).cache()
#
# train_ds = make_dataset(train_pairs)
# val_ds = make_dataset(val_pairs)

batch_size <- 64

format_dataset <- function(eng, spa) {
  eng <- source_vectorization(eng)
  spa <- target_vectorization(spa)
  # introduce a keras::k_slice()?
  # set options(tensorflow.extract.warn_negatives_pythonic = FALSE)?
  # This -1 below throws a warning
  # maybe stop throwing a warning since eager is the default now?
  inputs <- list(english =  eng, spanish =  spa[, NULL:-2]) # drop last column
  targets <- spa[, 2:NULL] # drop first column
  list(inputs, targets)
}


library(tfdatasets)
make_dataset <- function(pairs) {
  tensor_slices_dataset(list(pairs$english, pairs$spanish)) %>%
    dataset_batch(batch_size) %>%
    dataset_map(format_dataset, num_parallel_calls = 4) %>%
    dataset_shuffle(2048) %>%
    dataset_prefetch(16) %>%
    dataset_cache()
}
train_ds <-  make_dataset(train_pairs)
val_ds <- make_dataset(val_pairs)


# In[ ]:

#
# for inputs, targets in train_ds.take(1):
#     print(f"inputs['english'].shape: {inputs['english'].shape}")
#     print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
#     print(f"targets.shape: {targets.shape}")

c(inputs, targets) %<-% iter_next(as_iterator(dataset_take(train_ds, 1)))
inputs$english$shape
inputs$spanish$shape
targets$shape



# ### Sequence-to-sequence learning with RNNs

# **GRU-based encoder**

# In[ ]:

#
# from tensorflow import keras
# from tensorflow.keras import layers
#
# embed_dim = 256
# latent_dim = 1024
#
# source = keras.Input(shape=(None,), dtype="int64", name="english")
# x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
# encoded_source = layers.Bidirectional(
#     layers.GRU(latent_dim), merge_mode="sum")(x)

embed_dim <- 256
latent_dim <- 1024

# rename layer_input() to keras_input()?
# rename keras_sequential_model() to keras_sequential()?
source <- layer_input(c(NA), dtype="int64", name="english")
encoded_source <- source %>%
  layer_embedding(vocab_size, embed_dim, mask_zero=TRUE) %>%
  bidirectional(layer_gru(units = latent_dim), merge_mode="sum")


# **GRU-based decoder and the end-to-end model**

# In[ ]:

#
# past_target = keras.Input(shape=(None,), dtype="int64", name="spanish")
# x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
# decoder_gru = layers.GRU(latent_dim, return_sequences=True)
# x = decoder_gru(x, initial_state=encoded_source)
# x = layers.Dropout(0.5)(x)
# target_next_step = layers.Dense(vocab_size, activation="softmax")(x)
# seq2seq_rnn = keras.Model([source, past_target], target_next_step)

past_target <- layer_input(shape(NULL), dtype = "int64", name = "spanish")
decoder_gru <- layer_gru(units = latent_dim, return_sequences = TRUE)
target_next_step <- past_target %>%
  layer_embedding(vocab_size, embed_dim, mask_zero = TRUE) %>%
  decoder_gru(initial_state = encoded_source) %>%
  layer_dropout(0.5) %>%
  layer_dense(vocab_size, activation = "softmax")
seq2seq_rnn <- keras_model(inputs = list(source, past_target),
                           outputs = target_next_step)


# **Training our recurrent sequence-to-sequence model**

# In[ ]:

#
# seq2seq_rnn.compile(
#     optimizer="rmsprop",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"])
# seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds)

seq2seq_rnn %>% compile(optimizer = "rmsprop",
                        loss = "sparse_categorical_crossentropy",
                        metrics = "accuracy")
# seq2seq_rnn %>% fit(train_ds, epochs = 15, validation_data = val_ds)
seq2seq_rnn %>% fit(train_ds$take(10L), epochs = 1, validation_data = val_ds$take(20L))


# **Translating new sentences with our RNN encoder and decoder**

# In[ ]:

#
# import numpy as np
# spa_vocab = target_vectorization.get_vocabulary()
# spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
# max_decoded_sentence_length = 20
#
# def decode_sequence(input_sentence):
#     tokenized_input_sentence = source_vectorization([input_sentence])
#     decoded_sentence = "[start]"
#     for i in range(max_decoded_sentence_length):
#         tokenized_target_sentence = target_vectorization([decoded_sentence])
#         next_token_predictions = seq2seq_rnn.predict(
#             [tokenized_input_sentence, tokenized_target_sentence])
#         sampled_token_index = np.argmax(next_token_predictions[0, i, :])
#         sampled_token = spa_index_lookup[sampled_token_index]
#         decoded_sentence += " " + sampled_token
#         if sampled_token == "[end]":
#             break
#     return decoded_sentence
#
# test_eng_texts = [pair[0] for pair in test_pairs]
# for _ in range(20):
#     input_sentence = random.choice(test_eng_texts)
#     print("-")
#     print(input_sentence)
#     print(decode_sequence(input_sentence))

spa_vocab <- get_vocabulary(target_vectorization)
# spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length <- 20


decode_sequence <- function(input_sentence) {
  tokenized_input_sentence <-
    source_vectorization(array(input_sentence, dim = c(1, 1)))
  decoded_sentence <- "[start]"
  for (i in seq(max_decoded_sentence_length)) {
    tokenized_target_sentence <-
      target_vectorization(array(decoded_sentence, dim = c(1, 1)))
    next_token_predictions <- seq2seq_rnn %>%
      predict(list(tokenized_input_sentence,
                   tokenized_target_sentence))
    sampled_token_index <- which.max(next_token_predictions[1, i,])
    sampled_token <- spa_vocab[sampled_token_index]
    decoded_sentence <- paste(decoded_sentence, sampled_token)
    if (sampled_token == "[end]")
      break
  }
  decoded_sentence
}

for (i in seq(20)) {
    input_sentence <- sample(test_pairs$english, 1)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
}


# ### Sequence-to-sequence learning with Transformer

# #### The Transformer decoder

# **The `TransformerDecoder`**

# In[ ]:

#
# class TransformerDecoder(layers.Layer):
#     def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.dense_dim = dense_dim
#         self.num_heads = num_heads
#         self.attention_1 = layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=embed_dim)
#         self.attention_2 = layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=embed_dim)
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
#         config = super().get_config()
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
    self$embed_dim <- embed_dim
    self$dense_dim <- dense_dim
    self$num_heads <- num_heads
    self$attention_1 <- layer_multi_head_attention(num_heads = num_heads, key_dim = embed_dim)
    self$attention_2 <- layer_multi_head_attention(num_heads = num_heads, key_dim = embed_dim)
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
    modifyList(config, list(
      embed_dim = self.embed_dim,
      num_heads = self.num_heads,
      dense_dim = self.dense_dim
    ))
  }

  get_causal_attention_mask <- function(inputs) {
    # browser()
    input_shape = tf$shape(inputs)
    c(batch_size, sequence_length) %<-% list(input_shape[1], input_shape[2])
    i <- tf$range(sequence_length)[, tf$newaxis]
    j <- tf$range(sequence_length)
    mask <- tf$cast(i >= j, dtype = "int32")
    # can't use shape() here because input_shape can be a tracing tensor
    # and shape() only works eagerly
    # Need to decide if we should teach L integer notation here or
    # teach the backend wrappers.
    # Current though: L notation should be saved for a final advanced chapter
    # in a discussion about python and using the python tf api directly
    # and throughout the rest of the book, use the k_* wrappers as much as possible
    mask <- tf$reshape(mask, list(1L, input_shape[2], input_shape[2]))  # k_reshape()
    mult <- tf$concat(list(                                             # k_concatenate()
      tf$expand_dims(batch_size, -1L),                                  # k_expand_dims()
      tf$constant(c(1L, 1L), dtype = tf$int32)                          # as_tensor()
    ), axis = 0L)
    tf$tile(mask, mult)                                                 # k_tile()
  }

  call <- function(inputs, encoder_outputs, mask = NULL) {
    # browser()
    causal_mask <- self$get_causal_attention_mask(inputs)
    if (!is.null(mask)) {
      padding_mask <- tf$cast(mask[, tf$newaxis, ], dtype = "int32")
      padding_mask <- tf$minimum(padding_mask, causal_mask)
    }
    attention_output_1 <- self$attention_1(
      query = inputs,
      value = inputs,
      key = inputs,
      attention_mask = causal_mask
    )
    attention_output_1 = self$layernorm_1(inputs + attention_output_1)
    attention_output_2 = self$attention_2(
      query = attention_output_1,
      value = encoder_outputs,
      key = encoder_outputs,
      attention_mask = padding_mask
    )
    attention_output_2 = self$layernorm_2(attention_output_1 + attention_output_2)
    proj_output = self$dense_proj(attention_output_2)
    self$layernorm_3(attention_output_2 + proj_output)
  }

}


layer_transformer_decoder <- create_layer_wrapper(TransformerDecoder)

# #### Putting it all together: A Transformer for machine translation

# **PositionalEmbedding layer**

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
#         config = super(PositionalEmbedding, self).get_config()
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
    len <- tf$shape(inputs) %>% .[length(.)]  # take last
    # can also take last with %>%  .[-1]
    # doing %>% tail(1) is inconvenient because it doesn't drop dims,
    # returns a 1-d array instead of a scalar, which fails in tf.range()
    positions <- tf$range(start = 0, limit = len, delta = 1)
    embedded_tokens <- self$token_embeddings(inputs)
    embedded_positions <- self$position_embeddings(positions)
    embedded_tokens + embedded_positions
  }

  compute_mask <- function(inputs, mask = None) {
    inputs != 0
  }

  get_config <- function() {
    config <- super$get_config()
    modifyList(config, list(
      output_dim = self.output_dim,
      sequence_length = self.sequence_length,
      input_dim = self.input_dim
    ))
  }
}

layer_positional_embedding <- create_layer_wrapper(PositionalEmbedding)

# **End-to-end Transformer**

# In[ ]:

#
# embed_dim = 256
# dense_dim = 2048
# num_heads = 8
#
# encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
# encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
#
# decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
# x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
# x = layers.Dropout(0.5)(x)
# decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
# transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

## copy-pasted TransformerEncoder from chapter 11 part 3:

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


embed_dim <- 256
dense_dim <- 2048
num_heads <- 8

encoder_inputs <- layer_input(shape(NULL), dtype="int64", name="english")
encoder_outputs <- encoder_inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  layer_transformer_encoder(embed_dim, dense_dim, num_heads)

# The call to layer_transformer_decoder below is ugly right now because
# the call takes two arguments, and there is no way to pass through multiple args
# in create_layer(). Possible resolutions:
#
## Maybe modify create_layer() and all wrappers that use create_layer()
## to add new arg `extra_call_args` which takes a list of args that are
## spliced into the call of the layer instance? It would allow
## you to still use a %>% when piping into layers like TransformerDecoder below
## that take multiple args in their call. also, passing in custom masks,
## or setting training=FALSE for specific layer calls.
##
## Or change the signature of TransformerDecoder above so that the first argument
## is a list that you unpack using `%<-%` in the function body.

decoder_inputs <- layer_input(shape(NULL), dtype="int64", name="spanish")
decoder_outputs <- decoder_inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  {layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads)(., encoder_outputs)} %>%
  # update create_layer() to splat list into a call?
  layer_dropout(0.5) %>%
  layer_dense(vocab_size, activation="softmax")
transformer <- keras_model(list(encoder_inputs, decoder_inputs),
                           decoder_outputs)


# **Training the sequence-to-sequence Transformer**

# In[ ]:


# transformer.compile(
#     optimizer="rmsprop",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"])
# transformer.fit(train_ds, epochs=30, validation_data=val_ds)

transformer %>% compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics="accuracy")
# transformer %>% fit(train_ds, epochs=30, validation_data=val_ds)
transformer %>% fit(train_ds$take(6L), epochs=3, validation_data=val_ds$take(10L))


# **Translating new sentences with our Transformer model**

# In[ ]:

#
# import numpy as np
# spa_vocab = target_vectorization.get_vocabulary()
# spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
# max_decoded_sentence_length = 20
#
# def decode_sequence(input_sentence):
#     tokenized_input_sentence = source_vectorization([input_sentence])
#     decoded_sentence = "[start]"
#     for i in range(max_decoded_sentence_length):
#         tokenized_target_sentence = target_vectorization(
#             [decoded_sentence])[:, :-1]
#         predictions = transformer(
#             [tokenized_input_sentence, tokenized_target_sentence])
#         sampled_token_index = np.argmax(predictions[0, i, :])
#         sampled_token = spa_index_lookup[sampled_token_index]
#         decoded_sentence += " " + sampled_token
#         if sampled_token == "[end]":
#             break
#     return decoded_sentence
#
# test_eng_texts = [pair[0] for pair in test_pairs]
# for _ in range(20):
#     input_sentence = random.choice(test_eng_texts)
#     print("-")
#     print(input_sentence)
#     print(decode_sequence(input_sentence))

spa_vocab <- get_vocabulary(target_vectorization)
# spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length <- 20

decode_sequence <- function(input_sentence) {

    # Need list() here?
    tokenized_input_sentence <-
      source_vectorization(array(input_sentence, dim = c(1, 1)))
    decoded_sentence = "[start]"
    for (i in range(max_decoded_sentence_length)) {
      tokenized_target_sentence <-
        target_vectorization(array(decoded_sentence, dim = c(1, 1)))[, NULL:-2]
        predictions <- transformer(
            list(tokenized_input_sentence, tokenized_target_sentence))
        # use which.max() here? is this going to be graphed?
        sampled_token_index <- k_argmax(predictions[1, i, ])
        sampled_token <- spa_vocab[as.integer(sampled_token_index)] # token_index+1 ?
        decoded_sentence <- paste(decoded_sentence, sampled_token)
        if (sampled_token == "[end]")
            break
    }
    decoded_sentence
}


for (i in seq(20)) {
    input_sentence = sample(test_pairs$english, 1)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
}


# ## Summary
