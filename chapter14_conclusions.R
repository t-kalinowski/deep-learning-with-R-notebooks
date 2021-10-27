#!/usr/bin/env Rscript


# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.7.

# # Conclusions

# ## Key concepts in review

# ### Various approaches to AI

# ### What makes deep learning special within the field of machine learning

# ### How to think about deep learning

# ### Key enabling technologies

# ### The universal machine-learning workflow

# ### Key network architectures

# #### Densely connected networks

# In[ ]:

## invisible, just for making everything runnable
num_classes <- 10
num_values <- 4
num_inputs_features <- 20
num_timesteps <- 20
height <- width <- 200
channels <- 3
num_features <- 20

# from tensorflow import keras
# from tensorflow.keras import layers
library(keras)

# inputs = keras.Input(shape=(num_input_features,))
# x = layers.Dense(32, activation="relu")(inputs)
# x = layers.Dense(32, activation="relu")(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop", loss="binary_crossentropy")

inputs <- layer_input(shape = c(num_inputs_features))
outputs <- inputs %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer="rmsprop", loss="binary_crossentropy")



# In[ ]:


# inputs = keras.Input(shape=(num_input_features,))
# x = layers.Dense(32, activation="relu")(inputs)
# x = layers.Dense(32, activation="relu")(x)
# outputs = layers.Dense(num_classes, activation="softmax")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

inputs <- layer_input(shape = c(num_inputs_features))
outputs <- inputs %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(num_classes, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer="rmsprop", loss="categorical_crossentropy")


# In[ ]:


# inputs = keras.Input(shape=(num_input_features,))
# x = layers.Dense(32, activation="relu")(inputs)
# x = layers.Dense(32, activation="relu")(x)
# outputs = layers.Dense(num_classes, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop", loss="binary_crossentropy")

inputs <- layer_input(shape = c(num_inputs_features))
outputs <- inputs %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(num_classes, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer="rmsprop", loss="binary_crossentropy")

# In[ ]:


# inputs = keras.Input(shape=(num_input_features,))
# x = layers.Dense(32, activation="relu")(inputs)
# x = layers.Dense(32, activation="relu")(x)
# outputs layers.Dense(num_values)(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop", loss="mse")

inputs <- layer_input(shape = c(num_inputs_features))
outputs <- inputs %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(num_values)
model <- keras_model(inputs, outputs)
model %>% compile(optimizer="rmsprop", loss="mse")


# #### Convnets

# In[ ]:


# inputs = keras.Input(shape=(height, width, channels))
# x = layers.SeparableConv2D(32, 3, activation="relu")(inputs)
# x = layers.SeparableConv2D(64, 3, activation="relu")(x)
# x = layers.MaxPooling2D(2)(x)
# x = layers.SeparableConv2D(64, 3, activation="relu")(x)
# x = layers.SeparableConv2D(128, 3, activation="relu")(x)
# x = layers.MaxPooling2D(2)(x)
# x = layers.SeparableConv2D(64, 3, activation="relu")(x)
# x = layers.SeparableConv2D(128, 3, activation="relu")(x)
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(32, activation="relu")(x)
# outputs = layers.Dense(num_classes, activation="softmax")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

inputs <- layer_input(shape = c(height, width, channels))
outputs <- inputs %>%
  layer_separable_conv_2d(32, 3, activation = "relu") %>%
  layer_separable_conv_2d(64, 3, activation = "relu") %>%
  layer_max_pooling_2d(2) %>%
  layer_separable_conv_2d(64, 3, activation = "relu") %>%
  layer_separable_conv_2d(128, 3, activation = "relu") %>%
  layer_max_pooling_2d(2) %>%
  layer_separable_conv_2d(64, 3, activation = "relu") %>%
  layer_separable_conv_2d(128, 3, activation = "relu") %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(32, activation = "relu") %>%
  layer_dense(num_classes, activation = "softmax")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy")



# #### RNNs

# In[ ]:


# inputs = keras.Input(shape=(num_timesteps, num_features))
# x = layers.LSTM(32)(inputs)
# outputs = layers.Dense(num_classes, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop", loss="binary_crossentropy")

inputs <- layer_input(shape = c(num_timesteps, num_features))
outputs <- inputs %>%
  layer_lstm(32) %>%
  layer_dense(num_classes, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop", loss = "binary_crossentropy")


# In[ ]:


# inputs = keras.Input(shape=(num_timesteps, num_features))
# x = layers.LSTM(32, return_sequences=True)(inputs)
# x = layers.LSTM(32, return_sequences=True)(x)
# x = layers.LSTM(32)(x)
# outputs = layers.Dense(num_classes, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop", loss="binary_crossentropy")

inputs <- layer_input(shape = c(num_timesteps, num_features))
outputs <- inputs %>%
  layer_lstm(32, return_sequences = TRUE) %>%
  layer_lstm(32, return_sequences = TRUE) %>%
  layer_lstm(32) %>%
  layer_dense(num_classes, activation = "sigmoid")
model <- keras_model(inputs, outputs)
model %>% compile(optimizer = "rmsprop", loss = "binary_crossentropy")


# #### Transformers

# In[ ]:


# encoder_inputs = keras.Input(shape=(sequence_length,), dtype="int64")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
# encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
# decoder_inputs = keras.Input(shape=(None,), dtype="int64")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
# x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
# decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
# transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
# transformer.compile(optimizer="rmsprop", loss="categorical_crossentropy")


encoder_inputs = layer_input(shape = c(sequence_length), dtype = "int64")
encoder_outputs <- encoder_inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim)
transformer_encoder(embed_dim, dense_dim, num_heads)

decoder_inputs = layer_input(shape = c(NA), dtype = "int64")
decoder_outputs <- decoder_inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  # TODO: think about how to pipe into __call__ methods that take multiple args
  {transformer_decoder(embed_dim, dense_dim, num_heads)(., encoder_outputs)} %>%
  layer_dense(vocab_size, activation = "softmax")
transformer = keras_model(list(encoder_inputs, decoder_inputs), decoder_outputs)
transformer %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy")

# In[ ]:


# inputs = keras.Input(shape=(sequence_length,), dtype="int64")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
# x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
# x = layers.GlobalMaxPooling1D()(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)
# model.compile(optimizer="rmsprop", loss="binary_crossentropy")

inputs = layer_input(shape=c(sequence_length), dtype="int64")
outputs <- inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  transformer_encoder(embed_dim, dense_dim, num_heads) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(1, activation = "sigmoid")
model = keras_model(inputs, outputs)
model %>% compile(optimizer="rmsprop", loss="binary_crossentropy")


# ### The space of possibilities

# ## The limitations of deep learning

# ### The risk of anthropomorphizing machine-learning models

# ### Automatons vs. intelligent agents

# ### Local generalization vs. extreme generalization

# ### The purpose of intelligence

# ### Climbing the spectrum of generalization

# ## Setting the course toward greater generality in AI

# ### On the importance of setting the right objective: The shortcut rule

# ### A new target

# ## Implementing intelligence: The missing ingredients

# ### Intelligence as sensitivity to abstract analogies

# ### The two poles of abstraction

# #### Value-centric analogy

# #### Program-centric analogy

# #### Cognition as a combination of both kinds of abstraction

# ### The missing half of the picture

# ## The future of deep learning

# ### Models as programs

# ### Blending together deep learning and program synthesis

# #### Integrating deep-learning modules and algorithmic modules into hybrid systems

# #### Using deep learning to guide program search

# ### Lifelong learning and modular subroutine reuse

# ### The long-term vision

# ## Staying up to date in a fast-moving field

# ### Practice on real-world problems using Kaggle

# ### Read about the latest developments on arXiv

# ### Explore the Keras ecosystem

# ## Final words
