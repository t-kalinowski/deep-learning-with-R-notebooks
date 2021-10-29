#!/usr/bin/env R
# coding: utf-8

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.6.

# # Deep learning for text

# ## Natural-language processing: The bird's eye view

# ## Preparing text data

# ### Text standardization

# ### Text splitting (tokenization)

# ### Vocabulary indexing

# ### Using the TextVectorization layer

# In[ ]:


# import string
#
# class Vectorizer:
#     def standardize(self, text):
#         text = text.lower()
#         return "".join(char for char in text if char not in string.punctuation)
#
#     def tokenize(self, text):
#         text = self.standardize(text)
#         return text.split()
#
#     def make_vocabulary(self, dataset):
#         self.vocabulary = {"": 0, "[UNK]": 1}
#         for text in dataset:
#             text = self.standardize(text)
#             tokens = self.tokenize(text)
#             for token in tokens:
#                 if token not in self.vocabulary:
#                     self.vocabulary[token] = len(self.vocabulary)
#         self.inverse_vocabulary = dict(
#             (v, k) for k, v in self.vocabulary.items())
#
#     def encode(self, text):
#         text = self.standardize(text)
#         tokens = self.tokenize(text)
#         return [self.vocabulary.get(token, 1) for token in tokens]
#
#     def decode(self, int_sequence):
#         return " ".join(
#             self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence)
#
# vectorizer = Vectorizer()
# dataset = [
#     "I write, erase, rewrite",
#     "Erase again, and then",
#     "A poppy blooms.",
# ]
# vectorizer.make_vocabulary(dataset)
library(purrr)

library(R6)
Vectorizer <- R6Class(
  "Vectorizer",
  public = list(
    vocabulary = NULL,
    standardize = function(text) {
      text <- tolower(text)
      gsub("[[:punct:]]", "", text) # remove punctuation
    },
    tokenize = function(text) {
      text = self$standardize(text)
      strsplit(text, "[[:space:]]")[[1]]
    },
    make_vocabulary = function(dataset) {
      self$vocabulary <- dataset %>%
        map(function(text) {
          text %>%
            self$standardize() %>%
            self$tokenize()
        }) %>%
        unlist() %>%
        c("", "[UNK]", .) %>%
        unique()
    },
    encode = function(text) {
      text %>%
        self$standardize() %>%
        self$tokenize() %>%
        match(table = self$vocabulary, nomatch = 1L)
    },
    decode = function(int_sequence) {
      self$vocabulary[int_sequence]
    }
  )
)



vectorizer = Vectorizer$new()
dataset = c(
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms."
)
vectorizer$make_vocabulary(dataset)


# In[ ]:


# test_sentence = "I write, rewrite, and still rewrite again"
# encoded_sentence = vectorizer.encode(test_sentence)
# print(encoded_sentence)
test_sentence <- "I write, rewrite, and still rewrite again"
encoded_sentence <- vectorizer$encode(test_sentence)
print(encoded_sentence)

# In[ ]:


# decoded_sentence = vectorizer.decode(encoded_sentence)
# print(decoded_sentence)
decoded_sentence <- vectorizer$decode(encoded_sentence)
print(decoded_sentence)


# In[ ]:


# from tensorflow.keras.layers import TextVectorization
# text_vectorization = TextVectorization(
#     output_mode="int",
# )
library(keras)
text_vectorization <- layer_text_vectorization(output_mode = "int")


# In[ ]:


# import re
# import string
# import tensorflow as tf
#
# def custom_standardization_fn(string_tensor):
#     lowercase_string = tf.strings.lower(string_tensor)
#     return tf.strings.regex_replace(
#         lowercase_string, f"[{re.escape(string.punctuation)}]", "")
#
# def custom_split_fn(string_tensor):
#     return tf.strings.split(string_tensor)
#
# text_vectorization = TextVectorization(
#     output_mode="int",
#     standardize=custom_standardization_fn,
#     split=custom_split_fn,
# )
library(tensorflow)
library(stringr)

custom_standardization_fn <- function(string_tensor) {
  lowercase_string <- tolower(string_tensor)
  gsub("[[:punct:]]", "",  lowercase_string)
}

custom_split_fn <- function(string_tensor) {
  tf$strings$split(string_tensor)
}

text_vectorization <- layer_text_vectorization(
  output_mode = "int",
  standardize = custom_standardization_fn,
  split = custom_split_fn
)


# In[ ]:


# dataset = [
#     "I write, erase, rewrite",
#     "Erase again, and then",
#     "A poppy blooms.",
# ]
# text_vectorization.adapt(dataset)
dataset <- c("I write, erase, rewrite",
             "Erase again, and then",
             "A poppy blooms.")
adapt(text_vectorization, dataset)


# **Displaying the vocabulary**

# In[ ]:


# text_vectorization.get_vocabulary()
get_vocabulary(text_vectorization)


# In[ ]:


# vocabulary = text_vectorization.get_vocabulary()
# test_sentence = "I write, rewrite, and still rewrite again"
# encoded_sentence = text_vectorization(test_sentence)
# print(encoded_sentence)
vocabulary <- text_vectorization %>% get_vocabulary()
test_sentence <- "I write, rewrite, and still rewrite again"
encoded_sentence <- text_vectorization(test_sentence)
print(encoded_sentence)


# In[ ]:


# inverse_vocab = dict(enumerate(vocabulary))
# decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
# print(decoded_sentence)
decoded_sentence = paste(vocabulary[as.integer(encoded_sentence) + 1],
                         collapse = " ")
print(decoded_sentence)


# ## Two approaches for representing groups of words: Sets and sequences

# ### Preparing the IMDB movie reviews data

# In[ ]:

if(FALSE) {


# get_ipython().system('curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
# get_ipython().system('tar -xf aclImdb_v1.tar.gz')
system('curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
system('tar -xf aclImdb_v1.tar.gz')


# In[ ]:


# get_ipython().system('rm -r aclImdb/train/unsup')
system('rm -r aclImdb/train/unsup')


# In[ ]:


# get_ipython().system('cat aclImdb/train/pos/4077_10.txt')
system('cat aclImdb/train/pos/4077_10.txt')

}

# In[ ]:


# import os, pathlib, shutil, random
#
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

library(fs)
base_dir <- path("aclImdb")
val_dir <- base_dir / "val"
train_dir <- base_dir / "train"
# shuffle <- function(x) x[sample.int(length(x))]
shuffle <- function(x) sample(x, length(x))
for (category in c("neg", "pos")) {
  dir_create(val_dir / category)
  file_names <- dir_ls(train_dir / category) %>%
    basename() %>%
    shuffle()
  num_val_samples <- round(0.2 * length(file_names))
  val_file_names <- tail(file_names, num_val_samples)
  file_move(train_dir / category / val_file_names,
            val_dir / category / val_file_names)
}


# In[ ]:


# from tensorflow import keras
# batch_size = 32
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
library(keras)
library(tfdatasets)
# batch_size <- 32 # default is already 32

train_ds <- text_dataset_from_directory("aclImdb/train")
val_ds <- text_dataset_from_directory("aclImdb/val")
test_ds <- text_dataset_from_directory("aclImdb/test")


# **Displaying the shapes and dtypes of the first batch**

# In[ ]:


# for inputs, targets in train_ds:
#     print("inputs.shape:", inputs.shape)
#     print("inputs.dtype:", inputs.dtype)
#     print("targets.shape:", targets.shape)
#     print("targets.dtype:", targets.dtype)
#     print("inputs[0]:", inputs[0])
#     print("targets[0]:", targets[0])
#     break

c(inputs, targets) %<-% iter_next(as_iterator(train_ds))
inputs$shape
inputs$dtype
targets$shape
targets$dtype
inputs[1]
targets[1]


# ### Processing words as a set: The bag-of-words approach

# #### Single words (unigrams) with binary encoding

# **Preprocessing our datasets with a `TextVectorization` layer**

# In[ ]:

#
# text_vectorization = TextVectorization(
#     max_tokens=20000,
#     output_mode="multi_hot",
# )
# text_only_train_ds = train_ds.map(lambda x, y: x)
# text_vectorization.adapt(text_only_train_ds)
#
# binary_1gram_train_ds = train_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)
# binary_1gram_val_ds = val_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)
# binary_1gram_test_ds = test_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)

text_vectorization <- layer_text_vectorization(max_tokens = 20000,
                                               output_mode = "multi_hot")

text_only_train_ds <- train_ds %>%
  dataset_map(function(x, y) x)
adapt(text_vectorization, text_only_train_ds)

binary_1gram_train_ds <- train_ds %>%
  dataset_map(~list(text_vectorization(.x), .y), num_parallel_calls = 4)
binary_1gram_val_ds <- val_ds %>%
  dataset_map(~list(text_vectorization(.x), .y), num_parallel_calls = 4)
binary_1gram_test_ds <- test_ds %>%
  dataset_map(~list(text_vectorization(.x), .y), num_parallel_calls = 4)


# **Inspecting the output of our binary unigram dataset**

# In[ ]:


# for inputs, targets in binary_1gram_train_ds:
#     print("inputs.shape:", inputs.shape)
#     print("inputs.dtype:", inputs.dtype)
#     print("targets.shape:", targets.shape)
#     print("targets.dtype:", targets.dtype)
#     print("inputs[0]:", inputs[0])
#     print("targets[0]:", targets[0])
#     break

c(inputs, targets) %<-% iter_next(as_iterator(binary_1gram_train_ds))
inputs$shape
inputs$dtype
targets$shape
targets$dtype
inputs[1,]
targets[1]


# **Our model-building utility**

# In[ ]:


# from tensorflow import keras
# from tensorflow.keras import layers
#
# def get_model(max_tokens=20000, hidden_dim=16):
#     inputs = keras.Input(shape=(max_tokens,))
#     x = layers.Dense(hidden_dim, activation="relu")(inputs)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(1, activation="sigmoid")(x)
#     model = keras.Model(inputs, outputs)
#     model.compile(optimizer="rmsprop",
#                   loss="binary_crossentropy",
#                   metrics=["accuracy"])
#     return model

library(keras)
get_model <- function(max_tokens = 20000, hidden_dim = 16) {
  inputs <- layer_input(shape = c(max_tokens))
  outputs <- inputs %>%
    layer_dense(hidden_dim, activation = "relu") %>%
    layer_dropout(0.5) %>%
    layer_dense(1, activation = "sigmoid")
  model <- keras_model(inputs, outputs)
  model %>% compile(optimizer = "rmsprop",
                    loss = "binary_crossentropy",
                    metrics = "accuracy")
  model
}


# **Training and testing the binary unigram model**

# In[ ]:


# model = get_model()
# model.summary()
# callbacks = [
#     keras.callbacks.ModelCheckpoint("binary_1gram.keras",
#                                     save_best_only=True)
# ]
# model.fit(binary_1gram_train_ds.cache(),
#           validation_data=binary_1gram_val_ds.cache(),
#           epochs=10,
#           callbacks=callbacks)
# model = keras.models.load_model("binary_1gram.keras")
# print(f"Test acc: {model.evaluate(binary_1gram_test_ds)[1]:.3f}")
model <- get_model()
model
callbacks = list(
  callback_model_checkpoint("binary_1gram.keras", save_best_only = TRUE)
)

model %>% fit(
  dataset_cache(binary_1gram_train_ds),
  validation_data = dataset_cache(binary_1gram_val_ds),
  epochs = 10,
  callbacks = callbacks
)
model <- load_model_tf("binary_1gram.keras")
sprintf("Test acc: %.3f", evaluate(model, binary_1gram_test_ds)["accuracy"])


# #### Bigrams with binary encoding

# **Configuring the `TextVectorization` layer to return bigrams**

# In[ ]:


# text_vectorization = TextVectorization(
#     ngrams=2,
#     max_tokens=20000,
#     output_mode="multi_hot",
# )

text_vectorization <- layer_text_vectorization(
  ngrams = 2,
  max_tokens = 20000,
  output_mode = "multi_hot")


# **Training and testing the binary bigram model**

# In[ ]:


# text_vectorization.adapt(text_only_train_ds)
# binary_2gram_train_ds = train_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)
# binary_2gram_val_ds = val_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)
# binary_2gram_test_ds = test_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)
#
# model = get_model()
# model.summary()
# callbacks = [
#     keras.callbacks.ModelCheckpoint("binary_2gram.keras",
#                                     save_best_only=True)
# ]
# model.fit(binary_2gram_train_ds.cache(),
#           validation_data=binary_2gram_val_ds.cache(),
#           epochs=10,
#           callbacks=callbacks)
# model = keras.models.load_model("binary_2gram.keras")
# print(f"Test acc: {model.evaluate(binary_2gram_test_ds)[1]:.3f}")

adapt(text_vectorization, text_only_train_ds)
binary_2gram_train_ds <- train_ds %>%
  dataset_map(~ list(text_vectorization(.x), .y), num_parallel_calls = 4)
binary_2gram_val_ds = val_ds %>%
  dataset_map(~ list(text_vectorization(.x), .y), num_parallel_calls = 4)
binary_2gram_test_ds = test_ds %>%
  dataset_map(~ list(text_vectorization(.x), .y), num_parallel_calls = 4)

model <- get_model()
model
callbacks = list(
    callback_model_checkpoint("binary_2gram.keras",
                                    save_best_only=TRUE)
)
model %>% fit(
  dataset_cache(binary_2gram_train_ds),
  validation_data = dataset_cache(binary_2gram_val_ds),
  epochs = 10,
  callbacks = callbacks
)
model <- load_model_tf("binary_2gram.keras")
sprintf("Test acc: %.3f", evaluate(model, binary_2gram_test_ds)["accuracy"])


# #### Bigrams with TF-IDF encoding

# **Configuring the `TextVectorization` layer to return token counts**

# In[ ]:


# text_vectorization = TextVectorization(
#     ngrams=2,
#     max_tokens=20000,
#     output_mode="count"
# )
text_vectorization <- layer_text_vectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="count"
)


# **Configuring `TextVectorization` to return TF-IDF-weighted outputs**

# In[ ]:


# text_vectorization = TextVectorization(
#     ngrams=2,
#     max_tokens=20000,
#     output_mode="tf_idf",
# )

text_vectorization <- layer_text_vectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="tf_idf"
)

# **Training and testing the TF-IDF bigram model**

# In[ ]:


# text_vectorization.adapt(text_only_train_ds)
#
# tfidf_2gram_train_ds = train_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)
# tfidf_2gram_val_ds = val_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)
# tfidf_2gram_test_ds = test_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4)
#
# model = get_model()
# model.summary()
# callbacks = [
#     keras.callbacks.ModelCheckpoint("tfidf_2gram.keras",
#                                     save_best_only=True)
# ]
# model.fit(tfidf_2gram_train_ds.cache(),
#           validation_data=tfidf_2gram_val_ds.cache(),
#           epochs=10,
#           callbacks=callbacks)
# model = keras.models.load_model("tfidf_2gram.keras")
# print(f"Test acc: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}")

adapt(text_vectorization, text_only_train_ds)

tfidf_2gram_train_ds <- train_ds %>%
  dataset_map( ~ list(text_vectorization(.x), .y), num_parallel_calls = 4)
tfidf_2gram_val_ds = val_ds %>%
  dataset_map( ~ list(text_vectorization(.x), .y), num_parallel_calls = 4)
tfidf_2gram_test_ds = test_ds %>%
  dataset_map( ~ list(text_vectorization(.x), .y), num_parallel_calls = 4)

model <- get_model()
model
callbacks = list(
  callback_model_checkpoint("tfidf_2gram.keras", save_best_only = TRUE))

model %>% fit(
  dataset_cache(tfidf_2gram_train_ds),
  validation_data = dataset_cache(tfidf_2gram_val_ds),
  epochs = 10,
  callbacks = callbacks
)
model <- load_model_tf("tfidf_2gram.keras")
sprintf("Test acc: %.3f", evaluate(model, tfidf_2gram_test_ds)["accuracy"])


# In[ ]:


# inputs = keras.Input(shape=(1,), dtype="string")
# processed_inputs = text_vectorization(inputs)
# outputs = model(processed_inputs)
# inference_model = keras.Model(inputs, outputs)

inputs <- layer_input(shape=c(1), dtype="string")
processed_inputs <- text_vectorization(inputs)
outputs <- model(processed_inputs)
inference_model <- keras_model(inputs, outputs)


# In[ ]:


# import tensorflow as tf
# raw_text_data = tf.convert_to_tensor([
#     ["That was an excellent movie, I loved it."],
# ])
# predictions = inference_model(raw_text_data)
# print(f"{float(predictions[0] * 100):.2f} percent positive")

raw_text_data <- as_tensor(
  array(c("That was an excellent movie, I loved it."),
        dim = c(1, 1)))
predictions <- inference_model(raw_text_data)
sprintf("%.2f percent positive", as.numeric(predictions) * 100)
