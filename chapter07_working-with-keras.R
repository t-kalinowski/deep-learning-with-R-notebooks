#!/usr/bin/env Rscript

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.7.

# # Working with Keras: A deep dive

# ## A spectrum of workflows

# ## Different ways to build Keras models

# ### The Sequential model

# **The `Sequential` class**

# In[ ]:

#
# from tensorflow import keras
# from tensorflow.keras import layers
#
# model = keras.Sequential([
#     layers.Dense(64, activation="relu"),
#     layers.Dense(10, activation="softmax")
# ])


library(keras)
# library(tensorflow)

#Q: introduce alias keras_sequential() for keras_model_sequential()?
model <- keras_model_sequential() %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(10, activation = "softmax")




# **Incrementally building a Sequential model**

# In[ ]:


# model = keras.Sequential()
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dense(10, activation="softmax"))

model <- keras_model_sequential()
model %>% layer_dense(64, activation="relu")
model %>% layer_dense(10, activation="softmax")



# **Calling a model for the first time to build it**

# In[ ]:

#
# model.build(input_shape=(None, 3))
# model.weights

model$build(input_shape = shape(NULL, 3))
model$weights


# **The summary method**

# In[ ]:

#
# model.summary()

model

# **Naming models and layers with the `name` argument**

# In[ ]:

#
# model = keras.Sequential(name="my_example_model")
# model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
# model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
# model.build((None, 3))
# model.summary()

model = keras_model_sequential(name="my_example_model")
model %>% layer_dense(64, activation="relu", name="my_first_layer")
model %>% layer_dense(10, activation="softmax", name="my_last_layer")
#Q: reexport shape() from keras?
model$build(tensorflow::shape(NULL, 3))
model



# **Specifying the input shape of your model in advance**

# In[ ]:

#
# model = keras.Sequential()
# model.add(keras.Input(shape=(3,)))
# model.add(layers.Dense(64, activation="relu"))

model <-
  keras_model_sequential(input_shape = c(3)) %>%
  layer_dense(64, activation="relu")


# In[ ]:

#
# model.summary()

model


# In[ ]:

#
# model.add(layers.Dense(10, activation="softmax"))
# model.summary()

model %>% layer_dense(10, activation="softmax")
model


# ### The Functional API

# #### A simple example

# **A simple Functional model with two `Dense` layers**

# In[ ]:

#
# inputs = keras.Input(shape=(3,), name="my_input")
# features = layers.Dense(64, activation="relu")(inputs)
# outputs = layers.Dense(10, activation="softmax")(features)
# model = keras.Model(inputs=inputs, outputs=outputs)

inputs <- layer_input(shape=c(3), name="my_input")
features <- layer_dense(inputs, 64, activation="relu")
outputs <- layer_dense(features, 10, activation="softmax")
model <- keras_model(inputs=inputs, outputs=outputs)


# In[ ]:

#
# inputs = keras.Input(shape=(3,), name="my_input")

inputs <- layer_input(shape=c(3), name="my_input")


# In[ ]:


# inputs.shape
inputs$shape


# In[ ]:


# inputs.dtype
inputs$dtype


# In[ ]:


# features = layers.Dense(64, activation="relu")(inputs)
features <- layer_dense(inputs, 64, activation="relu")


# In[ ]:


# features.shape
features$shape


# In[ ]:

#
# outputs = layers.Dense(10, activation="softmax")(features)
# model = keras.Model(inputs=inputs, outputs=outputs)

outputs <- layer_dense(features, 10, activation="softmax")
model <- keras_model(inputs=inputs, outputs=outputs)


# In[ ]:


# model.summary()
model


# #### Multi-input, multi-output models

# **A multi-input, multi-output Functional model**

# In[ ]:


# vocabulary_size = 10000
# num_tags = 100
# num_departments = 4
#
# title = keras.Input(shape=(vocabulary_size,), name="title")
# text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
# tags = keras.Input(shape=(num_tags,), name="tags")
#
# features = layers.Concatenate()([title, text_body, tags])
# features = layers.Dense(64, activation="relu")(features)
#
# priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
# department = layers.Dense(
#     num_departments, activation="softmax", name="department")(features)
#
# model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])

vocabulary_size <- 10000
num_tags <- 100
num_departments <- 4

title     <- layer_input(shape = c(vocabulary_size), name = "title")
text_body <- layer_input(shape = c(vocabulary_size), name = "text_body")
tags      <- layer_input(shape = c(num_tags), name = "tags")

features <- layer_concatenate(list(title, text_body, tags)) %>%
  layer_dense(64, activation="relu")

priority <- features %>% layer_dense(1, activation="sigmoid", name="priority")
department <- features %>% layer_dense(num_departments, activation="softmax", name="department")

model <- keras_model(inputs=list(title, text_body, tags),
                     outputs=list(priority, department))


# #### Training a multi-input, multi-output model

# **Training a model by providing lists of input & target arrays**

# In[ ]:
#
#
# import numpy as np
#
# num_samples = 1280
#
# title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
# text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
# tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))
#
# priority_data = np.random.random(size=(num_samples, 1))
# department_data = np.random.randint(0, 2, size=(num_samples, num_departments))
#
# model.compile(optimizer="rmsprop",
#               loss=["mean_squared_error", "categorical_crossentropy"],
#               metrics=[["mean_absolute_error"], ["accuracy"]])
# model.fit([title_data, text_body_data, tags_data],
#           [priority_data, department_data],
#           epochs=1)
# model.evaluate([title_data, text_body_data, tags_data],
#                [priority_data, department_data])
# priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])
#



num_samples <- 1280

#Q: use positional matching to save space?
# k_random_uniform(c(num_samples, vocabulary_size), 0, 2, "int64")
title_data <- k_random_uniform(shape = c(num_samples, vocabulary_size),
                               minval = 0, maxval = 2, dtype = "int64")
text_body_data <- k_random_uniform(shape = c(num_samples, vocabulary_size),
                                   minval = 0, maxval = 2, dtype = "int64")
tags_data <- k_random_uniform(shape = c(num_samples, num_tags),
                              minval = 0, maxval = 2, dtype = "int64")

priority_data <- k_random_uniform(shape=c(num_samples, 1))
department_data  <- k_random_uniform(shape = c(num_samples, num_departments),
                                     minval = 0, maxval = 2, dtype = "int64")


model %>% compile(
  optimizer = "rmsprop",
  loss = c("mean_squared_error", "categorical_crossentropy"),
  metrics = list(c("mean_absolute_error"), c("accuracy"))
)

model %>% fit(
  list(title_data, text_body_data, tags_data),
  list(priority_data, department_data),
  epochs = 1
)
model %>% evaluate(list(title_data, text_body_data, tags_data),
                   list(priority_data, department_data))
result <- model %>% predict(list(title_data, text_body_data, tags_data))
c(priority_preds, department_preds) %<-% result


# **Training a model by providing dicts of input & target arrays**

# In[ ]:
#
# model.compile(optimizer="rmsprop",
#               loss={"priority": "mean_squared_error", "department": "categorical_crossentropy"},
#               metrics={"priority": ["mean_absolute_error"], "department": ["accuracy"]})
# model.fit({"title": title_data, "text_body": text_body_data, "tags": tags_data},
#           {"priority": priority_data, "department": department_data},
#           epochs=1)
# model.evaluate({"title": title_data, "text_body": text_body_data, "tags": tags_data},
#                {"priority": priority_data, "department": department_data})
# priority_preds, department_preds = model.predict(
#     {"title": title_data, "text_body": text_body_data, "tags": tags_data})
#


model %>% compile(
  optimizer = "rmsprop",
  loss = list(priority = "mean_squared_error",
              department =  "categorical_crossentropy"),
  metrics = list(priority =  "mean_absolute_error",
                 department = "accuracy")
)
model %>% fit(
  list(title = title_data,
       text_body = text_body_data,
       tags = tags_data),
  list(priority = priority_data, department = department_data),
  epochs = 1
)
model %>% evaluate(list(title = title_data,
                        text_body = text_body_data,
                        tags = tags_data),
                   list(priority = priority_data,
                        department = department_data)
)
c(priority_preds, department_preds) %<-%
  predict(model, list(title = title_data,
                      text_body = text_body_data,
                      tags = tags_data))



# #### The power of the Functional API: Access to layer connectivity

# In[ ]:

#
# keras.utils.plot_model(model, "ticket_classifier.png")

keras$utils$plot_model(model, "ticket_classifier.png")


# In[ ]:


# keras.utils.plot_model(model, "ticket_classifier_with_shape_info.png", show_shapes=True)
keras$utils$plot_model(model, "ticket_classifier_with_shape_info.png", show_shapes=TRUE)
#TODO: revisit wrapping plot_model(). At last visit, decided against wrapping
# because of onerous graphviz installation requirements.


# **Retrieving the inputs or outputs of a layer in a Functional model**

# In[ ]:


# model.layers
model$layers


# In[ ]:


# model.layers[3].input
model$layers[[4]]$input


# In[ ]:


# model.layers[3].output
model$layers[[4]]$output


# **Creating a new model by reusing intermediate layer outputs**

# In[ ]:

#
# features = model.layers[4].output
# difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)
#
# new_model = keras.Model(
#     inputs=[title, text_body, tags],
#     outputs=[priority, department, difficulty])

features <- model$layers[[5]]$output
difficulty <- features %>% layer_dense(3, activation="softmax", name="difficulty")

new_model <- keras_model(
  inputs = list(title, text_body, tags),
  outputs = list(priority, department, difficulty)
)


# In[ ]:


# keras.utils.plot_model(new_model, "updated_ticket_classifier.png", show_shapes=True)
keras$utils$plot_model(new_model, "updated_ticket_classifier.png", show_shapes=TRUE)


# ### Subclassing the Model class

# #### Rewriting our previous example as a subclassed model

# **A simple subclassed model**

# In[ ]:

#
# class CustomerTicketModel(keras.Model):
#
#     def __init__(self, num_departments):
#         super().__init__()
#         self.concat_layer = layers.Concatenate()
#         self.mixing_layer = layers.Dense(64, activation="relu")
#         self.priority_scorer = layers.Dense(1, activation="sigmoid")
#         self.department_classifier = layers.Dense(
#             num_departments, activation="softmax")
#
#     def call(self, inputs):
#         title = inputs["title"]
#         text_body = inputs["text_body"]
#         tags = inputs["tags"]
#
#         features = self.concat_layer([title, text_body, tags])
#         features = self.mixing_layer(features)
#         priority = self.priority_scorer(features)
#         department = self.department_classifier(features)
#         return priority, department
library(R6)
CustomerTicketModel <- create_layer_wrapper(R6Class(
  classname = "CustomerTicketModel",
  inherit = keras$Model,
  public = list(
    initialize = function(num_departments) {
      super$initialize()
      self$concat_layer <- layer_concatenate()
      self$mixing_layer <-
        layer_dense(units = 64, activation = "relu")
      self$priority_scorer <-
        layer_dense(units = 1, activation = "sigmoid")
      self$department_classifier <-
        layer_dense(units = num_departments,  activation = "softmax")
    },

    call = function(inputs) {
      title <- inputs$title
      text_body <- inputs$text_body
      tags <- inputs$tags

      features <- list(title, text_body, tags) %>%
        self$concat_layer() %>%
        self$mixing_layer()
      priority <- self$priority_scorer(features)
      department <- self$department_classifier(features)
      list(priority, department)
    }
  )
))





# In[ ]:

#
# model = CustomerTicketModel(num_departments=4)
#
# priority, department = model(
#     {"title": title_data, "text_body": text_body_data, "tags": tags_data})

model <- CustomerTicketModel(num_departments=4)

c(priority, department) %<-% model(list(title = title_data,
                                        text_body = text_body_data,
                                        tags = tags_data))


# In[ ]:

#
# model.compile(optimizer="rmsprop",
#               loss=["mean_squared_error", "categorical_crossentropy"],
#               metrics=[["mean_absolute_error"], ["accuracy"]])
# model.fit({"title": title_data,
#            "text_body": text_body_data,
#            "tags": tags_data},
#           [priority_data, department_data],
#           epochs=1)
# model.evaluate({"title": title_data,
#                 "text_body": text_body_data,
#                 "tags": tags_data},
#                [priority_data, department_data])
# priority_preds, department_preds = model.predict({"title": title_data,
#                                                   "text_body": text_body_data,
#                                                   "tags": tags_data})

model %>% compile(optimizer="rmsprop",
              loss=c("mean_squared_error", "categorical_crossentropy"),
              metrics=c("mean_absolute_error", "accuracy"))
model %>% fit(list(title = title_data,
                   text_body = text_body_data,
                   tags = tags_data),
              list(priority_data, department_data),
              epochs = 1)
model %>% evaluate(list(title = title_data,
                        text_body = text_body_data,
                        tags = tags_data),
                   list(priority_data, department_data))
c(priority_preds, department_preds) %<-% (
  model %>% predict(list(title = title_data,
                         text_body = text_body_data,
                         tags = tags_data))
)



# #### Beware: What subclassed models don't support

# ### Mixing and matching different components

# **Creating a Functional model that includes a subclassed model**

# In[ ]:

#
# class Classifier(keras.Model):
#
#     def __init__(self, num_classes=2):
#         super().__init__()
#         if num_classes == 2:
#             num_units = 1
#             activation = "sigmoid"
#         else:
#             num_units = num_classes
#             activation = "softmax"
#         self.dense = layers.Dense(num_units, activation=activation)
#
#     def call(self, inputs):
#         return self.dense(inputs)
#
# inputs = keras.Input(shape=(3,))
# features = layers.Dense(64, activation="relu")(inputs)
# outputs = Classifier(num_classes=10)(features)
# model = keras.Model(inputs=inputs, outputs=outputs)

Classifier <- create_layer_wrapper(R6Class(
  classname = "Classifier",
  inherit = keras$Model,
  public = list(
    initialize = function(num_classes = 2) {
      super$initialize()
      if (num_classes == 2) {
        num_units <- 1
        activation <- "sigmoid"
      } else {
        num_units <- num_classes
        activation <- "softmax"
      }
      self$dense = layer_dense(units = num_units, activation = activation)
    },

    call = function(inputs) {
      self$dense(inputs)
    }
    )
))

inputs  <- layer_input(shape= c(3))
features <- inputs %>% layer_dense(64, activation="relu")
outputs <- features %>% Classifier(num_classes=10)
model <- keras_model(inputs=inputs, outputs=outputs)


# **Creating a subclassed model that includes a Functional model**

# In[ ]:

#
# inputs = keras.Input(shape=(64,))
# outputs = layers.Dense(1, activation="sigmoid")(inputs)
# binary_classifier = keras.Model(inputs=inputs, outputs=outputs)
#
# class MyModel(keras.Model):
#
#     def __init__(self, num_classes=2):
#         super().__init__()
#         self.dense = layers.Dense(64, activation="relu")
#         self.classifier = binary_classifier
#
#     def call(self, inputs):
#         features = self.dense(inputs)
#         return self.classifier(features)
#
# model = MyModel()

inputs <- layer_input(shape=c(64))
outputs <- inputs %>% layer_dense(1, activation="sigmoid")
binary_classifier <- keras_model(inputs=inputs, outputs=outputs)

MyModel <- create_layer_wrapper(R6Class(
  classname = "MyModel",
  inherit = keras$Model,
  public = list(
    initialize = function(num_classes=2) {

        super$initialize()
        self$dense = layer_dense(units = 64, activation="relu")
        self$classifier = binary_classifier
    },

    call = function(inputs) {
      inputs %>%
        self$dense() %>%
        self$classifier()
    }
)))

model <- MyModel()


# ### Remember: Use the right tool for the job

# ## Using built-in training and evaluation loops

# **The standard workflow: `compile()`, `fit()`, `evaluate()`, `predict()`**

# In[ ]:

#
# from tensorflow.keras.datasets import mnist
#
# def get_mnist_model():
#     inputs = keras.Input(shape=(28 * 28,))
#     features = layers.Dense(512, activation="relu")(inputs)
#     features = layers.Dropout(0.5)(features)
#     outputs = layers.Dense(10, activation="softmax")(features)
#     model = keras.Model(inputs, outputs)
#     return model
#
# (images, labels), (test_images, test_labels) = mnist.load_data()
# images = images.reshape((60000, 28 * 28)).astype("float32") / 255
# test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
# train_images, val_images = images[10000:], images[:10000]
# train_labels, val_labels = labels[10000:], labels[:10000]
#
# model = get_mnist_model()
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_images, train_labels,
#           epochs=3,
#           validation_data=(val_images, val_labels))
# test_metrics = model.evaluate(test_images, test_labels)
# predictions = model.predict(test_images)


get_mnist_model <- function() {

    inputs <- layer_input(shape=c(28 * 28))
    features <- inputs %>%
      layer_dense(512, activation="relu") %>%
      layer_dropout(0.5)
    outputs <- features %>%
      layer_dense(10, activation="softmax")

    model <- keras_model(inputs, outputs)
    model
}

mnist <- dataset_mnist()
c(images, labels)           %<-% mnist$train
c(test_images, test_labels) %<-% mnist$test

images <- array_reshape(images, c(60000, 28 * 28)) / 255
test_images = array_reshape(test_images, c(10000, 28 * 28)) / 255
c(train_images, val_images) %<-% list(images[-seq(10000),], images[seq(10000),])
c(train_labels, val_labels) %<-% list(labels[-seq(10000)], labels[seq(10000)])

model <- get_mnist_model()
model %>% compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")
model %>% fit(train_images, train_labels,
              epochs=3,
              validation_data=list(val_images, val_labels))
test_metrics <-  model %>% evaluate(test_images, test_labels)
predictions <- model %>% predict(test_images)


# ### Writing your own metrics

# **Implementing a custom metric by subclassing the `Metric` class**

# In[ ]:

#
# import tensorflow as tf
#
# class RootMeanSquaredError(keras.metrics.Metric):
#
#     def __init__(self, name="rmse", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
#         self.total_samples = self.add_weight(
#             name="total_samples", initializer="zeros", dtype="int32")
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
#         mse = tf.reduce_sum(tf.square(y_true - y_pred))
#         self.mse_sum.assign_add(mse)
#         num_samples = tf.shape(y_pred)[0]
#         self.total_samples.assign_add(num_samples)
#
#     def result(self):
#         return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))
#
#     def reset_state(self):
#         self.mse_sum.assign(0.)
#         self.total_samples.assign(0)

library(tensorflow)

RootMeanSquaredError(keras$metrics$Metric) %py_class% {
  initialize <- function(name = "rmse", ...) {
    super$initialize(name = name, ...)
    self$mse_sum <-
      self$add_weight(name = "mse_sum", initializer = "zeros")
    self$total_samples <- self$add_weight(name = "total_samples",
                                          initializer = "zeros",
                                          dtype = "int32")
  }

  update_state <- function(y_true, y_pred, sample_weight = NULL) {
    y_true <- tf$one_hot(y_true, depth = tf$shape(y_pred)[2])
    mse <- tf$reduce_sum(tf$square(y_true - y_pred))
    self$mse_sum$assign_add(mse)
    num_samples <- tf$shape(y_pred)[1]
    self$total_samples$assign_add(num_samples)
  }

  result <- function()
    tf$sqrt(self$mse_sum / tf$cast(self$total_samples, tf$float32))

  reset_state <- function() {
    self$mse_sum$assign(0)
    self$total_samples$assign(0L)
  }
}

# In[ ]:

#
# model = get_mnist_model()
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy", RootMeanSquaredError()])
# model.fit(train_images, train_labels,
#           epochs=3,
#           validation_data=(val_images, val_labels))
# test_metrics = model.evaluate(test_images, test_labels)

model <- get_mnist_model()
model %>% compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=list("accuracy", RootMeanSquaredError()))
model %>% fit(train_images, train_labels,
              epochs=3,
              validation_data = list(val_images, val_labels))
test_metrics <- model %>% evaluate(test_images, test_labels)


# ### Using callbacks

# #### The EarlyStopping and ModelCheckpoint callbacks

# **Using the `callbacks` argument in the `fit()` method**

# In[ ]:

#
# callbacks_list = [
#     keras.callbacks.EarlyStopping(
#         monitor="val_accuracy",
#         patience=2,
#     ),
#     keras.callbacks.ModelCheckpoint(
#         filepath="checkpoint_path.keras",
#         monitor="val_loss",
#         save_best_only=True,
#     )
# ]
# model = get_mnist_model()
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_images, train_labels,
#           epochs=10,
#           callbacks=callbacks_list,
#           validation_data=(val_images, val_labels))
#


callbacks_list <- list(
  callback_early_stopping(monitor = "val_accuracy", patience = 2),
  callback_model_checkpoint(
    filepath = "checkpoint_path.keras",
    monitor = "val_loss",
    save_best_only = TRUE
  )
)
model <- get_mnist_model()
model %>% compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")
model %>% fit(train_images, train_labels,
              epochs=10,
              callbacks=callbacks_list,
              validation_data=list(val_images, val_labels))


# In[ ]:


# model = keras.models.load_model("checkpoint_path.keras")
model <- load_model_tf("checkpoint_path.keras")


# ### Writing your own callbacks

# **Creating a custom callback by subclassing the `Callback` class**

# In[ ]:

#
# from matplotlib import pyplot as plt
#
# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs):
#         self.per_batch_losses = []
#
#     def on_batch_end(self, batch, logs):
#         self.per_batch_losses.append(logs.get("loss"))
#
#     def on_epoch_end(self, epoch, logs):
#         plt.clf()
#         plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses,
#                  label="Training loss for each batch")
#         plt.xlabel(f"Batch (epoch {epoch})")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.savefig(f"plot_at_epoch_{epoch}")
#         self.per_batch_losses = []

`append1<-` <- function (x, value) {
    x[[length(x) + 1L]] <- value
    x
}
LossHistory(keras$callbacks$Callback) %py_class% {

    on_train_begin <- function(logs)
        self$per_batch_losses <- list()

    on_batch_end <- function(batch, logs)
        append1(self$per_batch_losses) <- logs$loss

    on_epoch_end <- function(epoch, logs) {
      #Q: use ggplot here?
        plot(self$per_batch_losses, type = "o",
             main = "Training loss for each batch",
             xlab = glue::glue("Batch (epoch {epoch})"), ylab = "Loss")
        self$per_batch_losses = list()
    }
}


# In[ ]:

#
# model = get_mnist_model()
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_images, train_labels,
#           epochs=10,
#           callbacks=[LossHistory()],
#           validation_data=(val_images, val_labels))

model <- get_mnist_model()
model %>% compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")
model %>% fit(train_images, train_labels,
              epochs=10,
              callbacks=list(LossHistory()),
              validation_data=list(val_images, val_labels))


# ### Monitoring and visualization with TensorBoard

# In[ ]:

#
# model = get_mnist_model()
# model.compile(optimizer="rmsprop",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
#
# tensorboard = keras.callbacks.TensorBoard(
#     log_dir="/full_path_to_your_log_dir",
# )
# model.fit(train_images, train_labels,
#           epochs=10,
#           validation_data=(val_images, val_labels),
#           callbacks=[tensorboard])

model <- get_mnist_model()
model %>% compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")

log_dir <- "/full_path_to_your_log_dir"
log_dir <- "logs"
model %>% fit(train_images, train_labels,
              epochs=10,
              validation_data=list(val_images, val_labels),
              callbacks=callback_tensorboard(log_dir))


# In[ ]:


# get_ipython().run_line_magic('load_ext', 'tensorboard')
# get_ipython().run_line_magic('tensorboard', '--logdir /full_path_to_your_log_dir')
tensorboard("logs")



# ## Writing your own training and evaluation loops

# ### Training versus inference

# ### Low-level usage of metrics

# In[ ]:

#
# metric = keras.metrics.SparseCategoricalAccuracy()
# targets = [0, 1, 2]
# predictions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# metric.update_state(targets, predictions)
# current_result = metric.result()
# print(f"result: {current_result:.2f}")

metric <- metric_sparse_categorical_accuracy()
targets <- c(0, 1, 2)
predictions <- rbind(c(1, 0, 0),
                     c(0, 1, 0),
                     c(0, 0, 1))
metric$update_state(targets, predictions)
current_result <- metric$result() %>% as.array()
sprintf("result: %.2f",  current_result)


# In[ ]:

# values = [0, 1, 2, 3, 4]
# mean_tracker = keras.metrics.Mean()
# for value in values:
#     mean_tracker.update_state(value)
# print(f"Mean of values: {mean_tracker.result():.2f}")

values <- c(0, 1, 2, 3, 4)
mean_tracker <- metric_mean()
for (value in values)
    mean_tracker$update_state(value)
sprintf("Mean of values: %.2f", as.array(mean_tracker$result()))


# ### A complete training and evaluation loop

# **Writing a step-by-step training loop: the training step function**

# In[ ]:

#
# model = get_mnist_model()
#
# loss_fn = keras.losses.SparseCategoricalCrossentropy()
# optimizer = keras.optimizers.RMSprop()
# metrics = [keras.metrics.SparseCategoricalAccuracy()]
# loss_tracking_metric = keras.metrics.Mean()
#
# def train_step(inputs, targets):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs, training=True)
#         loss = loss_fn(targets, predictions)
#     gradients = tape.gradient(loss, model.trainable_weights)
#     optimizer.apply_gradients(zip(gradients, model.trainable_weights))
#
#     logs = {}
#     for metric in metrics:
#         metric.update_state(targets, predictions)
#         logs[metric.name] = metric.result()
#
#     loss_tracking_metric.update_state(loss)
#     logs["loss"] = loss_tracking_metric.result()
#     return logs

model <- get_mnist_model()

loss_fn <- loss_sparse_categorical_crossentropy()
optimizer <- optimizer_rmsprop()
metrics <- list(metric_sparse_categorical_accuracy())
loss_tracking_metric <- metric_mean()

library(reticulate) # need it for %as%. #Q: rexport from keras?
train_step <- function(inputs, targets) {

  with(tf$GradientTape() %as% tape, {
    predictions <- model(inputs, training = TRUE)
    loss <- loss_fn(targets, predictions)
  })
  gradients <- tape$gradient(loss, model$trainable_weights)
  optimizer$apply_gradients(
    purrr::transpose(list(gradients, model$trainable_weights)))

  logs <- list()
  for (metric in metrics) {
    metric$update_state(targets, predictions)
    logs[[metric$name]] <- metric$result()
  }

  loss_tracking_metric$update_state(loss)
  logs$loss <- loss_tracking_metric$result()
  logs
}


# **Writing a step-by-step training loop: resetting the metrics**

# In[ ]:

#
# def reset_metrics():
#     for metric in metrics:
#         metric.reset_state()
#     loss_tracking_metric.reset_state()

reset_metrics <- function() {
  for (metric in metrics)
    metric$reset_state()
  loss_tracking_metric$reset_state()
}


# **Writing a step-by-step training loop: the loop itself**

# In[ ]:

#
# training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
# training_dataset = training_dataset.batch(32)
# epochs = 3
# for epoch in range(epochs):
#     reset_metrics()
#     for inputs_batch, targets_batch in training_dataset:
#         logs = train_step(inputs_batch, targets_batch)
#     print(f"Results at the end of epoch {epoch}")
#     for key, value in logs.items():
#         print(f"...{key}: {value:.4f}")
library(tfdatasets)
training_dataset <-
  tensor_slices_dataset(list(train_images, train_labels)) %>%
  dataset_batch(32)

epochs <- 3
training_dataset_iterator <- as_iterator(training_dataset)
#Q: just add as_iterator() to chain above?
for (epoch in seq(epochs)) {
  reset_metrics()
  c(inputs_batch, targets_batch) %<-%
    iter_next(training_dataset_iterator)
  logs <- train_step(inputs_batch, targets_batch)

  # print(glue("Results at the end of epoch {epoch}"))
  # print(glue_fmt("...{names(logs)}: {value:.4f}"))

  cat(
    sprintf("Results at the end of epoch %s\n", epoch),
    sprintf("...%s: %.4f\n", names(logs), purrr::map_dbl(logs, as.numeric))
  )
}


# **Writing a step-by-step evaluation loop**

# In[ ]:

#
# def test_step(inputs, targets):
#     predictions = model(inputs, training=False)
#     loss = loss_fn(targets, predictions)
#
#     logs = {}
#     for metric in metrics:
#         metric.update_state(targets, predictions)
#         logs["val_" + metric.name] = metric.result()
#
#     loss_tracking_metric.update_state(loss)
#     logs["val_loss"] = loss_tracking_metric.result()
#     return logs
#
# val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
# val_dataset = val_dataset.batch(32)
# reset_metrics()
# for inputs_batch, targets_batch in val_dataset:
#     logs = test_step(inputs_batch, targets_batch)
# print("Evaluation results:")
# for key, value in logs.items():
#     print(f"...{key}: {value:.4f}")

test_step <- function(inputs, targets) {
  predictions <- model(inputs, training = FALSE)
  loss <- loss_fn(targets, predictions)

  logs = list()
  for (metric in metrics) {
    metric$update_state(targets, predictions)
    logs[[paste0("val_", metric$name)]] <-
      metric$result() #Q: wrap in as.array() here? use stringr or glue?
  }

  loss_tracking_metric$update_state(loss)
  logs[["val_loss"]] <- loss_tracking_metric$result()
  logs
}

val_dataset <-
  tensor_slices_dataset(list(val_images, val_labels)) %>%
  dataset_batch(32)

envir::import_from(tfautograph, ag_for)
envir::import_from(purrr, map_dbl, transpose)
#Q: export tfautograph::ag_*?
reset_metrics()
if(FALSE) {
  # tuple unpacking not supported presently in ag_for()
  # just experimenting with syntax here
ag_for(c(inputs_batch, targets_batch), val_dataset, {
    logs <- test_step(inputs_batch, targets_batch)
})
}
ag_for(batch, val_dataset, {
  c(inputs_batch, targets_batch) %<-% batch
  logs <- test_step(inputs_batch, targets_batch)
})
cat("Evaluation results:\n",
    sprintf("...%s: %.4f\n", names(logs), map_dbl(logs, as.numeric)))


# ### Make it fast with tf.function

# **Adding a `tf.function` decorator to our evaluation-step function**

# In[ ]:

#
# @tf.function
# def test_step(inputs, targets):
#     predictions = model(inputs, training=False)
#     loss = loss_fn(targets, predictions)
#
#     logs = {}
#     for metric in metrics:
#         metric.update_state(targets, predictions)
#         logs["val_" + metric.name] = metric.result()
#
#     loss_tracking_metric.update_state(loss)
#     logs["val_loss"] = loss_tracking_metric.result()
#     return logs
#
# val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
# val_dataset = val_dataset.batch(32)
# reset_metrics()
# for inputs_batch, targets_batch in val_dataset:
#     logs = test_step(inputs_batch, targets_batch)
# print("Evaluation results:")
# for key, value in logs.items():
#     print(f"...{key}: {value:.4f}")

zip <- function(...) purrr::transpose(list(...))

map_and_name <- function(.x, .f, ...) {
  out <- purrr::map(.x, .f[-2L], ...)
  names(out) <- purrr::map_chr(.x, .f[-3L], ...)
  out
}
envir::import_from(purrr, walk)

test_step <- tf_function(function(inputs, targets) {
  predictions <- model(inputs, training = FALSE)
  loss <- loss_fn(targets, predictions)

  logs <- list()
  walk(metrics, \(metric) {
    metric$update_state(targets, predictions)
    logs[[paste0("val_", metric$name)]] <<- metric$result()
  })

  loss_tracking_metric$update_state(loss)
  logs$val_loss <- loss_tracking_metric$result()
  logs
})

val_dataset <-
  tensor_slices_dataset(list(val_images, val_labels)) %>%
  dataset_batch(32)

reset_metrics()
#Q: what is the intent of assigning to `logs` here?
val_dataset_iterator <- as_iterator(val_dataset)
while(!is.null(batch <- iter_next(val_dataset_iterator))) {
  c(inputs_batch, targets_batch) %<-% batch
  logs <-test_step(inputs_batch, targets_batch)
}

cat("Evaluation results:\n",
    sprintf("...%s: %.4f\n", names(logs), map_dbl(logs, as.numeric)))



# ### Leveraging fit() with a custom training loop

# **Implementing a custom training step to use with `fit()`**

# In[ ]:

#
# loss_fn = keras.losses.SparseCategoricalCrossentropy()
# loss_tracker = keras.metrics.Mean(name="loss")
#
# class CustomModel(keras.Model):
#     def train_step(self, data):
#         inputs, targets = data
#         with tf.GradientTape() as tape:
#             predictions = self(inputs, training=True)
#             loss = loss_fn(targets, predictions)
#         gradients = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(gradients, model.trainable_weights))
#
#         loss_tracker.update_state(loss)
#         return {"loss": loss_tracker.result()}
#
#     @property
#     def metrics(self):
#         return [loss_tracker]
#

loss_fn <- loss_sparse_categorical_crossentropy()
loss_tracker <- metric_mean(name="loss")


CustomModel(keras$Model) %py_class% {
  train_step <- function(data) {
    c(inputs, targets) %<-% data
    with(tf$GradientTape() %as% tape, {
      predictions <- self(inputs, training = TRUE)
      loss <- loss_fn(targets, predictions)
    })
    gradients <- tape$gradient(loss, model$trainable_weights)
    optimizer$apply_gradients(xyz(gradients, model.trainable_weights))

    loss_tracker$update_state(loss)
    list(loss = loss_tracker$result())
  }

  metrics %<-active% function() list(loss_tracker)
}


# In[ ]:

#
# inputs = keras.Input(shape=(28 * 28,))
# features = layers.Dense(512, activation="relu")(inputs)
# features = layers.Dropout(0.5)(features)
# outputs = layers.Dense(10, activation="softmax")(features)
# model = CustomModel(inputs, outputs)
#
# model.compile(optimizer=keras.optimizers.RMSprop())
# model.fit(train_images, train_labels, epochs=3)

inputs <- layer_input(shape=c(28 * 28))
features <- inputs %>%
  layer_dense(512, activation="relu") %>%
  layer_dropout(0.5)
outputs <- features %>%
  layer_dense(10, activation="softmax")
model <- CustomModel(inputs, outputs)

model %>% compile(optimizer = optimizer_rmsprop())
model %>% fit(train_images, train_labels, epochs = 3)


# In[ ]:

#
# class CustomModel(keras.Model):
#     def train_step(self, data):
#         inputs, targets = data
#         with tf.GradientTape() as tape:
#             predictions = self(inputs, training=True)
#             loss = self.compiled_loss(targets, predictions)
#         gradients = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(gradients, model.trainable_weights))
#         self.compiled_metrics.update_state(targets, predictions)
#         return {m.name: m.result() for m in self.metrics}

CustomModel(keras$Model) %py_class% {
  train_step <- function(data) {
    c(inputs, targets) %<-% data
    with(tf$GradientTape() %as% tape, {
      predictions <- self(inputs, training = TRUE)
      loss <- self$compiled_loss(targets, predictions)
    })
    gradients  <- tape$gradient(loss, model$trainable_weights)
    optimizer$apply_gradients(zip(gradients, model$trainable_weights))
    self$compiled_metrics$update_state(targets, predictions)
    map_and_name(self$metrics, .x$name ~ .x$result())
  }
}


# In[ ]:

#
# inputs = keras.Input(shape=(28 * 28,))
# features = layers.Dense(512, activation="relu")(inputs)
# features = layers.Dropout(0.5)(features)
# outputs = layers.Dense(10, activation="softmax")(features)
# model = CustomModel(inputs, outputs)
#
# model.compile(optimizer=keras.optimizers.RMSprop(),
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=[keras.metrics.SparseCategoricalAccuracy()])
# model.fit(train_images, train_labels, epochs=3)

inputs <- layer_input(shape=c(28 * 28))
features <- inputs %>%
  layer_dense(512, activation="relu") %>%
  layer_dropout(0.5)

outputs <- features %>% layer_dense(10, activation="softmax")
model <- CustomModel(inputs, outputs)

model %>% compile(optimizer = optimizer_rmsprop(),
                  loss = loss_sparse_categorical_crossentropy(),
                  metrics = metric_sparse_categorical_accuracy())
model %>% fit(train_images, train_labels, epochs = 3)


# ## Summary
