#!/usr/bin/env python

# This is a companion notebook for the book [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-R-second-edition). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.7.

# ## DeepDream

# ### Implementing DeepDream in Keras

# **Fetching the test image**

# In[ ]:

#
# from tensorflow import keras
# import matplotlib.pyplot as plt
#
# base_image_path = keras.utils.get_file(
#     "coast.jpg", origin="https://img-datasets.s3.amazonaws.com/coast.jpg")
#
# plt.axis("off")
# plt.imshow(keras.utils.load_img(base_image_path))

library(tensorflow)
library(keras)

base_image_path <- keras::get_file(
    "coast.jpg", origin="https://img-datasets.s3.amazonaws.com/coast.jpg")

plot(as.raster(jpeg::readJPEG(base_image_path)))

# **Instantiating a pretrained `InceptionV3` model**

# In[ ]:

#
# from tensorflow.keras.applications import inception_v3
# model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

model <- application_inception_v3(weights = "imagenet", include_top = FALSE)


# **Configuring the contribution of each layer to the DeepDream loss**

# In[ ]:

#
# layer_settings = {
#     "mixed4": 1.0,
#     "mixed5": 1.5,
#     "mixed6": 2.0,
#     "mixed7": 2.5,
# }
# outputs_dict = dict(
#     [
#         (layer.name, layer.output)
#         for layer in [model.get_layer(name) for name in layer_settings.keys()]
#     ]
# )
# feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

layer_settings <- list(
  "mixed4" = 1.0,
  "mixed5" = 1.5,
  "mixed6" = 2.0,
  "mixed7" = 2.5
)
library(purrr)
outputs_dict <- names(layer_settings) %>%
  set_names() %>%
  map(\(name) model$get_layer(name)$output)

feature_extractor <-
  keras_model(inputs = model$inputs, outputs = outputs_dict)


# **The DeepDream loss**

# In[ ]:

# def compute_loss(input_image):
#     features = feature_extractor(input_image)
#     loss = tf.zeros(shape=())
#     for name in features.keys():
#         coeff = layer_settings[name]
#         activation = features[name]
#         loss += coeff * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :]))
#     return loss

compute_loss <- function(input_image) {
  features <- feature_extractor(input_image)
  loss <- tf$zeros(shape = shape())
  for (name in names(features)) {
    coeff <- layer_settings[[name]]
    activation <- features[[name]]
    loss <- loss +
      (coeff * tf$reduce_mean(tf$square(activation[, 3:-3, 3:-3, ])))
  }
  loss
}


# **The DeepDream gradient ascent process**

# In[ ]:

# import tensorflow as tf
#
# @tf.function
# def gradient_ascent_step(image, learning_rate):
#     with tf.GradientTape() as tape:
#         tape.watch(image)
#         loss = compute_loss(image)
#     grads = tape.gradient(loss, image)
#     grads = tf.math.l2_normalize(grads)
#     image += learning_rate * grads
#     return loss, image
#
#
# def gradient_ascent_loop(image, iterations, learning_rate, max_loss=None):
#     for i in range(iterations):
#         loss, image = gradient_ascent_step(image, learning_rate)
#         if max_loss is not None and loss > max_loss:
#             break
#         print(f"... Loss value at step {i}: {loss:.2f}")
#     return image


gradient_ascent_step <- tf_function(
  function(image, learning_rate) {
    with(tf$GradientTape() %as% tape, {
      tape$watch(image)
      loss <- compute_loss(image)
    })
    grads <- tape$gradient(loss, image)
    grads <- tf$math$l2_normalize(grads)
    image <- image + (learning_rate * grads)
    list(loss, image)
  })



gradient_ascent_loop <-
  function(image, iterations, learning_rate, max_loss = NULL) {
    for (i in seq(iterations)) {
      c(loss, image) %<-% gradient_ascent_step(image, learning_rate)
      loss <- as.numeric(loss) # unwrap tensor value
      if (!is.null(max_loss) && loss > max_loss)
        break
      cat(sprintf("... Loss value at step %i: %.2f\n", i, loss))
    }
    image
  }

# In[ ]:

#
# step = 20.
# num_octave = 3
# octave_scale = 1.4
# iterations = 30
# max_loss = 15.

step <- 20.
num_octave <- 3
octave_scale <- 1.4
iterations <- 30
max_loss <- 15.


# **Image processing utilities**

# In[ ]:

#
# import numpy as np
#
# def preprocess_image(image_path):
#     img = keras.utils.load_img(image_path)
#     img = keras.utils.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = keras.applications.inception_v3.preprocess_input(img)
#     return img
#
# def deprocess_image(img):
#     img = img.reshape((img.shape[1], img.shape[2], 3))
#     img /= 2.0
#     img += 0.5
#     img *= 255.
#     img = np.clip(img, 0, 255).astype("uint8")
#     return img


preprocess_image <- function(image_path) {
  image_path %>%
    image_load() %>%
    image_to_array() %>%
    k_expand_dims(axis = 1) %>%
    inception_v3_preprocess_input()
}

deprocess_image <- function(img) {
  img %>%
    k_reshape(c(dim(img)[2:3], 3)) %>%
    `/`(2.0) %>%
    `+`(.5) %>%
    `*`(255) %>%
    k_clip(0, 255) %>%
    k_cast("uint8")
  # TODO: think about adding an as.raster() method for tensors
}


# **Running gradient ascent over multiple successive "octaves"**

# In[ ]:

#
# original_img = preprocess_image(base_image_path)
# original_shape = original_img.shape[1:3]
#
# successive_shapes = [original_shape]
# for i in range(1, num_octave):
#     shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
#     successive_shapes.append(shape)
# successive_shapes = successive_shapes[::-1]
#
# shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])
#
# img = tf.identity(original_img)
# for i, shape in enumerate(successive_shapes):
#     print(f"Processing octave {i} with shape {shape}")
#     img = tf.image.resize(img, shape)
#     img = gradient_ascent_loop(
#         img, iterations=iterations, learning_rate=step, max_loss=max_loss
#     )
#     upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
#     same_size_original = tf.image.resize(original_img, shape)
#     lost_detail = same_size_original - upscaled_shrunk_original_img
#     img += lost_detail
#     shrunk_original_img = tf.image.resize(original_img, shape)
#
# keras.utils.save_img("dream.png", deprocess_image(img.numpy()))

`append1<-` <- function (x, value) {
    stopifnot(is.list(x) || identical(mode(x), mode(value)))
    x[[length(x) + 1L]] <- value
    x
}

original_img <- preprocess_image(base_image_path)
original_shape <- original_img$shape[2:3]

successive_shapes <- list(original_shape)
for (i in 1:(num_octave - 1)) {
  shp <- lapply(original_shape, function(dim)
    as.integer(dim / (octave_scale ^ i)))
  append1(successive_shapes) <- shape(dims = shp)
}

# successive_shapes <- rev(successive_shapes)
library(magrittr, include.only = "%<>%")
successive_shapes %<>% rev()

shrunk_original_img <-
  tf$image$resize(original_img, successive_shapes[[1]])

img <- tf$identity(original_img)
for (i in seq_along(successive_shapes)) {
  shp <- successive_shapes[[i]]
  # browser()
  cat(sprintf("Processing octave %i with shape %s\n", i, format(shp)))
  img <- tf$image$resize(img, shp)
  img <- gradient_ascent_loop(
    img,
    iterations = iterations,
    learning_rate = step,
    max_loss = max_loss
  )
  upscaled_shrunk_original_img <-
    tf$image$resize(shrunk_original_img, shp)
  same_size_original <- tf$image$resize(original_img, shp)
  lost_detail <- same_size_original - upscaled_shrunk_original_img
  img <- img + lost_detail
  shrunk_original_img <- tf$image$resize(original_img, shp)
}

img <- deprocess_image(img)
image_array_save(img, "dream.png")
plot(as.raster(as.array(img), max = 255))


# ### Wrapping up

