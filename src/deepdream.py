'''
This is a Tensorflow 2.0 implementation of Google's Deepdream by Timothy Nguyen.

cited:
https://github.com/keras-team/keras/blob/master/examples/deep_dream.py
https://colab.research.google.com/github/random-forests/applied-dl/blob/master/examples/9-deep-dream-minimal.ipynb?fbclid=IwAR2IrpxyA6CmqV5p1UVixzuAh84wh_L61uKOxExaxtMWaP52PF6KeRpndBw#scrollTo=q_Nrh8thxye5
'''


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import inception_v3
import scipy


K.set_learning_phase(0)


def dream(image_path, # path to image upon to dream
          model, # pretrained model for dreaming
          layer_contributions, # dict, names of layers to use in dreaming and weights to moderate their effects
          output_path = None,
          step = 0.01,  # Gradient ascent step size
          octaves = 3,  # Number of scales at which to run gradient ascent
          octave_scale = 1.4,  # Size ratio between scales
          iterations = 20,  # Number of ascent steps per scale
          max_loss = 10., # loss limit at which we interrupt gradient ascent to avoid ugly artifacts
          verbose = False):
    # Load the image into a Numpy array
    img = _preprocess_image(image_path)

    # We prepare a list of shape tuples defining the different scales at which we will run gradient ascent
    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, octaves):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)

    # Reverse list of shapes, so that they are in increasing order
    successive_shapes = successive_shapes[::-1]

    # Resize the Numpy array of the image to our smallest scale
    original_img = np.copy(img)
    shrunk_original_img = _resize_img(img, successive_shapes[0])

    for shape in successive_shapes:
        if verbose:
            print('Processing image shape', shape)
        img = _resize_img(img, shape)
        img = _gradient_ascent(img,
                              model,
                              layer_contributions,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss,
                              verbose=verbose)
        upscaled_shrunk_original_img = _resize_img(shrunk_original_img, shape)
        same_size_original = _resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img.assign_add(lost_detail)
        shrunk_original_img = _resize_img(original_img, shape)

    if output_path:
        try:
            _save_img(img, fname=output_path)
        except:
            pass
    
    return _deprocess_image(img)


def _gradient_ascent(x, model, layer_contributions, iterations, step, max_loss=None, verbose = False):
    x = tf.Variable(x)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            loss_value = _forward_loss(x, model, layer_contributions)
            
        grad_values = tape.gradient(loss_value, x)
        grad_values /= grad_values.numpy().std() + 1e-8 
        if verbose:
            print('..Loss value at', i, ':', loss_value)
        if max_loss is not None and loss_value > max_loss:
            break
        x.assign_add(step * grad_values)
    return x


def _forward_loss(x, model, layer_contributions):
    total_loss = 0
    activations = model(x)
    for i, act in enumerate(activations):
        coeff = list(layer_contributions.values())[i]
        loss = tf.norm(act) / K.prod(K.cast(K.shape(act), 'float32'))
        total_loss += loss
    return total_loss


def _save_img(img, fname):
    pil_img = _deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


def _preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def _deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    x = x.numpy()
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def _resize_img(img, size):
    if type(img) is not np.ndarray:
        img = img.numpy()
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)


