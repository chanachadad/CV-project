# -------- Neural Networks for Image Restoration ------ #

# ----------------------------------------------------- #
#                    IMPORT                             #
# ----------------------------------------------------- #
from imageio import imread
from skimage.color import rgb2gray
from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History
import os, random
import numpy as np
from skimage.draw import line
from file import *


# ----------------------------------------------------- #
#                    CONSTANTS                          #
# ----------------------------------------------------- #

MAX_PIXEL_NUMBER = 255
GRAY_REPRESENTATION = 1
RGB_REPRESENTATION = 2

# ----------------------------------------------------- #
#                    FUNCTIONS                          #
# ----------------------------------------------------- #


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.
    path: path to a directory to search for images.
    use_shuffle: option to shuffle order of files. Uses a fixed shuffled order.
    """
    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']
    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """Returns a list of image paths to be used for image denoising"""
    return list_images(relpath('image_dataset/train'), True)


def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.
    kernel_size: the height and width of the kernel. Controls strength of blur.
    angle: angle in the range [0, np.pi) for the direction of the motion.
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size-1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2*half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size-1 - p1[0], kernel_size-1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1-norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2*half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size-1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel


def read_image(filename, representation):
    """
     reads an image file and converts it into a given representation
    :return:  an image, represented by a matrix of type np.float64 with intensities normalized to the range [0, 1].
   """
    if (representation != GRAY_REPRESENTATION) and (representation != RGB_REPRESENTATION):
        return False
    im = imread(filename)
    im_float = im.astype(np.float64)
    im_float /= MAX_PIXEL_NUMBER
    if representation == GRAY_REPRESENTATION:
        im_float = rgb2gray(im_float)
    return im_float


# ------------------------------------ Dataset handling -------------------------------------- #

def crop_random(crop_size_x, crop_size_y, image, corrupted_im=None):
    """
    return cropped image in a random valid location according to the requirements.
    """
    h, w = image.shape
    limit_x, limit_y = h - crop_size_x, w - crop_size_y
    start_x = random.randint(0, limit_x)
    start_y = random.randint(0, limit_y)
    cropped_im = image[start_x: start_x + crop_size_x, start_y: start_y + crop_size_y]
    if corrupted_im is not None:
        corrupted_im = corrupted_im[start_x: start_x + crop_size_x, start_y: start_y + crop_size_y]
    return cropped_im, corrupted_im


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    This function create a data_generator, which outputs random tuples of the form (source_batch, target_batch).
    :return: data_generator, which outputs random tuples of the form (source_batch, target_batch),
     where each output variable is an array of shape (batch_size, height, width, 1)
    """
    cache = {}  # create an empty dictionary.
    height, width = crop_size
    while True:
        source_batch = np.zeros((batch_size, height, width, 1))
        target_batch = np.zeros((batch_size, height, width, 1))
        counter = 0

        while counter < batch_size:
            random_filename = random.choice(filenames)
            if random_filename in cache:
                im = cache[random_filename]
            else:
                im = read_image(random_filename, GRAY_REPRESENTATION)
                cache[random_filename] = im  # insert the image to the cache.

            im_to_corrupt = crop_random(3*height, 3*width, im)[0]
            corrupted_im = corruption_func(im_to_corrupt)
            target, source = crop_random(height, width, im_to_corrupt, corrupted_im)
            source_batch[counter, :, :, 0] = source - 0.5
            target_batch[counter, :, :, 0] = target - 0.5
            counter += 1

        yield (source_batch, target_batch)


# -------------------------------- Neural Network Model -------------------------------------- #

def resblock(input_tensor, num_channels):
    """
    create a symbolic output tensor according to the input and num_channels.
    :return: symbolic output tensor of the layer configuration.
    """
    step_1_conv = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    step_2_relu = Activation('relu')(step_1_conv)
    step_3_conv = Conv2D(num_channels, (3, 3), padding='same')(step_2_relu)
    output = Add()([input_tensor, step_3_conv])
    return Activation('relu')(output)


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    this function create an untraind model according to the given values.
    :return: untraind model according
    """
    # create a tensor for the initial input.
    input_tensor = Input(shape=(height, width, 1))
    step_1_conv = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    step_2_relu = Activation('relu')(step_1_conv)
    residual_block = resblock(step_2_relu, num_channels)

    for i in range(num_res_blocks - 1):
        residual_block = resblock(residual_block, num_channels)

    last_conv = Conv2D(1, (3, 3), padding='same')(residual_block)
    output = Add()([last_conv, input_tensor])
    return Model(inputs=input_tensor, outputs=output)


# -------------------------- Training Networks for Image Restoration -------------------------- #

def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    this function train a model with the appropriate data sets.
    First we divide the images into a training set and validation set, using an 80-20 split,
    after that we generate from each set a data set with the given batch size and corruption function.
    At the end, we call to the compile() method of the model using the “mean squared error” loss and ADAM optimizer
    :return: a trained model.
    """
    # create a training and validation sets:
    split_index = int(0.8*len(images))
    training_set = images[:split_index]     # 80%
    validation_set = images[split_index:]   # 20%

    training_generator = load_dataset(training_set, batch_size, corruption_func, model.input_shape[1:3])
    validation_generator = load_dataset(validation_set, batch_size, corruption_func, model.input_shape[1:3])

    adam = Adam(beta_2=0.9)
    history = History()
    model.compile(optimizer=adam, loss='mean_squared_error')
    model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=validation_generator, validation_steps=num_valid_samples/batch_size,
                        callbacks=[history])


# -------------------------- Image Restoration of Complete Images ------------------------------ #

def restore_image(corrupted_im, base_model):
    """
    This function restore full images of any size, using th given base model.
    :return: the restored image.
    """

    height, width = corrupted_im.shape
    input_tensor = Input(shape=(height, width, 1))
    output_tensor = base_model(input_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # modify the values of the image to the requirement of the model- range [-0.5, 0.5]
    corrupted_im = np.array([corrupted_im - 0.5])
    corrupted_im = np.expand_dims(corrupted_im, axis=3)
    restored_im = model.predict(corrupted_im)
    restored_im = restored_im + 0.5
    restored_im = restored_im.reshape(restored_im.shape[1], restored_im.shape[2])
    restored_im = np.clip(restored_im, 0, 1).astype(np.float64)
    return restored_im


# ---------------------------------- Image Denoising ------------------------------------------ #

def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    This function add a random noise to the given image.
    :return: corrupted image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    corrupted_im = image + np.random.normal(0, sigma, image.shape)
    rounded_im = (np.round(corrupted_im * MAX_PIXEL_NUMBER)) / MAX_PIXEL_NUMBER
    return np.clip(rounded_im, 0, 1).astype(np.float64)


def learn_denoising_model(num_res_blocks=5):
    """
    This function train a network which expect patches of size 24×24, using 48 channels for all but the last layer.
    :return: a trained denoising model
    """
    image_paths = images_for_denoising()
    model = build_nn_model(24, 24, 48, num_res_blocks)
    train_model(model, image_paths, lambda image: add_gaussian_noise(image, 0, 0.2), 100, 100, 5, 1000)
    return model

