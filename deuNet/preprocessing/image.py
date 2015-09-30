import numpy as np
import re
from scipy import ndimage
from scipy import linalg

from os import listdir
from os.path import isfile, join
from PIL import Image
import random, math

"""
Image Data augmentation. Derived from Keras
github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
"""

def random_rotation(x, rg, fill_mode='nearest',cval=0):
    """
    Rotate the image
    x: input image ndarray
    rg: rotation range: 0 - 180 degree

    !!!IMPORTANT: We assume x is a 3d np.array, dim-0 is channel, dim-1 and dim-2 are width and height
    
    """
    angle = random.uniform(-rg, rg)
    x = ndimage.interpolation.rotate(x, angle, axes=(1,2), reshape=False,mode=fill_mode, cval=cval)
    return x


def random_shift(x, wrg, hrg, fill_mode='nearest',cval=0.):
    """
    Shift the image
    x: input image ndarray
    wrg: width shift range (fraction of total width)
    hrg: height shift range (fraction of total height)

    !!!IMPORTANT: We assume x is a 3d np.array, dim-0 is channel, dim-1 and dim-2 are width and height
    
    """
    crop_left_pixels = 0
    crop_right_pixels = 0
    crop_top_pixels = 0
    crop_bottom_pixels = 0
    
    original_w = x.shape[1]
    original_h = x.shape[2]

    if wrg:
        crop = random.uniform(0., wrg)
        split = random.uniform(0., 1)
        crop_left_pixels = int(split*crop*x.shape[1])
        crop_right_pixels = int((1-split)*crop*x.shape[1])

    if hrg:
        crop = random.uniform(0., hrg)
        split = random.uniform(0, 1)
        crop_top_pixels = int(split*crop*x.shape[2])
        crop_bottom_pixels = int((1-split)*crop*x.shape[2])

    x = ndimage.interpolation.shift(x, (0, crop_left_pixels, crop_top_pixels), mode = fill_mode, cval = cval)

    return x


def horizontal_flip(x):
    for i in range(x.shape[0]):
        x[i] = np.fliplr(x[i])

    return x

def vertical_flip(x):
    for i in range(x.shape[1]):
        x[i] = np.flipud(x[i])

    return x


def array_to_img(x, scale=True):
    x = x.transpose(1,2,0) # in the image settings, dim-2 is the channel
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype("uint8"), "RGB")
    else:
        # grayscale
        return Image.fromarray(x[:,:,0].astype("uint8"),"L")


def img_to_array(img):
    x = np.asarray(img, dypte='float32')
    if len(x.shape)==3:
        # RGB: height, width, channel -> channel, height, width
        x = x.transpose(2,0,1)
    else:
        # grayscale: height, width -> channel, height, width
        x = x.reshape((1,x.shape[0],x.shape[1]))
    
    return x


def load_img(path, grayscale=False):
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [join(directory,f) for f in listdir(directory) \
            if isfile(join(directory,f)) and re.match('([\w]+\.(?:' + ext + '))', f)]



class ImageDataGenerator(object):
    """
    generate minibatches with
    realtime data augmentation
    """
    def __init__(self,
            featurewise_center = True, # set iput mean to 0 over the dataset
            samplewise_center = False, # set each sample mean to 0
            featurewise_std_normalization = True, # divide inputs by std of the dataset
            samplewise_std_normalization = False, # divide each input by its std
            
            zca_whitening = False, # apply ZCA whitening
            rotation_range = 0., # degree (0 - 180)
            width_shift_range = 0, # fraction of total width
            height_shift_range = 0, # fraction of total height
            horizontal_flip = False, 
            vertical_flip = False):
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None

    def flow(self, X, y, batch_size=32,
            shuffle=False, seed=None, 
            save_to_dir=None, save_prefix="",save_format='jpeg',transform=True):
        if seed:
            random.seed(seed)

        if shuffle:
            seed = random.randint(1, 10e6)
            np.random.seed(seed)
            np.random.shuffle(X)
            np.random.seed(seed)
            np.random.shuffle(y)

        nb_batch = int(math.ceil(float(X.shape[0])/batch_size))
        for b in range(nb_batch):
            batch_end = (b+1)*batch_size
            if batch_end > X.shape[0]:
                nb_samples = X.shape[0] - b*batch_size
            else:
                nb_samples = batch_size

            bX = np.zeros(tuple([nb_samples]+list(X.shape)[1:]))
            for i in range(nb_samples):
                x = X[b*batch_size + i]
                if transform:
                    x = self.random_transform(x.astype("float32"))
                x = self.standardize(x)
                bX[i] = x

            if save_to_dir:
                for i in range(nb_samples):
                    img = array_to_img(bX[i], scale=True)
                    img.save(save_to_dir + "/" + save_prefix + "_" + str(i) + "." + save_format)
            
            yield bX, y[b*batch_size:b*batch_size+nb_samples]


    def standardize(self, x):
        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= self.std

        if self.zca_whitening:
            flatx = np.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2]))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0],x.shape[1], x.shape[2]))

        if self.samplewise_center:
            x -= np.mean(x)
        if self.samplewise_std_normalization:
            x /= np.std(x)

        return x

    def random_transform(self, x):
        if self.rotation_range:
            x = random_rotation(x, self.rotation_range)
        if self.width_shift_range or self.height_shift_range:
            x = random_shift(x,self.width_shift_range, self.height_shift_range)
        if self.horizontal_flip:
            if random.random() < 0.5:
                x = horizontal_flip(x)
        if self.vertical_flip:
            if random.random() < 0.5:
                x = vertical_flip(x)

        return x


    def fit(self, X, seed=None):
        """
            Required for centering, normalization, and whitening.
        """
        X = np.copy(X)

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= self.std

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))
            fudge = 10e-6
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + fudge))), U.T)


