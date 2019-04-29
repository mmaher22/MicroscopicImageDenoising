# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:44:07 2019
@author: Mohamed Maher
"""
from __future__ import print_function, division

import numpy as np
#import h5py
#import sys
from keras.layers import Input, Dense, Conv2D, Flatten, LeakyReLU, Concatenate
from keras.applications.vgg19 import VGG19
from keras.models import Model, Sequential
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from functools import partial
import keras.backend as K


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANVGG():
    def __init__(self):
        #batch size = 128 in original paper
        self.batch_size = 64
        self.imgRows = 64
        self.imgCols = 64
        self.channels = 1
        self.imgShape = (self.imgRows, self.imgCols, self.channels)
        # number of discriminator iterations for each generator iteration
        self.discIters = 4
        # Optimizer --> Adam, Alpha = 1e-5, Beta1 = 0.5, Beta2 = 0.9 (in original paper)
        optimizer = Adam(lr = 1e-4, beta_1 = 0.5, beta_2 = 0.9)
        # Build the generator, VGG and discriminator models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.vgg = self.build_vgg19()
        
        # Image input (High Quality Image)
        realImg = Input(shape = self.imgShape)
        # Noise input (Noisy Image)
        zImg = Input(shape = self.imgShape)
        # Generate image based on noisy Image
        fakeImg = self.generator(zImg)
        
        #----------------------------------
        # Construct Graph of Discriminator
        #----------------------------------
        # Freeze generator's layers while training discriminator
        self.generator.trainable = False
        # Discriminator determines validity of the real and generated images
        discFake = self.discriminator(fakeImg)
        discReal = self.discriminator(realImg)
        # Construct weighted average between real and fake images
        interpolatedImg = RandomWeightedAverage()([realImg, fakeImg])
        # Use Python partial to provide loss function with additional 'averaged_samples' argument
        partialGpLoss = partial(self.gradient_penalty_loss, averaged_samples = interpolatedImg)
        partialGpLoss.__name__ = 'gradient_penalty' # Keras requires function names
        #w_distance loss
        wDisLoss = K.mean(discFake) - K.mean(discReal)
        wDisLoss.__name__ = 'wdis_penalty' # Keras requires function names
        
        # Determine validity of weighted sample
        validityInterpolated = self.discriminator(interpolatedImg)
        self.discriminatorModel = Model(inputs = [realImg, zImg], outputs = [discReal, discFake, validityInterpolated])
        
        #----------------------------------
        # Construct VGG Loss
        #----------------------------------
        #Output from VGG network
        real_conc = Concatenate()([realImg, realImg, realImg]) #VGG need 3 channel images
        fake_conc = Concatenate()([fakeImg, fakeImg, fakeImg]) #VGG need 3 channel images
        vggReal = self.vgg(real_conc)
        vggFake = self.vgg(fake_conc)
        #VGG Loss
        vggLoss = K.sum(K.square(vggReal - vggFake)) / 2.0 / self.batch_size
        print('vggLoss: ', vggLoss)
        vggLoss.__name__ = 'vgg_penalty' # Keras requires function names
        
        #-------------------------------
        # Construct Graph of Generator
        #-------------------------------
        # For the generator we freeze the discriminator layers
        self.discriminator.trainable = False
        self.generator.trainable = True
        #Generator Loss
        genLoss = -1.0 * K.mean(discFake)
        genLoss.__name__ = 'gen_penalty' # Keras requires function names
        # Defines generator model
        self.generatorModel = Model(inputs = zImg, output = [discFake, fake_conc])
        #Compile
        self.generatorModel.compile(loss = genLoss + 0.1 * vggLoss, optimizer = optimizer)
        self.discriminatorModel.compile(loss = wDisLoss + partialGpLoss, optimizer = optimizer)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
    
    # Create Generator model
    def build_generator(self):
        model = Sequential()
        # Add model layers
        # First to Seventh Layers (n32-s1)
        model.add(Conv2D(32, kernel_size = 3, padding = "same", activation = "relu"))
        model.add(Conv2D(32, kernel_size = 3, padding = "same", activation = "relu"))
        model.add(Conv2D(32, kernel_size = 3, padding = "same", activation = "relu"))
        model.add(Conv2D(32, kernel_size = 3, padding = "same", activation = "relu"))
        model.add(Conv2D(32, kernel_size = 3, padding = "same", activation = "relu"))
        model.add(Conv2D(32, kernel_size = 3, padding = "same", activation = "relu"))
        model.add(Conv2D(32, kernel_size = 3, padding = "same", activation = "relu"))
        # Eighth Layer (n1-s1)
        model.add(Conv2D( 1, kernel_size = 3, padding = "same", activation = "relu"))
        
        noiseIn = Input(shape = self.imgShape)
        fakeOut = model(noiseIn)
        #model summary
        #model.summary()
        
        return Model(noiseIn, fakeOut)
    
    # VGG19 Network for feature Extraction
    def build_vgg19(self):
        # create model
        #If include_top = True --> shape should be (224, 224, 3)
        vgg = VGG19(input_shape = (64,64,3), include_top=False, weights='imagenet')
        #vgg summary
        #vgg.layers[-6].summary()
        
        return Model(inputs = vgg.get_input_at(0), outputs = vgg.layers[-6].get_output_at(0))

    def build_discriminator(self):
        model = Sequential()
        # Add model layers
        # First and Second Layers (n64-s1) + (n64-s2)
        model.add(Conv2D(64, kernel_size = 3, padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        # Third and Fourth Layers (n128-s1) + (n128-s2)
        model.add(Conv2D(128, kernel_size = 3, padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        # Fifth and Sixth Layers (n256-s1) + (n256-s2)
        model.add(Conv2D(256, kernel_size = 3, padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(256, kernel_size = 3, strides = 2, padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        # Flatten Layer
        model.add(Flatten())
        # First FC Layer (output = 1024)
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha = 0.2))
        # Second FC Layer (output = 1)
        model.add(Dense(1))
        
        img = Input(shape = self.imgShape)
        validity = model(img)
        #model.summary()
        return Model(img, validity)

    def train(self, epochs):
        # Load the dataset
        data = np.load('noisySample.npy') #Noisy Images
        label = np.load('originalSample.npy') #High Quality Images
        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty
        numBatches = data.shape[0] // self.batch_size
        
        for epoch in range(epochs):
            for i in numBatches:
                for _ in range(self.discIters):
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    # Select a random batch of images
                    index = np.random.randint(0, data.shape[0] - self.batch_size, size = 1)[0]
                    batchLbls = np.array(label[index:(index + self.batch_size)])
                    batchData = np.array(data [index:(index + self.batch_size)])
                    
                    # Train the discriminator
                    _dLoss, _, _ = self.discriminatorModel.train_on_batch([batchLbls, batchData], [valid, fake, dummy])

                # ---------------------
                #  Train Generator
                # ---------------------
                batchLbls = np.array(label [i * self.batch_size:(i + 1) * self.batch_size])
                batchData = np.array(data [i * self.batch_size:(i + 1) * self.batch_size])
                _gLoss, _ = self.generatorModel.train_on_batch([batchData], valid)
                # Plot the progress
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, _dLoss[0], _gLoss))

            self.generatorModel.save_weights('Checkpoints/wgan_vgg_gen' + repr(epoch) + '.h5')
            self.discriminatorModel.save_weights('Checkpoints/wgan_vgg_disc' + repr(epoch) + '.h5')
            print("Checkpoint: Saved model to disk")

if __name__ == '__main__':
    wgan = WGANVGG()
    #Epochs = 100
    wgan.train(epochs = 100)