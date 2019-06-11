# -*- coding: utf-8 -*-
from __future__ import print_function

import tensorflow as tf
import numpy as np
from networks import *
import h5py

def main():
    # Flag to Train / Test (False means train the network / True means use a checkpoint for making predictions on test set)
    testFlag = True
    
    with tf.device('/device:GPU:0'):        
        #Train Data
        data = np.load('dataset/noisy_train_sines.npy')[:, :, :, np.newaxis]
        #Train Ground Truth Data
        label = np.load('dataset/train_sines.npy')[:, :, :, np.newaxis]
        #Test Data 
        test_data = np.load('dataset/noisy_test_sines.npy')[:, :, :, np.newaxis]
        #Directory to save checkpoints
        chkPointSave = './Network/CNN-MSE/cnn-mse'
        #Directory to save predictions on test set
        output_test_dir = 'output_sines_cnnvgg.npy'
        #Directory to write the loss values
        losses_dir = 'losses_cnnvgg.txt'
        #Batch Size
        batch_size = 128
        
        tf.reset_default_graph()
        input_width = data.shape[1]
        input_height = data.shape[2]
        output_width = label.shape[1]
        output_height = label.shape[2]
        
        # generator networks
        X = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_width, input_height, 1])
        with tf.variable_scope('cnn_model') as scope:
            Y_ = cnn_model(X, padding='same')
            
        ##############GENERATOR PREDICTIONS#####################
        X2 = tf.placeholder(dtype=tf.float32, shape=[1, input_width, input_height, 1])
        with tf.variable_scope('cnn_model', reuse = True) as scope:
            Y_pred = cnn_model(X2, padding='same')
        ########################################################
        real_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, output_width, output_height, 1])
        gen_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn_model')
        mse_cost = tf.reduce_sum(tf.squared_difference(Y_, real_data)) / (2.0 * batch_size)
        
        # generator loss
        gen_cost = mse_cost
        
        # optimizer
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-5, 
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_cost, var_list=gen_params)
        
        ################# MAKE PREDICTIONS ONLY (Comment this part if you want to train the network) ###################
        if(testFlag == True):
            #Directory to Checkpoints Folder
            chkPoint = 'C:/Users/s-moh/2-Neural Networks/project/tf implementation/Neworks_Checkpoints/CNN-MSE/cnn-mse'
            #Directory to save the predictions
            outputFile = 'output_sines_cnnvmse2.npy'
            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(chkPoint))
            output_all = sess.run([Y_pred], feed_dict = {X2: test_data[np.newaxis,0,:,:,:]})[0]
            for i in range(1, len(test_data)):
                output_tmp = sess.run([Y_pred], feed_dict = {X2: test_data[np.newaxis,i,:,:,:]})[0]
                output_all = np.concatenate((output_all, output_tmp), axis = 0)
            np.save(outputFile, output_all)
            return 0
		################################################################################################################
        # training
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        num_epoch = 100
        num_batches = data.shape[0] // batch_size
        saver = tf.train.Saver()
        mseCost = []
        print("Start the Training Process:")
        for iteration in range(num_epoch):   
            i = 0   
            while i < num_batches:
                # generator
                batch_data = np.array(data[i*batch_size:(i+1)*batch_size])
                batch_label = np.array(label[i*batch_size:(i+1)*batch_size])
                _mse_cost, _ = sess.run([mse_cost, gen_train_op], feed_dict={X: batch_data, real_data: batch_label})
                mseCost.append(_mse_cost)
                print('Epoch: %d --> num_batch: %d --> mse_loss: %.6f' % (iteration, i+1, _mse_cost))
                i = i + 1
            saver.save(sess, chkPointSave +repr(iteration)+'.ckpt')
            
        ######################### Make Predictions on the test set ###############################
        output_all = sess.run([Y_pred], feed_dict = {X2: test_data[np.newaxis,0,:,:,:]})[0]
        for i in range(1, len(test_data)):
            output_tmp = sess.run([Y_pred], feed_dict = {X2: test_data[np.newaxis,i,:,:,:]})[0]
            output_all = np.concatenate((output_all, output_tmp), axis = 0)
        np.save(output_test_dir, output_all)
        ##########################################################################################    
        #Save Loss values   
        with open(losses_dir, 'w') as f:
            for mse1 in mseCost:
                f.write("%s\n" % (mse1))
        sess.close()

#Running the app
if __name__ == "__main__":
    main()