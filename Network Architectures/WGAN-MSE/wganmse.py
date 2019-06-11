import tensorflow as tf
import numpy as np
from networks import *

def main():
    # Flag to Train / Test (False means train the network / True means use a checkpoint for making predictions on test set)
    testFlag = True
    # Factors multiplied by the MSE Loss and Gradient Penalty Loss (Review the original paper for its meaning)
    lambd = 1
    lambd1 = 1
    with tf.device('/device:GPU:0'):
        #Train Data
        data = np.load('dataset/noisy_train_sines.npy')[:, :, :, np.newaxis]
        #Train Ground Truth Data
        label = np.load('dataset/train_sines.npy')[:, :, :, np.newaxis]
        #Test Data 
        test_data = np.load('dataset/noisy_test_sines.npy')[:, :, :, np.newaxis]
        #Directory to save checkpoints
        chkPointSave = './Network/WGAN-MSE/wgan-mse'
        #Directory to save predictions on test set
        output_test_dir = 'output_sines_wganmse.npy'
        #Directory to write the loss values
        losses_dir = 'losses_wganmse.txt'
        #Batch Size
        batch_size = 128

        tf.reset_default_graph()        
        input_width = data.shape[1]
        input_height = data.shape[2]
        output_width = label.shape[1]
        output_height = label.shape[2]
        
        print('------------ FINISHED READING DATA -----------')
        batch_size = 128

		# generator networks
        X = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_width, input_height, 1])
        with tf.variable_scope('cnn_model') as scope:
            Y_ = cnn_model(X, padding='same')

		##############GENERATOR PREDICTIONS#####################
        X2 = tf.placeholder(dtype=tf.float32, shape=[1, input_width, input_height, 1])
        with tf.variable_scope('cnn_model', reuse = True) as scope:
            Y_pred = cnn_model(X2, padding='same')
		########################################################
			
		# discriminator networks
        real_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, output_width, output_height, 1])
        alpha = tf.random_uniform(shape=[batch_size,1], minval=0., maxval=1.)
        
        with tf.variable_scope('discriminator_model') as scope:
            disc_real = discriminator_model(real_data)
            scope.reuse_variables()
            disc_fake = discriminator_model(Y_)
            interpolates = alpha*tf.reshape(real_data, [batch_size, -1]) + (1-alpha)*tf.reshape(Y_, [batch_size, -1])
            interpolates = tf.reshape(interpolates, [batch_size, output_width, output_height, 1])
            gradients = tf.gradients(discriminator_model(interpolates), [interpolates])[0]
        
        # generator loss
        gen_cost = tf.reduce_mean(disc_fake)    
        # discriminator loss
        w_distance = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)  
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost = - w_distance + lambd * gradient_penalty   # add gradient constraint to discriminator loss
        
        gen_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn_model')
        disc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator_model')
        print('------------ FINISHED DEFINING DISCRIMINATOR -----------')

        mse_cost = tf.reduce_sum(tf.squared_difference(Y_, real_data)) / (2.0 * batch_size)
        
        # generator loss
        gen_cost = gen_cost + lambd1 * mse_cost

		# optimizer
        gen_train_op = tf.train.AdamOptimizer(learning_rate = 1e-5, beta1 = 0.5, beta2 = 0.9).minimize(gen_cost, var_list = gen_params)
        disc_train_op = tf.train.AdamOptimizer(learning_rate = 1e-5, beta1 = 0.5, beta2 = 0.9).minimize(disc_cost, var_list = disc_params)
        
        ########################## PREDICTIONS (Comment this part if you want to train the network) ######################
        if(testFlag == True):
            #Directory to Checkpoints Folder
            chkPoint = 'C:/Users/s-moh/2-Neural Networks/project/tf implementation/Neworks_Checkpoints/WGAN-MSE/wgan-mse'
            #Directory to save the predictions
            outputFile = 'output_sines_wganmse.npy'
            sess = tf.Session()
            saver = tf.train.Saver()
            #Directory to Checkpoints Folder
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
        disc_iters = 4 
        num_batches = data.shape[0] // batch_size
        saver = tf.train.Saver()
        
        genCost = []
        wCost = []
        discCost = []
        
        print("Start the Training Process:")
        for iteration in range(num_epoch):
            i = 0
            while i < num_batches:
                # discriminator
                for j in range(disc_iters):
                    index = np.random.randint(0, data.shape[0]-batch_size, size=1)[0]
                    batch_data = np.array(data[index:(index+batch_size)])
                    batch_label = np.array(label[index:(index+batch_size)])
                    _disc_cost, _w_distance, _ = sess.run([disc_cost, w_distance, disc_train_op], feed_dict={real_data: batch_label, X: batch_data})

				# generator
                batch_data = np.array(data[i*batch_size:(i+1)*batch_size])
                batch_label = np.array(label[i*batch_size:(i+1)*batch_size])
                
                _gen_cost, _ = sess.run([gen_cost, gen_train_op], feed_dict={X: batch_data, real_data: batch_label})
                genCost.append(_gen_cost)
                wCost.append(_w_distance)
                discCost.append(_disc_cost)
                print('Epoch: %d --> Batch: %d --> genLoss: %.6f --> discLoss: %.6f --> wDistance: %.6f' % (iteration, i+1, _gen_cost, _disc_cost, _w_distance))
                i = i + 1	
            saver.save(sess, chkPointSave + repr(iteration) + '.ckpt')
				
		######################### Make Predictions on the test set ###############################
        output_all = sess.run([Y_pred], feed_dict = {X2: test_data[np.newaxis,0,:,:,:]})[0]
        for i in range(1, len(test_data)):
            output_tmp = sess.run([Y_pred], feed_dict = {X2: test_data[np.newaxis,i,:,:,:]})[0]
            output_all = np.concatenate((output_all, output_tmp), axis = 0)
        np.save(output_test_dir, output_all)
		##########################################################################################    
        #Save Loss values
        with open(losses_dir, 'w') as f:
            for gen1, w1, disc1 in zip(genCost, wCost, discCost):
                f.write("%s,%s,%s\n" % (gen1, w1, disc1))
				
        sess.close()
#Running the app
if __name__ == "__main__":
    main()