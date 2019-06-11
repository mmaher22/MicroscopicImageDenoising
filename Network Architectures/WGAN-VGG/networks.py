import tensorflow as tf

def leaky_relu(inputs, alpha):
    return 0.5 * (1 + alpha) * inputs + 0.5 * (1-alpha) * tf.abs(inputs)

def cnn_model(inputs, padding='same'):
    outputs = tf.layers.conv2d(inputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1', use_bias=False)
    outputs = tf.nn.relu(outputs)
    
    outputs = tf.layers.conv2d(outputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2', use_bias=False)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3', use_bias=False)
    outputs = tf.nn.relu(outputs)
    
    outputs = tf.layers.conv2d(outputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4', use_bias=False)
    outputs = tf.nn.relu(outputs)
    
    outputs = tf.layers.conv2d(outputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5', use_bias=False)
    outputs = tf.nn.relu(outputs)
    
    outputs = tf.layers.conv2d(outputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv6', use_bias=False)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv7', use_bias=False)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, 1, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv8', use_bias=False)
    outputs = tf.nn.relu(outputs)
    
    return outputs

def discriminator_model(inputs):
    outputs = tf.layers.conv2d(inputs, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 64, 3, padding='same', strides=(2,2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', strides=(2,2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', strides=(2,2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv6')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.contrib.layers.flatten(outputs)
    outputs = tf.layers.dense(outputs, units=1024, name='dense1')
    outputs = leaky_relu(outputs, alpha=0.2)
    outputs = tf.layers.dense(outputs, units=1, name='dense2')
    return outputs
        

def vgg_model(inputs):
    outputs = tf.concat([inputs*255-103.939, inputs*255-116.779, inputs*255-123.68], 3)
    outputs = tf.layers.conv2d(outputs, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv1_1')
    outputs = tf.layers.conv2d(outputs, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv1_2')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2,2), padding='same', name='pool1')
	
    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv2_1')
    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv2_2')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2,2), padding='same', name='pool2')
	
    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_1')
    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_2')
    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_3')
    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_4')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2,2), padding='same', name='pool3')

    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_1')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_2')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_3')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_4')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2,2), padding='same', name='pool4')
	
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_1')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_2')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_3')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_4')
    
    return outputs
