import tensorflow as tf
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


def trainAutoencoder(vectorop):
    size = np.shape(vectorop)[1]
    print(size)
    rs = int(size/10)
    ls = int(size/2)
    if size>1000:
    	ls = 750
    	rs = 128

    text = open("op3.txt", "r")
    data = [ ]
    for line in text:
        data.append( line.strip().split() )
    data=np.asarray(data)
    # Parameters
    learning_rate = 0.08
    training_epochs = 130
    batch_size = 1
    display_step = 1


    # Network Parameters
    n_input = size
    n_hidden_1 = ls # 1st layer num features
    n_hidden_2 = ls-50 # 2nd layer num features
    n_hidden_3 = ls-50
    n_hidden_4 = ls-100
    n_hidden_5 = min(rs,ls-100)
     

    # tf Graph input (only pictures)
    h1=tf.Variable(tf.random_normal([n_input, n_hidden_1]))
    h2=tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
    h3=tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]))
    h4=tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4]))
    h5=tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5]))

    X = tf.placeholder("float", [None, n_input])
    weights = {
        'encoder_h1': h1,
        'encoder_h2': h2,
        'encoder_h3': h3,
        'encoder_h4': h4,
        'encoder_h5': h5,
        'decoder_h1': tf.transpose(h5),
        'decoder_h2': tf.transpose(h4),
        'decoder_h3': tf.transpose(h3),
        'decoder_h4': tf.transpose(h2),
        'decoder_h5': tf.transpose(h1),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_4])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_3])),
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b4': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b5': tf.Variable(tf.random_normal([n_input])),
    }

    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with relu activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                       biases['encoder_b3']))
        layer_4= tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                       biases['encoder_b4']))
        layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['encoder_h5']),
                                       biases['encoder_b5']))
        return layer_5


    # Building the decoder
    def decoder(x):
        # Encoder Hidden layer with relu activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # Decoder Hidden layer with relu activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                       biases['decoder_b3']))
        layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                       biases['decoder_b4']))
        layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['decoder_h5']),
                                       biases['decoder_b5']))

        return layer_5
    # Construct model



    encoder_op = encoder(X)

    decoder_op = decoder(encoder_op)


    #Sentence code
    sent_code=encoder_op
    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X
    # Define loss and optimizer, minimize the sencode_decodeuared error
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            # Loop over all batches
            for i in range(1):
                
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={X: data})

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(c))

        
        encode_decode = sess.run(
        sent_code, feed_dict={X: data})
        print(">>>", encode_decode.shape)

        numpy.savetxt('sentcode.txt', encode_decode, fmt='%.9e', delimiter=' ', newline='\n')
        print("Optimization Finished!")


       
        
      # Applying encode and decode over test set
