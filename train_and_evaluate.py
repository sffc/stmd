#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import prettytensor

import utils

# Global constants
BATCH_SIZE = 12
NUM_CLASSES = 2
NUM_EPOCHS = 500
LEARNING_RATE = 0.001
NUM_DATA_PTS = 144


# Load the data and split it into training and test.
dataX, dataY = utils.get_data_batches(NUM_DATA_PTS, BATCH_SIZE)


# Set up the neural network using PrettyTensor.
# Use a simple CNN with one hidden layer of 50 neurons.
input_t = tf.placeholder(tf.float32, (BATCH_SIZE, dataX[0].shape[1], dataX[0].shape[2], 1), name="input_t")
labels_t = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_CLASSES), name="labels_t")
input_p = prettytensor.wrap(input_t)
hidden_p = (input_p
	.conv2d(3, 4, edges='VALID')
	.max_pool(2, 2)
	.flatten()
	.fully_connected(50))
softmax_p, loss_p = hidden_p.softmax_classifier(NUM_CLASSES, labels_t)
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
optim_p = prettytensor.apply_optimizer(optimizer, losses=[loss_p])


# Train and evaluate the neural network.
with tf.Session() as sess:
	tf.initialize_all_variables().run()
	loss_over_time = []
	vloss_over_time = []
	for epoch in range(NUM_EPOCHS):
		# Save 1 batch for validation.
		for i in range(len(dataX)-1):
			loss, _ = sess.run([loss_p, optim_p], {
				input_t: dataX[i],
				labels_t: dataY[i]
			})
			loss_over_time.append(loss)
		validX = dataX[-1]
		validY = dataY[-1]
		vloss, vpred = sess.run([loss_p, softmax_p], {
			input_t: validX,
			labels_t: validY
		})
		vloss_over_time.append(vloss)
		vcorr = sum(np.argmax(vpred, 1) == np.argmax(validY, 1))
		print("Epoch %03d: %6.4f validation loss; %02d/%02d correct" % (epoch, vloss, vcorr, BATCH_SIZE))

	skip_length = int(len(loss_over_time) / 100)
	utils.terminal_plot("Training Loss", list(range(100)), loss_over_time[::skip_length])
	utils.terminal_plot("Validation Loss", list(range(NUM_EPOCHS)), vloss_over_time)


