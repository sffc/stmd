import numpy as np
import os
import random
import subprocess

import PIL.Image


def reservoir_sample(generator, k):
	"""Selects k elements at random from the generator."""
	try:
		result = [next(generator) for i in range(k)]
	except StopIteration as err:
		raise ValueError("Fewer than k=%d elements are in reservoir %s" % (k, generator))
	for i, item in enumerate(generator):
		j = random.randrange(i + k)
		if j < k:
			result[j] = item
	random.shuffle(result)
	return result


def read_stmd(organ):
	basedir = os.path.join(os.path.dirname(__file__), "data", organ)
	for filename in os.listdir(basedir):
		img_path = os.path.join(basedir, filename)
		with PIL.Image.open(img_path) as img:
			arr = np.asarray(img, dtype=np.uint8)
			assert arr.shape[2] == 2
			yield arr[:,:,0]  # gets rid of the alpha channel


def terminal_plot(title, x, y):
	"""Based on http://stackoverflow.com/a/20411508/1407170"""
	gnuplot = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE)
	gnuplot.stdin.write(bytes("set term dumb 140 25\n", "utf-8"))
	gnuplot.stdin.write(bytes("plot '-' using 1:2 title '%s' with linespoints \n" % title, "utf-8"))
	for i,j in zip(x,y):
	   gnuplot.stdin.write(bytes("%f %f\n" % (i,j), "utf-8"))
	gnuplot.stdin.write(bytes("e\n", "utf-8"))
	gnuplot.stdin.flush()


def split_and_batch_data(all_X, all_Y, batch_size, num_classes):
	height, width = all_X[0].shape
	num_batches = len(all_X) / batch_size
	assert num_batches % 1 == 0
	num_batches = int(num_batches)

	# Randomize the order.
	combo_list = list(zip(all_X, all_Y))
	random.shuffle(combo_list)
	all_data_rand, all_labels_rand = zip(*combo_list)

	# Batch the data into the specified batch size.
	dataX_batched = []
	dataY_batched = []
	for b in range(num_batches):
		batchX = np.ndarray((batch_size, height, width, 1), dtype=np.float32)
		batchY = np.zeros((batch_size, num_classes), dtype=np.float32)
		for i in range(batch_size):
			img = all_data_rand[i+b*batch_size]
			label = all_labels_rand[i+b*batch_size]
			# Ensure that the images are normalized and presented as floats between 0 and 1.
			batchX[i,:,:,0] = img / float(np.max(img))
			batchY[i,label] = 1
		dataX_batched.append(batchX)
		dataY_batched.append(batchY)

	return dataX_batched, dataY_batched
