import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import data_handler


def process_numpy_data(file_path='data/', region='ME'):
	"""
	processes data from the numpy array for the given region into lists that are more well suited for
	subsequent function calls and the model itself

	args:
		- file_path: file_path of the numpy array data
		- region: which region will be processed (ie: 'ME')

	returns:
		- train_list: list containing training data
		- val_list: list containing validation data
		- test_list: list containing test data
	"""

	# load the numpy array data for a given region and get train, val, and test data splits
	region_data = np.load(file_path + region + '.npy')
	energy_train, energy_val, energy_test = data_handler.split_data(region_data)

	# turn the split-up data into lists where each entry contains data for 1 encoder/decoder cycle
	train_list = data_handler.create_lists_of_data(energy_train)
	val_list = data_handler.create_lists_of_data(energy_val)
	test_list = data_handler.create_lists_of_data(energy_test)

	return train_list, val_list, test_list



def get_single_batch(enc_dec_list, current_spot):
	"""
	creates the appropriate list of encoder/decoder inputs from the specified "current_spot" in the data list

	args:
		- enc_dec_list: a list whos entries each specify one encoder-decoder cycle
		- current_spot: what position should be broken further up into list forms

	returns:
		- enc_list: list of encoder inputs
		- dec_list: list of decoder inputs
		- new_current_spot: position to transform data to lists next cycle

	"""

	enc_list, dec_list, target_list = data_handler.separate_enc_dec(enc_dec_list[current_spot])
	new_current_spot = current_spot + 1

	return enc_list, dec_list, target_list, new_current_spot




def main():

	# get the data to be worked with
	training_list, validation_list, test_list = process_numpy_data()
	print "training_list size: ", len(training_list)

	# general parameters
	enc_len = 24 * 5 * 1  # 5 days
	dec_len = 24 * 1 * 1  # 1 day
	enc_insize = 3		   # day/hour/energy_demand
	dec_insize = 2		   # day/hour
	batch_size = 1
	pos_in_data = 0		   # what "list" of the total data are we going to train
	steps_per_epoch = len(training_list)  # how many encoder_decoder combined lists are in training data

	# hyperparameters
	learning_rate = 0.0001
	grad_clip_value = 10
	num_layers = 2
	num_hidden = 10
	total_num_epochs = 10  # number of epochs to train the data

	# placeholders & variables
	encoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, enc_insize]) for x in range(enc_len)]
	decoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, dec_insize]) for x in range(dec_len)]
	y_correct = [tf.placeholder(tf.float32, shape=[batch_size, 1]) for x in range(dec_len)]
	W_out = tf.Variable(tf.random_normal([num_hidden, 1], stddev=0.01))
	b_out = tf.Variable(tf.random_normal([1], stddev=0.01))

	# create cell object
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
	cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

	# outputs and states of model
	output_list, states_list = tf.nn.seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)

	losses = []
	# find predicted values & compute final cost for this batch
	for i in xrange(dec_len):
		y_hat = tf.matmul(output_list[i], W_out) + b_out
		losses.append(tf.squared_difference(y_hat, y_correct))

	losses = tf.reshape(tf.concat(1, losses), [-1, dec_len])
	cost = tf.reduce_mean(losses)

	# create training node to minimize MSE cost
	training_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# create initializer node to set up all trainable variables
	init_step = tf.initialize_all_variables()

	# create and run a session
	with tf.Session() as session:
		session.run(init_step)

		# list to hold costs, and variable to hold steps through data
		costs = []
		step = 0

		while step * batch_size < steps_per_epoch * total_num_epochs:

			# get a single batch
			# this means: getting a group of three lists for encoder inputs, decoder inputs, and targets
			#			  as well as remembering your place in the "entire list" of data
			enc_in, dec_in, targets, pos_in_data = get_single_batch(training_list, pos_in_data)

			# prevent going off the end of the list
			if pos_in_data >= steps_per_epoch:
				pos_in_data = 0

			x_e_list = {key: value for (key, value) in zip(encoder_inputs, enc_in)}
			x_d_list = {key: value for (key, value) in zip(decoder_inputs, dec_in)}
			y_list = {key: value for (key, value) in zip(y_correct, targets)}

			session.run(training_step, feed_dict=dict(x_e_list.items() + x_d_list.items() + y_list.items()))
			current_cost = session.run(cost, feed_dict=dict(x_e_list.items() + x_d_list.items() + y_list.items()))

			# add to running list of costs
			costs.append(current_cost)

			# increment step counter
			step += 1
			print "current cost: ",current_cost

		plt.plot(costs)
		plt.show()

if __name__ == '__main__':
	main()







