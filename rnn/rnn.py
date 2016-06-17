# coding: utf-8

###
#	University of Edinburgh, School of Informatics
#	Natural Language Understanding, Assignment 2 (Semester 2, 2016)
#	http://www.inf.ed.ac.uk/teaching/courses/nlu/
#
#	Recurrent Neural Network for Language Modelling
# 	Author: Paul W. Coles
#	Note: functions or Classes provided in skeleton code for this assignment
#		  are indicated in respective comments
###

from math import *
from rnnmath import *
from numpy import *
from sys import stdout
import time

class RNN(object):
	'''
	This class implements Recurrent Neural Networks.

	Code implemented by author for following functions:
		predict				->	predict an output sequence for a given input sequence
		acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
		acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
		compute_loss 		->	compute the (cross entropy) loss between the desired output and predicted output for a given input sequence
		compute_mean_loss	->	compute the average loss over all sequences in a corpus
		generate_sequence	->	use the RNN to generate a new (unseen) sequnce
	'''

	def __init__(self, vocab_size, hidden_dims):
		'''
		initialize the RNN with random weight matrices.
		** This function provided in skeleton code **

		vocab_size		size of vocabulary that is being used
		hidden_dims		number of hidden units
		'''
		self.vocab_size = vocab_size
		self.hidden_dims = hidden_dims

		# matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
		self.V = random.randn(self.hidden_dims, self.vocab_size)*sqrt(0.1)
		self.W = random.randn(self.vocab_size, self.hidden_dims)*sqrt(0.1)
		self.U = random.randn(self.hidden_dims, self.hidden_dims)*sqrt(0.1)

		# aggregated weight changes for V, W, U
		self.deltaV = zeros((self.hidden_dims, self.vocab_size))
		self.deltaW = zeros((self.vocab_size, self.hidden_dims))
		self.deltaU = zeros((self.hidden_dims, self.hidden_dims))


	def predict(self, x):
		'''
		predict an output sequence y for a given input sequence x
		x	list of words, as indices, e.g.: [0, 4, 2]

		returns	y,s
		y	matrix of probability vectors for each input word
		s	matrix of hidden layers for each input word
		'''

		# matrix s for hidden states, y for output states, given input x.
		#rows correspond to times t, i.e., input words
		# s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )
		s = zeros((len(x)+1, self.hidden_dims))
		y = zeros((len(x), self.vocab_size))

		for t in range(len(x)):
			one_hot = make_onehot(x[t], self.vocab_size)
			s[t] = sigmoid(dot(self.V, one_hot) + dot(self.U,s[t-1]))
			net_out = dot(self.W, s[t])
			y[t] = softmax(net_out)
		return y, s


	def acc_deltas(self, x, d, y, s):
		'''
		accumulate updates for V, W, U
		standard back propagation

		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

		x	list of words, as indices, e.g.: [0, 4, 2]
		d	list of words, as indices, e.g.: [4, 2, 3]
		y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
			should be part of the ret			in_delta = dot(self.W.transpose(), out_delta)*f_derivative
urn value of predict(x)
		s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
			should be part of the return value of predict(x)

		no return values
		'''
		for t in reversed(range(len(x))):
			# Update deltaW
			d_hot = make_onehot(d[t], self.vocab_size)
			out_delta = d_hot - y[t]
			self.deltaW += outer(out_delta, s[t])

			# Update deltaV
			x_hot = make_onehot(x[t], self.vocab_size)
			f_derivative = grad(s[t])
			in_delta = dot(self.W.transpose(), out_delta)*f_derivative
			self.deltaV += outer(in_delta, x_hot)

			# Update deltaU
			self.deltaU += outer(in_delta, s[t-1])

	def acc_deltas_bptt(self, x, d, y, s, steps):
		'''
		accumulate updates for V, W, U
		back propagation through time (BPTT)

		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

		x		list of words, as indices, e.g.: [0, 4, 2]
		d		list of words, as indices, e.g.: [4, 2, 3]
		y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
				should be part of the return value of predict(x)
		s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
				should be part of the return value of predict(x)
		steps	number of time steps to go back in BPTT

		no return values
		'''
		# # Initialise deltas
		# self.acc_deltas(x, d, y, s)

		for t in reversed(range(len(x))):
			# Update deltaW
			d_hot = make_onehot(d[t], self.vocab_size)
			out_delta = d_hot - y[t]
			self.deltaW += outer(out_delta, s[t])

			# Update deltaV
			f_derivative = grad(s[t])
			in_delta = dot(self.W.transpose(), out_delta)*f_derivative

			for tau in range(steps + 1):
				if t - tau >= 0:
					x_hot = make_onehot(x[t - tau], self.vocab_size)
					bp_derivative = grad(s[t - tau - 1])
					self.deltaV += outer(in_delta, x_hot)
					self.deltaU += outer(in_delta, s[t - tau - 1])
					in_delta = dot(self.U.transpose(), in_delta) * bp_derivative

	def compute_loss(self, x, d):
		'''
		compute the loss between predictions y for x, and desired output d.

		first predicts the output for x using the RNN, then computes the loss w.r.t. d

		x		list of words, as indices, e.g.: [0, 4, 2]
		d		list of words, as indices, e.g.: [4, 2, 3]

		return loss		the combined loss for all words
		'''
		y,s = self.predict(x)
		loss = 0.
		for t in range(len(y)):
			d_hot = make_onehot(d[t],self.vocab_size)
			loss += log(y[t][d[t]])
		loss = -loss
		return loss

	def compute_mean_loss(self, X, D):
		'''
		compute the mean loss over a dataset X, D

		for each instance, the mean loss is the total loss for x, d, divided by the length of the instance

		X	a list of input vectors, e.g., 		[[0, 4, 2], [1, 3, 0]]
		D	a list of desired outputs, e.g., 	[[4, 2, 3], [3, 0, 3]]

		return mean_loss	mean loss over all instances
		'''
		# Global call
		if isinstance(X[0], number) is False:
			global_loss = 0.
			instances_count = len(X)
			elements_count = 0
			for index in range(instances_count):
				local_loss, instance_len = self.compute_mean_loss(X[index], D[index])
				elements_count += instance_len
				global_loss += local_loss
			return global_loss / elements_count

		# Local call
		else:
			local_loss = 0.
			instance_len = len(X)
			for index in range(instance_len):
				local_loss += self.compute_loss(X, D)
				return local_loss, instance_len


	def generate_sequence(self, start, end, maxLength):
		'''
		generate a new sequence, using the RNN

		starting from the word-index for a start symbol, generate some output until the word-index of an end symbol is generated, or the sequence
		exceed maxLength

		HINT: make use of the "multinomial_sample" method in utils.py !!!

		start		word index of start symbol (the symbol <s> in a vocabulary)
		end			word index of end symbol (the symbol </s> in a vocabulary)
		maxLength	maximum length of the generated sequence

		return sequence, loss

		sequence	the generated sequence as a list of word indices, e.g., [4, 2, 3, 5]
		loss		the loss of the generated sequence
		'''
		sequence = [start]
		loss = 0.
		x = [start]

		# maxLength shouldn't include start, end symbols
		maxLength += 2

		while len(sequence) < maxLength:
			y, s = self.predict(x)
			# Sample for a word, if it's </s> append it and break, else just append it
			next_word = multinomial_sample(y)
			# Control for error where sampled word is out of vocab indexes
			# if next_word < self.vocab_size:
			loss += log(y[0][next_word])
			if next_word != end:
				sequence.append(next_word)
			else:
				sequence.append(end)
				break
		x = [sequence]
		return sequence, loss

	def train(self, X, D, X_dev, D_dev, epochs, learning_rate, anneal, back_steps, batch_size, min_change):
		'''
		train the RNN on some training set X, D while optimizing the loss on a dev set X_dev, D_dev

		** This function provided in skeleton code **

		training stops after the first of the following is true:
			* number of epochs reached
			* minimum change observed for more than 2 consecutive epochs

		X				a list of input vectors, e.g., 		[[0, 4, 2], [1, 3, 0]]
		D				a list of desired outputs, e.g., 	[[4, 2, 3], [3, 0, 3]]
		X_dev			a list of input vectors, e.g., 		[[0, 4, 2], [1, 3, 0]]
		D_dev			a list of desired outputs, e.g., 	[[4, 2, 3], [3, 0, 3]]
		epochs			maximum number of epochs (iterations) over the training set
		learning_rate	initial learning rate for training
		anneal			positive integer. if > 0, lowers the learning rate in a harmonically after each epoch.
						higher annealing rate means less change per epoch.
						anneal=0 will not change the learning rate over time
		back_steps		positive integer. number of timesteps for BPTT. if back_steps < 2, standard BP will be used
		batch_size		number of training instances to use before updating the RNN's weight matrices.
						if set to 1, weights will be updated after each instance. if set to len(X), weights are only updated after each epoch
		min_change		minimum change in loss between 2 epochs. if the change in loss is smaller than min_change, training stops regardless of
						number of epochs left
		'''
		print("Training model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(epochs, len(X), batch_size))
		print("Optimizing loss on {0} sentences".format(len(X_dev)))
		print("Vocab size: {0}\nHidden units: {1}".format(self.vocab_size, self.hidden_dims))
		print("Steps for back propagation: {0}".format(back_steps))
		print("Initial learning rate set to {0}, annealing set to {1}".format(learning_rate, anneal))
		print("\ncalculating initial mean loss on dev set")

		initial_loss = self.compute_mean_loss(X_dev,D_dev)
		print("initial mean loss: {0}".format(initial_loss))

		prev_loss = initial_loss
		loss_watch_count = -1
		min_change_count = -1

		a0 = learning_rate

		best_loss = initial_loss
		bestU, bestV, bestW = self.U, self.V, self.W
		best_epoch = 0

		for epoch in range(epochs):
			t0 = time.time()

			if anneal > 0:
				learning_rate = a0/((epoch+0.0+anneal)/anneal)
			else:
				learning_rate = a0
			print("\nepoch %d, learning rate %.04f" % (epoch+1, learning_rate))

			count = 0

			# use random sequence of instances in the training set (tries to avoid local maxima when training on batches)
			for i in random.permutation(range(len(X))):
				count += 1
				stdout.write("\r\tpair {0}".format(count))

				x_i = X[i]
				d_i = D[i]
				y_i, s_i = self.predict(x_i)
				if back_steps < 2:
					self.acc_deltas(x_i, d_i, y_i, s_i)
				else:
					self.acc_deltas_bptt(x_i, d_i, y_i, s_i, back_steps)

				if count % batch_size == 0:
					self.deltaU /= batch_size
					self.deltaV /= batch_size
					self.deltaW /= batch_size
					self.apply_deltas(learning_rate)

			if count % batch_size > 0:
				self.deltaU /= (count % batch_size)
				self.deltaV /= (count % batch_size)
				self.deltaW /= (count % batch_size)
				self.apply_deltas(learning_rate)

			print("\n\tcalculating new loss on dev set")
			loss = self.compute_mean_loss(X_dev,D_dev)
			print("\tmean loss: {0}".format(loss))
			print("\tepoch done in %.02f seconds" % (time.time() - t0))

			if loss < best_loss:
				best_loss = loss
				bestU, bestV, bestW = self.U.copy(), self.V.copy(), self.W.copy()
				best_epoch = epoch

			# make sure we change the RNN enough
			if abs(prev_loss - loss) < min_change:
				min_change_count += 1
			else:
				min_change_count = 0
			if min_change_count > 2:
				print("\ntraining finished after {0} epochs due to minimal change in loss".format(epoch+1))
				break

			prev_loss = loss
		print("\ntraining finished after reaching maximum of {0} epochs".format(epochs))
		print("best observed loss was {0}, at epoch {1}".format(best_loss, (best_epoch+1)))
		print("setting U, V, W to matrices from best epoch")

		self.U, self.V, self.W = bestU, bestV, bestW

	def apply_deltas(self, learning_rate):
		'''
		update the RNN's weight matrices with corrections accumulated over some training instances

		** This function provided in skeleton code **
		learning_rate	scaling factor for update weights
		'''

		# apply updates to U, V, W
		self.W += learning_rate*self.deltaW
		self.V += learning_rate*self.deltaV
		self.U += learning_rate*self.deltaU

		# reset delta matrices
		self.deltaW.fill(0.)
		self.deltaV.fill(0.)
		self.deltaU.fill(0.)

if __name__ == "__main__":
	import sys
	from utils import *
	from rnnmath import *
	mode = sys.argv[1].lower()

	if mode == "train":
		import time
		data_folder = sys.argv[2]
		vocabsize = 2000
		# Make a logfile
		experiment_started = time.strftime("%d_%m_%Y_%H%M%S")
		hdim = 100
		print('Experiment started at ' + experiment_started + '\n\n')
		# get the data set vocabulary
		vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		num_to_word = dict(enumerate(vocab.index[:vocabsize]))
		word_to_num = invert_dict(num_to_word)
		# calculate loss vocabulary words due to vocabsize
		fraction_lost = fraq_loss(vocab, word_to_num, vocabsize)
		print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocabsize, len(vocab), 100*(1-fraction_lost)))
		docs = load_dataset(data_folder + '/ptb-train.txt')
		S_train = docs_to_indices(docs, word_to_num)
		X_train, D_train = seqs_to_lmXY(S_train)
		# Load the dev set (for tuning hyperparameters)
		docs = load_dataset(data_folder + '/ptb-dev.txt')
		S_dev = docs_to_indices(docs, word_to_num)
		X_dev, D_dev = seqs_to_lmXY(S_dev)

		X = X_train
		D = D_train
		X_dev_training = X_dev[:1000]
		D_dev_training = D_dev[:1000]

		r = RNN(vocabsize, hdim)
		# train the RNN
		#r.train(X, D, X_dev, D_dev, epochs, learning_rate, anneal, back_steps, batch_size, min_change):
		print "\nTraining RNN..."
		r.train(X, D, X_dev_training, D_dev_training, 25, 0.5, 5, 3, 100, 0.0001)
		print "\n Computing loss against truncated dev set..."
		dev_loss = r.compute_mean_loss(X_dev_training, D_dev_training)
		print("mean loss (truncated data): {0}".format(dev_loss))
		# q = best unigram frequency from omitted vocab
		# this is the best expected loss out of that set
		q = vocab.freq[vocabsize] / sum(vocab.freq[vocabsize:])
		print "Unadjusted: %.03f" % exp(dev_loss)
		print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost, q))
		# save RNN matricse
		print "\nSaving RNN matricies..."
		save("rnn.U", r.U)
		save("rnn.V", r.V)
		save("rnn.W", r.W)

		#Evaluate against whole dev set
		print "\n\nEvaluating against whole dev set..."
		dev_loss = r.compute_mean_loss(X_dev, D_dev)
		print "\nmean loss (all data): {0}".format(dev_loss)
		# q = best unigram frequency from omitted vocab
		# this is the best expected loss out of that set
		q = vocab.freq[vocabsize] / sum(vocab.freq[vocabsize:])
		print "Unadjusted (all data): %.03f" % exp(dev_loss)
		print "Adjusted for missing vocab (all data): %.03f" % exp(adjust_loss(dev_loss, fraction_lost, q))


	if mode == "parameter_estimate":
		'''
		starter code for parameter estimation.
		change this to different values, or use it to get you started with your own testing class
		'''
		import time
		import os
		data_folder = sys.argv[2]
		vocabsize = 2000

		# Make a logfile
		experiment_started = time.strftime("%d_%m_%Y_%H%M%S")
		with open(os.path.join('./logs', 'log_' + experiment_started), 'w') as logfile:
			logfile.write('Experiment started at ' + experiment_started + '\n\n')

			# Grid search for parameters
			iteration = 1
			# get the data set vocabulary
			vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
			num_to_word = dict(enumerate(vocab.index[:vocabsize]))
			word_to_num = invert_dict(num_to_word)

			# calculate loss vocabulary words due to vocabsize
			fraction_lost = fraq_loss(vocab, word_to_num, vocabsize)
			print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocabsize, len(vocab), 100*(1-fraction_lost)))
			logfile.write("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocabsize, len(vocab), 100*(1-fraction_lost)))

			docs = load_dataset(data_folder + '/ptb-train.txt')
			S_train = docs_to_indices(docs, word_to_num)
			X_train, D_train = seqs_to_lmXY(S_train)

			# Load the dev set (for tuning hyperparameters)
			docs = load_dataset(data_folder + '/ptb-dev.txt')
			S_dev = docs_to_indices(docs, word_to_num)
			X_dev, D_dev = seqs_to_lmXY(S_dev)

			X = X_train[:1000]
			D = D_train[:1000]
			X_dev = X_dev[:1000]
			D_dev = D_dev[:1000]
			for h_units in (10,50,100):
				hdim = h_units
				for l_rate in (0.5, 0.1, 0.05):
					for look_back in (0, 3, 10):
						print 'Iteration:\t' + str(iteration)
						print 'h_units:  \t' + str(h_units)
						print 'l_rate:   \t' + str(l_rate)
						print 'look_back:\t' + str(look_back)
						logfile.write('Iteration:\t' + str(iteration))
						logfile.write('\nh_units:  \t' + str(h_units))
						logfile.write('\nl_rate:   \t' + str(l_rate))
						logfile.write('\nlook_back:\t' + str(look_back))
						logfile.write('\n')
						r = RNN(vocabsize, hdim)
						# train the RNN
						#r.train(X, D, X_dev, D_dev, epochs, learning_rate, anneal, back_steps, batch_size, min_change):
						r.train(X, D, X_dev, D_dev, 25, l_rate, 5, look_back, 100, 0.0001)

						dev_loss = r.compute_mean_loss(X_dev, D_dev)
						print("mean loss: {0}".format(dev_loss))
						logfile.write("mean loss: {0}".format(dev_loss))

						# q = best unigram frequency from omitted vocab
						# this is the best expected loss out of that set
						q = vocab.freq[vocabsize] / sum(vocab.freq[vocabsize:])
						print "Unadjusted: %.03f" % exp(dev_loss)
						print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost, q))
						logfile.write("\nUnadjusted: %.03f" % exp(dev_loss))
						logfile.write("\nAdjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost, q)))
						logfile.write('\n === Iteration Ends === \n\n')
						iteration += 1

						# save RNN matricse
						save("rnn.U", r.U)
						save("rnn.V", r.V)
						save("rnn.W", r.W)

	if mode == "generate":
		'''
		starter code for sequence generation
		.
		change this to different values, or use it to get you started with your own testing class
		'''
		data_folder = sys.argv[2]
		rnn_folder = sys.argv[3]
		maxLength = int(sys.argv[4])

		# get saved RNN matrices and setup RNN
		U,V,W = load(rnn_folder + "/rnn.U.npy"), load(rnn_folder + "/rnn.V.npy"), load(rnn_folder + "/rnn.W.npy")
		vocabsize = len(V[0])
		hdim = len(U[0])

		r = RNN(vocabsize, hdim)
		r.U = U
		r.V = V
		r.W = W

		# get vocabulary
		vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		num_to_word = dict(enumerate(vocab.index[:vocabsize]))
		word_to_num = invert_dict(num_to_word)

		# predict something
		generated, loss = r.generate_sequence(word_to_num["<s>"], word_to_num["</s>"], maxLength)
		print("Generated the sequence\n{0}".format(generated))
		print("loss: {0}".format(loss))
		sent = [num_to_word[word] for word in generated]
		while "UUUNKKK" in sent:
			sent.remove("UUUNKKK")
		print(sent)
		perplexity = exp(loss)
		print "Perplexity of predicted sequence:\t%.5e" %perplexity
