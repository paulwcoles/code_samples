# coding: utf-8

###
#	University of Edinburgh, School of Informatics
#	Natural Language Understanding, Assignment 1 (Semester 2, 2016)
#	http://www.inf.ed.ac.uk/teaching/courses/nlu/
#
#	Distributional Models of Semantics
# 	Author: Paul W. Coles
#	Note: functions or Classes provided in skeleton code for this assignment
#		  are indicated in respective comments
###

import gensim
import math
from copy import copy
import re
from operator import itemgetter
from collections import Counter
import numpy


'''
Determine whether a vector is densely encoded
input: any vector
output: Boolean (True if input is densely encoded, else False)
'''
def is_it_dense(vector):
	dense = True
	if len(vector) > 0 and isinstance(vector[0], tuple):
		dense = False
	return dense


'''
Convert sparse vector to dense encoding
input: any vector
output: input vector densely encoded
'''
def convert_to_dense(sparse_vector):
	if is_it_dense(sparse_vector):
		return sparse_vector
	else:
		sparse_values = dict(sparse_vector)
		dense_vector = []
		dense_vector_len = max(sparse_values.keys(), key=int) + 1
		for index in xrange(0, dense_vector_len):
			if index in sparse_values:
				dense_vector.append(sparse_values[index])
			else:
				dense_vector.append(0)
		return dense_vector

'''
Convert dense vector to sparse encoding
input: any vector
output: input vector sparsely encoded
'''
def convert_to_sparse(dense_vector):
	if not is_it_dense(dense_vector):
		return dense_vector
	else:
		sparse_vector = []
		index = 0
		for element in dense_vector:
			if element != 0:
				sparse_vector.append((index, element))
			index += 1
		return sparse_vector


'''
Compute magnitude of a vector
input: any vector
output: magnitude of input vector
'''
def compute_magnitude(vector):
	vector = convert_to_dense(vector)
	summation = 0.0
	for element in vector:
		summation += (float(element) * float(element))
	return math.sqrt(summation)

'''
Read corpus from disk
input: vocabFile containing vocabulary
input: contextFile containing word contexts
output: id2word mapping word IDs to words
output: word2id mapping words to word IDs
output: vectors for the corpus, as a list of sparse vectors
'''
def load_corpus(vocabFile, contextFile):
	id2word = {}
	word2id = {}
	vectors = []
	# make word2id, id2word mappings; id of vector is line number in vocabFile
	id = 0
	with open(vocabFile, 'r') as vocab_opened:
		for line in vocab_opened:
			word = line.strip('\n')
			word2id[word] = id
			id2word[id] = word
			id += 1
	# make one sparsely-encoded vector per line in contextFile, append it to
	# 'vectors' list; each context pair as a tuple of the index and its count
	with open(contextFile, 'r') as context_opened:
		for line in context_opened:
			context_pairs = line.split()
			vector = []
			for context_pair in context_pairs[1:]:
				index, frequency = re.split(r':', context_pair)
				vector.append((int(index), int(frequency)))
			vectors.append(vector)
	return id2word, word2id, vectors


'''
Calculate cosine similarity of two vectors
input: vector1
input: vector2
output: cosine similarity between vector1 and vector2 as a real number
'''
def cosine_similarity(vector1, vector2):
	# convert sparse to dense if necessary:
	vector1, vector2 = convert_to_dense(vector1), convert_to_dense(vector2)
	# now assuming dense encoding, compute dot product:
	dot_product = sum(float(vec1_element) * float(vec2_element) \
	for vec1_element, vec2_element in zip(vector1, vector2))
	# compute magnitudes
	mag_vec1, mag_vec2 = compute_magnitude(vector1), compute_magnitude(vector2)
	# return cosine similarity
	try:
		cosine_sim = dot_product / (mag_vec1 * mag_vec2)
	except:
		cosine_sim = 0
	return cosine_sim


'''
Convert frequency-based vector space into tf-idf weighted vector space
input: freqVectors, a list of frequency-based vectors
output: tfIdfVectors, a list of tf-idf-based vectors
'''
def tf_idf(freqVectors):
	# Ensure every doc vector in freqVectors is sparsely-encoded
	sparse_freq_vectors = []
	for vector in freqVectors:
		if len(vector) > 0:
			if is_it_dense(vector):
				sparse_freq_vectors.append(convert_to_sparse(vector))
			else:
				sparse_freq_vectors.append(vector)
		else:
			sparse_freq_vectors.append([])
	# Now assuming sparse encoding...
	# N = count of docs in corpus
	N = len(sparse_freq_vectors)
	# Get count of docs that contain each term
	term_in_docs = {}
	for doc in sparse_freq_vectors:
		for term in doc:
			term_id = term[0]
			if term_id in term_in_docs:
				term_in_docs[term_id] += 1
			else:
				term_in_docs[term_id] = 1
	# For each doc, make new vector in tf-idf space
	tfIdfVectors = []
	for doc in sparse_freq_vectors:
		tfIdfVector = []
		for term in doc:
			index = term[0]
			freq_value = term[1]
			converted_freq = (1 + math.log(freq_value, 2)) * (math.log \
			((N / 1 + term_in_docs[index]), 2))
			tfIdfVector.append((index, converted_freq))
		tfIdfVectors.append(tfIdfVector)
	# Return vector space with same order but in tf-idf form
	return tfIdfVectors

'''
Build word2vec vector model of the corpus with 100 dimensions and window size 5
'''
def word2vec(corpus, learningRate, downsampleRate, negSampling):
	model = gensim.models.Word2Vec(sentences, size=100, alpha=learningRate, min_count=1, window=5, \
								   sample=downsampleRate, negative=negSampling)
	result_dict = model.accuracy(data_dir + 'accuracy_test.txt')[-1]
	correct_count = float(len(result_dict['correct']))
	incorrect_count = float(len(result_dict['incorrect']))
	percent_correct = 100 * (correct_count / (correct_count + incorrect_count))
	print str('{0:5f}'.format(percent_correct)) + '%% percent of test data correct for parameter ' \
							'combination: %s learning rate, %s downsample rate, %s negative sampling.' \
							% (str(learning_rate), str(downsample_rate), str(neg_sampling))
	return result_dict, percent_correct


'''
Build an LDA model with 100 topics from a frequency vector space
input: vectors
input: wordMapping (optional) mapping from word IDs to words
output: an LDA topic model with 100 topics, using the frequency vectors
'''
def lda(vectors, wordMapping):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	lda_model = gensim.models.ldamodel.LdaModel(corpus=vectors, id2word=wordMapping, num_topics=100, update_every=0, passes=10)
	print 'Model complete. Saving to file.'
	lda_model.save(fname= results_dir + 'lda_model')


'''
Get words in a given LDA topic
input: ldaModel, pre-trained Gensim LDA model
input: topicID, ID of the topic for which to get topic words
input: wordMapping, mapping from words to IDs (optional)
'''
def get_topic_words(ldaModel, topicID, wordMapping):
	words_in_topic = []
	for word in ldaModel.print_topic(topicID, topn=20).split('+'):
		words_in_topic.append(word)
	return words_in_topic


'''
**This Class provided in skeleton code**
provides an iterator over sentences in the provided BNC corpus
input: corpus path to the BNC corpus
input: n, number of sentences to retrieve (optional, standard -1: all)
'''
class BncSentences:
	def __init__(self, corpus, n=-1):
		self.corpus = corpus
		self.n = n

	def __iter__(self):
		n = self.n
		ret = []
		for line in open(self.corpus):
			line = line.strip().lower()
			if line.startswith("<s "):
				ret = []
			elif line.strip() == "</s>":
				if n > 0:
					n -= 1
				if n == 0:
					break
				yield copy(ret)
			else:
				parts = line.split("\t")
				if len(parts) == 3:
					word = parts[-1]
					idx = word.rfind("-")
					word, pos = word[:idx], word[idx + 1:]
					if word in ["thus", "late", "often", "only"]:
						pos = "r"
					if pos == "j":
						pos = "a"
					ret.append(gensim.utils.any2unicode(word + "." + pos))



if __name__ == '__main__':
	import sys
	import os
	# specify central data store, prepare local environment for results recording
	results_dir = './results'
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	data_dir = '/afs/inf.ed.ac.uk/group/project/nlu/data/'
	part = sys.argv[1].lower()

	# these are indices for house, home and time in the data. Don't change.
	house_noun = 80
	home_noun = 143
	time_noun = 12
	if part == 'test':
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		convert_to_dense(vectors[0])

	if part == "a":
		print("(a): load corpus")
		try:
			id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
			if not id2word:
				print("\tError: id2word is None or empty")
				exit()
			if not word2id:
				print("\tError: id2word is None or empty")
				exit()
			if not vectors:
				print("\tError: id2word is None or empty")
				exit()
			print("\tPass: load corpus from file")
		except Exception:
			print("\tError: could not load corpus from disk")

		try:
			if not id2word[house_noun] == "house.n" or not id2word[home_noun] == "home.n" or not id2word[
				time_noun] == "time.n":
				print("\tError: id2word fails to retrive correct words for ids")
			else:
				print("\tPass: id2word")
		except Exception:
			print("\tError: Exception in id2word")

		try:
			if not word2id["house.n"] == house_noun or not word2id["home.n"] == home_noun or not word2id[
				"time.n"] == time_noun:
				print("\tError: word2id fails to retrive correct ids for words")
			else:
				print("\tPass: word2id")
		except Exception:
			print("\tError: Exception in word2id")

	if part == "b":
		import numpy
		print("(b): cosine similarity")
		try:
			cos = cosine_similarity([(0, 1), (2, 1), (4, 2)], [(0, 1), (1, 2), (4, 1)])
			if not numpy.isclose(0.5, cos):
				print("\tError: sparse expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: sparse vector similarity")
		except Exception:
			print("\tError: failed for sparse vector")
		try:
			cos = cosine_similarity([1, 0, 1, 0, 2], [1, 2, 0, 0, 1])
			if not numpy.isclose(0.5, cos):
				print("\tError: full expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: full vector similarity")
		except Exception:
			print("\tError: failed for full vector")

	if part == "c":
		print("(c) similarity of house, home and time in frequency space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		house_id, home_id, time_id = word2id['house.n'], word2id['home.n'], word2id['time.n']
		print "house:home similarity: \t" + str(cosine_similarity(vectors[house_id], vectors[home_id]))
		print "house:time similarity: \t" + str(cosine_similarity(vectors[house_id], vectors[time_id]))
		print "home:time similarity: \t" + str(cosine_similarity(vectors[home_id], vectors[time_id]))

	if part == "d":
		print("(d) converting to tf-idf space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		try:
			tfIdfSpace = tf_idf(vectors)
			if not len(vectors) == len(tfIdfSpace):
				print("\tError: tf-idf space does not correspond to original vector space")
			else:
				print("\tPass: converted to tf-idf space")
		except Exception as e:
			print("\tError: could not convert to tf-idf space")
			print(e)

	if part == "e":
		print("(e) similarity of house, home and time in tf-idf space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		try:
			tfIdfSpace = tf_idf(vectors)
			house_id, home_id, time_id = word2id['house.n'], word2id['home.n'], word2id['time.n']
			house_home_sim = cosine_similarity(tfIdfSpace[house_id], tfIdfSpace[home_id])
			house_time_sim = cosine_similarity(tfIdfSpace[house_id], tfIdfSpace[time_id])
			home_time_sim  = cosine_similarity(tfIdfSpace[home_id], tfIdfSpace[time_id])
			print "house:home similarity: \t" + str(house_home_sim)
			print "house:time similarity: \t" + str(house_time_sim)
			print "home:time  similarity: \t" + str(home_time_sim)
		except Exception as e:
			print("\tError: could not convert to tf-idf space")
			print(e)

	if part == "f1":
		import json
		print("(f1) word2vec, estimating best learning rate, sample rate, negative sampling")
		# Instantiate corpus class
		corpus = BncSentences("/afs/inf.ed.ac.uk/group/project/nlu/data/bnc.vert", 100000)
		sentences = list(corpus.__iter__())
		# Specify values for test variables, initialise dict for test results
		test_learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05]
		test_downsample_rates = [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
		test_neg_sampling = [0, 2, 4, 6, 8, 10]
		full_results = {}
		proportion_correct = {}
		for learning_rate in test_learning_rates:
			for downsample_rate in test_downsample_rates:
				for neg_sampling in test_neg_sampling:
					result_dict, percent_correct = word2vec(sentences, learning_rate, downsample_rate, neg_sampling)
					full_results[str((learning_rate, downsample_rate, neg_sampling))] = result_dict
					proportion_correct[str((learning_rate, downsample_rate, neg_sampling))] = percent_correct
		# Write results to file for reference
		with open(results_dir + 'parameter_estimation_full.json', 'w') as f:
			json.dump(full_results, f)
		with open(results_dir + 'parameter_estimation_proportions.json', 'w') as f:
			json.dump(proportion_correct, f)
		# Sort summary results, print top 10 scoring parameter combinations to console
		proportions_sorted = sorted(proportion_correct.iteritems(), key=lambda (k, v): (-v, k))[:10]
		print '\nTop Ten Parameter Combinations for Overall Performance: \nCombination \t Score'
		for top_scorer in proportions_sorted:
			print str(top_scorer[0]) + '\t' + str(top_scorer[1]) + '%%'

	if part == "f2":
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		print("(f2) word2vec, building full model with best parameters. May take a while.")
		corpus = BncSentences(data_dir + "bnc.vert")
		sentences = list(corpus.__iter__())
		best_model = gensim.models.Word2Vec(sentences, size=100, alpha= 0.04, min_count=1, window=5, \
								   sample= 0.0, negative= 8)
		best_model.save(results_dir + 'best_model')

	if part == "g":
		print("(g): word2vec based similarity")
		print 'Loading model...'
		best_model = gensim.models.Word2Vec.load(results_dir + 'best_model')
		house_home_sim = best_model.similarity('house.n','home.n')
		house_time_sim = best_model.similarity('house.n','time.n')
		home_time_sim = best_model.similarity('home.n','time.n')
		print "house:home similarity: \t" + str(house_home_sim)
		print "house:time similarity: \t" + str(house_time_sim)
		print "home:time  similarity: \t" + str(home_time_sim)

	if part == "h":
		# Train LDA model, write it to file
		print("(h) training LDA model...")
		import logging
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		lda(vectors, id2word)

	if part == "i":
		print("(i): lda-based similarity")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		house_id, home_id, time_id = word2id['house.n'], word2id['home.n'], word2id['time.n']
		lda_model = gensim.models.ldamodel.LdaModel.load(results_dir + 'lda_model')
		house_lda_vector = lda_model[vectors[house_id]]
		home_lda_vector = lda_model[vectors[home_id]]
		time_lda_vector = lda_model[vectors[time_id]]
		house_home_sim = cosine_similarity(house_lda_vector, home_lda_vector)
		house_time_sim = cosine_similarity(house_lda_vector, time_lda_vector)
		home_time_sim  = cosine_similarity(home_lda_vector, time_lda_vector)
		print "house:home similarity: \t" + str(house_home_sim)
		print "house:time similarity: \t" + str(house_time_sim)
		print "home:time  similarity: \t" + str(home_time_sim)

	if part == "j":
		print("(j) get words from LDA topics")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		ldaModel = gensim.models.ldamodel.LdaModel.load(results_dir + 'lda_model')

		# Write word distributions for each topic to file (for reporting)
		with open(results_dir + 'words_by_topic.txt', 'w') as f:
			for topicID in xrange(0,100):
				f.write('Topic:\t %s \n' % topicID)
				rank = 1
				for topic_word in get_topic_words(ldaModel, topicID, id2word):
					f.write('%i.\t%s\n' % (rank, topic_word))
					rank += 1
				f.write('\n')
