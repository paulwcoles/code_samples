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

from question1 import *
import json


'''
Verify whether a given word is represented in an LDA model
input: context word to find in model
input: LDA model (loaded from Gensim load method)
output: Boolean, true iff context word has representation in model
'''
def word_in_model(context_word, model):
	word_in_model = True
	try:
		model[context_word]
	except:
		word_in_model = False
	return word_in_model


'''
Add two vectors
input: vector1
input: vector2
output: addVector, the resulting vector when adding vector1 and vector2
'''
def addition(vector1, vector2):
	# Where both input vectors dense:
	if is_it_dense(vector1) and is_it_dense(vector2):
		p_vector = []
		index = 0
		for element in vector1:
			if index < len(vector2):
				p_vector.insert(index, element + vector2[index])
			else:
				p_vector.insert(index, element)
			index += 1
		return p_vector
	# Where at least one input vector sparse, return sparse vector
	else:
		if not is_it_dense(vector1):
			vector1 = convert_to_dense(vector1)
		if not is_it_dense(vector2):
			vector2 = convert_to_dense(vector2)
		p_vector = addition(vector1, vector2)
		return convert_to_sparse(p_vector)

'''
Multiply two vectors
input: vector1
input: vector2
output: mulVector, the resulting vector when multiplying vector1 and vector2
'''
def multiplication(vector1, vector2):
	# Where both input vectors dense:
	if is_it_dense(vector1) and is_it_dense(vector2):
		p_vector = []
		index = 0
		for element in vector1:
			if index < len(vector2):
				p_vector.insert(index, element * vector2[index])
			else:
				p_vector.insert(index, 0)
			index += 1
		return p_vector
	# Where at least one input vector sparse, return sparse vector
	else:
		if not is_it_dense(vector1):
			vector1 = convert_to_dense(vector1)
		if not is_it_dense(vector2):
			vector2 = convert_to_dense(vector2)
		p_vector = multiplication(vector1, vector2)
		return convert_to_sparse(p_vector)

'''
Get probability of LDA topic z, given target word w
input: ldaModel
input: topicID as an integer
input: wordVector in frequency space
output: probability of the topic with topicID in the ldaModel, given the wordVector
'''
def prob_z_given_w(ldaModel, topicID, wordVector):
	try:
		# Verify sparse encoding
		if is_it_dense(wordVector):
			sparse_wordVector = convert_to_sparse(wordVector)
		else:
			sparse_wordVector = wordVector
			topic_distr = ldaModel[sparse_wordVector]
			for topic_tuple in topic_distr:
				if topic_tuple[0] == topicID:
					p_z_given_w = topic_tuple[1]
					break
				else:
					p_z_given_w = 0
	except Exception:
		print 'Topic word is out of vocabulary, zero probability returned'
		p_z_given_w = 0
	return p_z_given_w

'''
Get probability of target word w, given LDA topic z
input: ldaModel
input: targetWord as a string
input: topicID as an integer
output: probability of the targetWord, given the topic with topicID in the ldaModel
'''
def prob_w_given_z(ldaModel, targetWord, topicID):
	try:
		words_distr = ldaModel.show_topic(topicID, topn = 100000)
		for word_tuple in words_distr:
			if word_tuple[1] == targetWord:
				p_w_given_z = word_tuple[0]
				break
			else:
				p_w_given_z = 0
	except Exception:
		print 'Word is out of vocabulary, zero probability returned'
		p_z_given_w = 0
	return p_w_given_z

'''
Get the best substitution word in a given sentence, according to
a given model space (tf-idf, word2vec, LDA) and type (addition, multiplication, lda)
'''
def best_substitute(jsonSentence, thesaurus, word2id, id2word, model, model_choice, frequencyVectors, csType):
	## 1. Extract values from JSON
	sentence_dict = json.loads(jsonSentence)
	sentence_tokens = sentence_dict["sentence"].split()
	sentence_id = sentence_dict["id"]
	target_word = sentence_dict["target_word"]
	target_id = word2id[sentence_dict["target_word"]]
	target_position = int(sentence_dict["target_position"])

	## 2. Establish context window
	context_word_ids = []
	if target_position > 4:
		window_start = target_position - 5
	else:
		window_start = 0
	if len(sentence_tokens) - target_position < 4:
		window_end = len(sentence_tokens)
	else:
		window_end = target_position + 5
	# List context word IDs
	for context_word in sentence_tokens[window_start:window_end + 1]:
		context_index = 0
		# Ignore out of vocab words
		if context_word in word2id:
			context_word_ids.insert(context_index, word2id[context_word])

	## 3. Get context-sensitive vectors
	context_sen_vectors = {}
	print '\nMaking context-sensitive vectors...'
	if csType == "addition":
		for context_word_id in context_word_ids:
				if model_choice == 'tf-idf':
					context_sen_vectors[context_word_id] = addition(model[target_id], \
					model[context_word_id])
				else:	# Using word2vec space
					context_word = id2word[context_word_id]
					if len(context_word_ids) > 0 and word_in_model(context_word, model):
						context_sen_vectors[context_word_id] = addition(model[target_word], model[context_word])
					else:
						context_sen_vectors[context_word_id] = []
	elif csType == "multiplication":
		for context_word_id in context_word_ids:
			if model_choice == 'tf-idf':
				context_sen_vectors[context_word_id] = multiplication\
				(model[target_id], model[context_word_id])
			else:	# Using word2vec space
				context_word = id2word[context_word_id]
				if len(context_word_ids) > 0 and word_in_model(context_word, model):
					context_sen_vectors[context_word_id] = multiplication(model[target_word], model[context_word])
				else:
					context_sen_vectors[context_word_id] = []
	elif csType == "lda":
		for context_word_id in context_word_ids:
			v_t_c = []
			for topicID in xrange(0,100):
				try:
					left_prob = prob_z_given_w(model, topicID, frequencyVectors[context_word_id])
					right_prob = prob_w_given_z(model, id2word[context_word_id], topicID)
					v_t_c.append(left_prob * right_prob)
				except:
					print 'Empty vector appendeded.'
			context_sen_vectors[context_word_id] = v_t_c

	## 4. Find the best substitution
	print 'Choosing best substitute...'
	top_score = ('no winner', 0.0)
	# Iterate over all possible substitutions
	for candidate in thesaurus[target_word]:
		score = 0.0
	# Get vector for each candidate
		if model_choice == 'tf-idf':
			candidate_vector = model[word2id[candidate]]
		elif model_choice == 'word2vec':
			try:
				candidate_vector = model[candidate]
			except Exception:
				candidate_vector = []
				print 'No vector for candidate %s' % candidate
	# LDA
		else:
			candidate_vector = model[frequencyVectors[word2id[candidate]]]

	# Sum, over all context words, cosine similarity of candidate and context-sensitive vector
		for context_word_id in context_word_ids:
			score += cosine_similarity(candidate_vector, context_sen_vectors[context_word_id])
	# Update top score if beaten
		if score > top_score[1]:
			top_score = (candidate, score)
	## 5. Prepare return values (inc. remove POS tag from winner)
	winning_sub = str(top_score[0])
	winning_score = top_score[1]
	winning_sub = winning_sub.split('.')[0]
	return target_word, sentence_id, winning_sub, winning_score


'''
Get user choice of vector space during substitution prediction
input: None
output: string of user choice, either 'tf-idf' or 'word2vec'
'''
def get_model_choice():
	model_choice = raw_input('Choose vector space:\n1. tf-idf\n2. word2vec\n--> ')
	if model_choice == "1":
		return "tf-idf"
	elif model_choice == "2":
		return "word2vec"
	else:
		print 'Invalid choice, enter 1 or 2.'
		get_model_choice()



'''
Get user choice of vector aggregation method during substitution prediction
input: None
output: string of user choice, either 'tf-idf' or 'word2vec'
'''
def get_agg_choice():
	agg_choice = raw_input('Choose aggregation:\n1. addition\n2. multiplication\n--> ')
	if agg_choice == "1":
		return 'addition'
	elif agg_choice == "2":
		return 'multiplication'
	else:
		print 'Invalid choice, enter 1 or 2.'
		get_agg_choice()


'''
** This function provided in skeleton code **
Load a thesaurus from disk
input: thesaurusFile, file on disk containing a thesaurus of substitution words for targets
output: the thesaurus, as a mapping from target words to lists of substitution words
'''
def load_thesaurus(thesaurusFile):
	thesaurus = {}
	with open(thesaurusFile) as inFile:
		for line in inFile.readlines():
			word, subs = line.strip().split("\t")
			thesaurus[word] = subs.split(" ")
	return thesaurus

if __name__ == "__main__":
	import sys
	import os
	# specify central data location, prepare local environment for results recording
	results_dir = './results/'
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	data_dir = '/afs/inf.ed.ac.uk/group/project/nlu/data/'
	part = sys.argv[1]

	if part == "a":
		print("(a): vector addition and multiplication")
		v1, v2, v3 , v4 = [(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)], [1, 0, 1, 0, 2], [1, 2, 0, 0, 1]
		try:
			if not set(addition(v1, v2)) == set([(0, 2), (2, 1), (4, 3), (1, 2)]):
				print("\tError: sparse addition returned wrong result")
			else:
				print("\tPass: sparse addition")
		except Exception as e:
			print("\tError: exception raised in sparse addition")
			print(e)
		try:
			if not set(multiplication(v1, v2)) == set([(0,1), (4,2)]):
				print("\tError: sparse multiplication returned wrong result")
			else:
				print("\tPass: sparse multiplication")
		except Exception as e:
			print("\tError: exception raised in sparse multiplication")
			print(e)
		try:
			addition(v3,v4)
			print("\tPass: full addition")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)
		try:
			multiplication(v3,v4)
			print("\tPass: full multiplication")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)

	if part == "b" or part == "c":
		print("(b), (c) compute best substitution words")
		id2word, word2id, frequencyVectors = load_corpus(sys.argv[2], sys.argv[3])
		thesaurus = load_thesaurus(data_dir + "test_thesaurus.txt")

		# for model_choice in ['tf-idf', 'word2vec']:
		# 	for aggregation_choice in ['addition', 'multiplication']:
		for model_choice in ['word2vec']:
			for aggregation_choice in ['multiplication']:
				print '\nComputing substitution for %s vector space, using %s aggregation method...'\
				 % (model_choice, aggregation_choice)
		 		if model_choice == 'tf-idf':
		 			model = tf_idf(frequencyVectors)
		 		else:
		 			model = gensim.models.Word2Vec.load(results_dir + 'best_model')
				with open(results_dir + '%s_%s.txt' % (aggregation_choice, model_choice)\
				, 'w') as results_doc:
					with open(data_dir + 'test.txt', 'r') as test_sentences:
						count = 0
						for jsonSentence in test_sentences:
							target_word, sentence_id, winning_sub, winning_score  = best_substitute\
							(jsonSentence, thesaurus, word2id, id2word, model, model_choice, frequencyVectors,\
							aggregation_choice)
							print 'Winning substitute chosen for sentence %s, target '\
							'\'%s\':\t%s' %(sentence_id, target_word, winning_sub)
							if winning_score > 0.0:
								results_doc.write(target_word + ' ' + str(sentence_id) \
								+ ' :: ' + winning_sub + '\n')
							else:
								count +=1
								results_doc.write(target_word + ' ' + str(sentence_id) \
								+ ' :: ' + '\n')
				print count

	if part == "d":
		print("(d): calculating P(Z|w) and P(w|Z)")
		print("\tloading corpus")
		id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
		print("\tloading LDA model")
		ldaModel = gensim.models.ldamodel.LdaModel.load(results_dir + 'lda_model')
		houseTopic = ldaModel[vectors[word2id["house.n"]]][0][0]
		try:
			if prob_z_given_w(ldaModel, houseTopic, vectors[word2id["house.n"]]) > 0.0:
				print("\tPass: P(Z|w)")
			else:
				print("\tFail: P(Z|w)")
		except Exception as e:
			print("\tError: exception during P(Z|w)")
			print(e)
		try:
			if prob_w_given_z(ldaModel, "house.n", houseTopic) > 0.0:
				print("\tPass: P(w|Z)")
			else:
				print("\tFail: P(w|Z)")
		except Exception as e:
			print("\tError: exception during P(w|Z)")
			print(e)

	if part == "e":
		print("(e): using LDA to calculate best substitution words")
		id2word, word2id, frequencyVectors = load_corpus(sys.argv[2], sys.argv[3])
		thesaurus = load_thesaurus(data_dir + "test_thesaurus.txt")
		model = gensim.models.ldamodel.LdaModel.load(results_dir + "lda_model")
		with open(data_dir + 'test.txt', 'r') as test_sentences:
			with open(results_dir + 'output_lda.txt', 'w') as results_doc:
				for jsonSentence in test_sentences:
					target_word, sentence_id, winning_sub, winning_score = best_substitute(jsonSentence, thesaurus, word2id, id2word, model, 'lda', frequencyVectors, 'lda')
					print 'Winning substitute chosen for sentence %s, target '\
					'\'%s\':\t%s\n' %(sentence_id, target_word, winning_sub)
					# Write non-zero-scoring results to file
					if winning_score > 0.0:
						results_doc.write(target_word + ' ' + str(sentence_id) \
						+ ' :: ' + winning_sub + '\n')
					else:
						results_doc.write(target_word + ' ' + str(sentence_id) \
						+ ' :: ' + '\n')
