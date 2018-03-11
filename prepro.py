import nltk
import re
import os
import sys
import numpy
import numpy as np
import matplotlib.pylab as plt
numpy.set_printoptions(threshold=numpy.nan)
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

def getSentenceVectors(filepath):
	if os.path.isfile(filepath)==False:
		print("Error: File path does not exist")
		os._exit(0)

	inputFile = open(filepath, 'r')
	inputFileContents = inputFile.read()
	tokenized_input = sent_tokenize(inputFileContents)
	op_file = open('output1.txt', 'w')
	print(tokenized_input, file=op_file)
	word_tokenized = [word_tokenize(t) for t in tokenized_input]

	stemmed = []
	porter = PorterStemmer()
	for i in range(0,len(tokenized_input)):
		stemmed.append([porter.stem(q) for q in word_tokenized[i]])

	print(stemmed, file=op_file)

	a=[]
	for i in range(0, len(stemmed)):
		for t in stemmed[i]:
			a.append(''.join(t))


	write_contents = ' '.join(a)

	r = sent_tokenize(write_contents)
	vec = CountVectorizer(r, stop_words=u'english')
	vectorop = vec.fit_transform(r)
	op2_file = open('op3.txt','w')
	np.savetxt("op3.txt", vectorop.toarray(), newline='\n')
	return vectorop


