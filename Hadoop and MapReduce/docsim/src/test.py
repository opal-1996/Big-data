#! /usr/bin/env python
import re
import string
import pathlib
import numpy as np

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.compat import jobconf_from_env

WORD_RE = re.compile(r"[\S]+")

class MRDocSim(MRJob):
	"""
	A class to count word frequency in an input file.
	"""
	
	def mapper1(self, _, line):

		"""
		First Mapper.
		Parameters:
			-: None
				A value parsed from input and by default it is None because the input is just raw text.
				We do not need to use this parameter.
			line: str
				each single line a file with newline stripped
			Yields:
				key = name_word
				value = 1
		"""
		# This part extracts the name of the current document being processed
		current_file = jobconf_from_env("mapreduce.map.input.file")
		doc_name = pathlib.Path(current_file).stem
		for word in WORD_RE.findall(line):
			# strip any punctuation
			word = word.strip(string.punctuation)
			# enforce lowercase
			word = word.lower()
			# TODO: start implementing here!
			name_word = doc_name + " " + word
			yield (name_word, 1)

	def combiner1(self, name_word, counts):
		doc_name, word = name_word.split(' ')
		yield (name_word, (doc_name, word, sum(counts)))

	def reducer1(self, name_word, counts):
		doc_name, word = name_word.split(' ')
		yield (name_word, (doc_name, word, sum(counts)))

	def mapper2(self, _, line):
		doc_name, word, num = line
		yield word, (doc_name,num)

	def combiner2(self, word, doc_name_sum):
		m = len(doc_name_sum)
		sim_matrix = [ ]
		for i in range(m):
			temp = [ ]
			for j in range(m):
				itm = doc_name_sum[i][1][0] + " " + doc_name_sum[j][1][0] + " " + str(min(doc_name_sum[i][1][1], doc_name_sum[j][1][1]))
				temp.append(itm)
			sim_matrix.append(temp)
		yield (word, sim_matrix)

	def reducer2(self, word, doc_name_num):
		# m = len(doc_name_sum)
		# sim_matrix = [ ]
		# for i in range(m):
		# 	temp = [ ]
		# 	for j in range(m):
		# 		itm = doc_name_sum[i][1][0] + " " + doc_name_sum[j][1][0] + " " + str(min(doc_name_sum[i][1][1], doc_name_sum[j][1][1]))
		# 		temp.append(itm)
		# 	sim_matrix.append(temp)
		# yield (word, sim_matrix)
		yield (word, 'example')

	def reducer3(self, _, list_of_matrix):
		"""
		Third reducer.
		Parameters: 
			list_of_matrix: a list of sim_matrix from last reducer
		"""
		docA_docB_list = []
		for i in range(len(list_of_matrix)):
			temp = list_of_matrix[i]
			for j in range(temp.shape[0]):
				if temp[j][0][0] not in docA_docB_list:
					docA_docB_list.append(temp[j][0][0])

		#Initialize a dictionary that gives the sim(A,B)
		sim = {}
		for i in docA_docB_list:
			sim[i] = 0

		for i in range(len(list_of_matrix)):
			temp = list_of_matrix[i].flatten()
			for j in range(len(temp)):
				if temp[j][0] in sim.keys():
					sim[temp[j][0]] += temp[j][1]

		for i in docA_docB_list:
			yield (i, sim.get(i,0))


	def steps(self):
		return [
			MRStep(mapper=self.mapper1,
				combiner=self.combiner1,
				reducer=self.reducer1),
			MRStep(mapper=self.mapper2,
				combiner=self.combiner2,
				reducer=self.reducer2)
			MRStep(reducer=self.reducer3)
			]

# this '__name__' == '__main__' clause is required: without it, `mrjob` will
# fail. The reason for this is because `mrjob` imports this exact same file
# several times to run the map-reduce job, and if we didn't have this
# if-clause, we'd be recursively requesting new map-reduce jobs.
if __name__ == "__main__":
	# this is how we call a Map-Reduce job in `mrjob`:
	MRDocSim.run()
