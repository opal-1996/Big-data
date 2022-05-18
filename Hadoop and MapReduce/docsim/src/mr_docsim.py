#! /usr/bin/env python

import re
import string
import pathlib

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.compat import jobconf_from_env
from collections import Counter



WORD_RE = re.compile(r"[\S]+")


class MRDocSim(MRJob):
    """
    A class to count word frequency in an input file.
    """

    def mapper_get_words(self, _, line):
        """

        Parameters:
            -: None
                A value parsed from input and by default it is None because the input is just raw text.
                We do not need to use this parameter.
            line: str
                each single line a file with newline stripped

            Yields:
                (key, value) pairs
        """

        # This part extracts the name of the current document being processed
        current_file = jobconf_from_env("mapreduce.map.input.file")

        # Use this doc_name as the identifier of the document
        doc_name = pathlib.Path(current_file).stem

        for word in WORD_RE.findall(line):
            # strip any punctuation
            word = word.strip(string.punctuation)

            # enforce lowercase
            word = word.lower()

            # TODO: start implementing here!
            yield doc_name, word

    def reducer_each_book(self, doc_name, word):
        
        yield doc_name, list(word)

    def mapper_books(self, doc_name, words):
        d = Counter()
        for word in words: 
            d[word] += 1
        yield 'all', (doc_name, d)


    def reducer_books(self, key, values):
        k = []
        v = []
        # if len(list(values)[0]) == 2:
        #     res = list(values)[0]
        #     yield '1', len(res)
        #     # yield (res[0][0], res[0][1]), (res[1][0], res[1][1])
        # else:
        for tup in list(values):
            k.append(tup[0]) #keys for doc
            v.append(tup[1]) #values of each doc, which would be dictionary
        for i in range(len(k)):
            for j in range(i+1, len(k)):
                yield(k[i], k[j]),(v[i], v[j])
    
    def reducer_fin(self, key, values):
        values = list(values)
        # print(type(values))
        s1 = set(values[0])
        s2 = set(values[1])
        common = s1.intersection(s2)
        v=0
        for k in common:
            v += min(values[0][k], values[1][k])
        # yield key, (list(values)[0], list(values)[1])
        yield key, v

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_get_words,
                reducer = self.reducer_each_book
            ),
           MRStep(
                mapper=self.mapper_books,
                reducer=self.reducer_books
            )
            ,
            MRStep(
                mapper = self.reducer_fin
            )
        ]

# this '__name__' == '__main__' clause is required: without it, `mrjob` will
# fail. The reason for this is because `mrjob` imports this exact same file
# several times to run the map-reduce job, and if we didn't have this
# if-clause, we'd be recursively requesting new map-reduce jobs.
if __name__ == "__main__":
    # this is how we call a Map-Reduce job in `mrjob`:
    MRDocSim.run()
