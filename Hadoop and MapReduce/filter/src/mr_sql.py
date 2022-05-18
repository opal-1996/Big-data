#! /usr/bin/env python

from mrjob.job import MRJob


class MRMoviesByGenreCount(MRJob):
    """
    Find the distinct number of movies in each genre.
    """

    def mapper(self, _, line):
        """
        Implement your mapper here!

        Don't forget that we don't care about movies that don't have a genre
        assigned to them.
        
        Parameters:
            -: None
                A value parsed from input and by default it is None because the input is just raw text.
                We do not need to use this parameter.
            line: str
                each single line a file with newline stripped

            Yields:
                (key, value) 
                key: str
                value: int
                You'll have to design your format for the (key, value) pairs so that
                your reducer does the result operations correctly.

        """
        # yield key, value pairs for your program
        genre = line.split(",")[1]
        if genre != "Horror" and genre != "(no genres listed)":
            yield(genre, 1)

    # optional: implement the combiner:
    def combiner(self, genre, counts):
        #start using the key-value pairs to calculate the query result
        yield (genre, sum(counts))

    def reducer(self, genre, counts):
        """
        Implement your reducer here!

        You'll have to use the (key, list(values)) pairs that you designed
        to calculate the end result and output a (key, result) pair.
        
        Parameters:
            key: str
                same as the key defined in the mapper
            values: list
                list containing values corresponding to the key defined in the mapper

            Yields:
                key: str
                    same as the key defined in the mapper
                value: int
                    value corresponding to each key. This is the resulting matrix C
        """
        # use the key-value pairs to calculate the query result
        if sum(counts) >= 100:
            yield (genre, sum(counts))

# don't forget the '__name__' == '__main__' clause!
if __name__ == '__main__':
    MRMoviesByGenreCount.run()