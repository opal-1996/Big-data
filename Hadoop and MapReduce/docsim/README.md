# Part 5: Document similarity

## Explain each step of your solution

There are mainly three steps for solving this problem:

Step1, use the first mapping-reducing process to get the sum of counts of doc-word, that is :(doc_name + " " + word, sum(counts)), which represents the number of occurence of specific word in doc_name.

Step2, use the second mapping-reducing process to get the similarity matrix for a single word, that is : (word, sim_matrix), where each item in sim_matrix represents (doc_name_i doc_name_j, minimum occurence of word in both docs).

Step3, use another reducer to sum up the similarity score based on words. The matrix after summation would represent the final result we want.

## What problems, if any, did you encounter?

I kept receiving "can't fetch history log and missing job ID" errors during the whole process. 

## How does your solution scale with the number of documents?

Even though I tested with tiny dataset, it didn't seem faster than dealing with larger amount of data, so Hadoop and MapReduce might be more useful when dealing with big data.
