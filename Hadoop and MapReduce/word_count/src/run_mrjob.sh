#!/bin/bash

module purge
module load python/gcc/3.7.9

python mr_wordcount.py ../book.txt -r hadoop \
       --hadoop-streaming-jar $HADOOP_LIBPATH/$HADOOP_STREAMING \
       --output-dir word_count \
       --python-bin /share/apps/peel/python/3.7.9/gcc/bin/python \
