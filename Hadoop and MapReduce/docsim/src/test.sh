#!/bin/bash

module purge
module load python/gcc/3.7.9


python test.py ../tiny/text*  -r hadoop \
       --hadoop-streaming-jar $HADOOP_LIBPATH/$HADOOP_STREAMING \
       --output-dir docsim\
       --python-bin /share/apps/peel/python/3.7.9/gcc/bin/python \
