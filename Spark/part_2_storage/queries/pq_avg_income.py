#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Python script to run benchmark on a query with a file path.
Usage:
    $ spark-submit pq_avg_income.py <file_path>
'''


# Import command line arguments and helper functions
import sys
import bench
import statistics

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def pq_avg_income(spark, file_path):
    '''Construct a basic query on the people dataset

    This function returns a uncomputed dataframe that
    will compute the average income grouped by zipcode

    Parameters
    ----------
    spark : spark session object

    file_path : string
        The path (in HDFS) to the Parquet-backed file, e.g.,
        `hdfs:/user/{YOUR NETID}/people_small.parquet

    Returns
    df_avg_income:
        Uncomputed dataframe of average income grouped by zipcode
    '''

    #TODO
    people = spark.read.parquet(file_path)
    people.createOrReplaceTempView("people")
    
    #Optimization Methods:
    ##1. Change the number of partitions
    # spark.conf.set("spark.sql.shuffle.partitions",200)
    # spark.conf.set("spark.sql.shuffle.partitions",100)
    # spark.conf.set("spark.sql.shuffle.partitions",10)

    ##2. Change the HDFS replication factor to 5
    #Replication factor dictates how many copies of a block should be kept in our cluster

    ##3. partition by column zipcode: transformations on partitioned data run faster as they execute transformations parallelly for each partition

    df_avg_income = spark.sql("SELECT zipcode, AVG(income) FROM people GROUP BY zipcode")

    return df_avg_income



def main(spark, file_path):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    which_dataset : string, size of dataset to be analyzed
    '''
    #TODO
    times = bench.benchmark(spark, 25, pq_avg_income, file_path)
    print(f'Times to run Basic Query 25 times on {file_path}')
    print(times)
    print(f'Maximum Time taken to run Basic Query 25 times on {file_path}:{max(times)}')
    print(f'Minimum Time taken to run Basic Query 25 times on {file_path}:{min(times)}')
    print(f'Median Time taken to run Basic Query 25 times on {file_path}:{statistics.median(times)}')


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
