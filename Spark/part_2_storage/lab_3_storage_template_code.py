#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit lab_3_storage_template_code.py <any arguments you wish to add>
'''


# Import command line arguments and helper functions(if necessary)
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession



def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''
    #####--------------YOUR CODE STARTS HERE--------------#####
    #Use this template to as much as you want for your parquet saving and optimizations!
    # peopleSmall = spark.read.csv('hdfs:/user/bm106/pub/people_small.csv', header=True, schema="first_name STRING, last_name STRING, income FLOAT, zipcode INT")
    # peopleMedium = spark.read.csv('hdfs:/user/bm106/pub/people_medium.csv', header=True, schema="first_name STRING, last_name STRING, income FLOAT, zipcode INT")
    # peopleLarge = spark.read.csv('hdfs:/user/bm106/pub/people_large.csv', header=True, schema="first_name STRING, last_name STRING, income FLOAT, zipcode INT")

    # peopleSmall.write.parquet("hdfs:/user/qy692/peopleSmall.parquet")
    # peopleMedium.write.parquet("hdfs:/user/qy692/peopleMedium.parquet")
    # peopleLarge.write.parquet("hdfs:/user/qy692/peopleLarge.parquet")    

    # peopleSmall = spark.read.parquet('hdfs:/user/qy692/peopleSmall.parquet')
    # peopleMedium = spark.read.parquet('hdfs:/user/qy692/peopleMedium.parquet')
    # peopleLarge = spark.read.parquet('hdfs:/user/qy692/peopleLarge.parquet')


    #-------------------Partition by column zipcode for query in pq_avg_income.py---------------------------------#
    # peopleSmall.write.option("header", True).partitionBy("zipcode").mode("overwrite").parquet("./avg_small/")
    # peopleMedium.write.option("header", True).partitionBy("zipcode").mode("overwrite").parquet("./avg_medium/")
    # peopleLarge.write.option("header", True).partitionBy("zipcode").mode("overwrite").parquet("./avg_large/")

    #-------------------Repartition by column zipcode for query in pq_avg_income.py---------------------------------#
    # avg_small = peopleSmall.repartition("zipcode")
    # avg_small.write.option("header",True).mode("overwrite").parquet("./avg_small_zipcode/")

    # avg_medium = peopleMedium.repartition("zipcode")
    # avg_medium.write.option("header",True).mode("overwrite").parquet("./avg_medium_zipcode/")

    # avg_large = peopleLarge.repartition("zipcode")
    # avg_large.write.option("header",True).mode("overwrite").parquet("./avg_large_zipcode/")

    ##The difference between partitionBy and repartition:
    #PySpark repartition() is a DataFrame method that is used to increase or reduce the partitions in memory and when written to disk, it create all part files in a single directory.
    #PySpark partitionBy() is a method of DataFrameWriter class which is used to write the DataFrame to disk in partitions, one sub-directory for each unique value in partition columns.

    #-------------------Partition by column first_name for query in pq_anna.py---------------------------------#
    # peopleSmall.write.option("header", True).partitionBy("first_name").mode("overwrite").parquet("./anna_small")
    # peopleMedium.write.option("header", True).partitionBy("first_name").mode("overwrite").parquet("./anna_medium/")
    # peopleLarge.write.option("header", True).partitionBy("first_name").mode("overwrite").parquet("./anna_large/")

    #-------------------Partition by column last_name for query in pq_max_income.py---------------------------------#
    # peopleSmall.write.option("header", True).partitionBy("last_name").mode("overwrite").parquet("./max_small")
    # peopleMedium.write.option("header", True).partitionBy("last_name").mode("overwrite").parquet("./max_medium/")
    # peopleLarge.write.option("header", True).partitionBy("last_name").mode("overwrite").parquet("./max_large/")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
    