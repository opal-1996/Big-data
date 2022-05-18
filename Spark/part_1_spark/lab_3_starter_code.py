#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, countDistinct, desc, asc
import pyspark.sql.functions as F
import numpy as np
from pyspark.sql.types import FloatType, StringType

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Lab 3 Example dataframe loading and SQL query')

    # Load the boats.txt and sailors.json data into DataFrame
    boats = spark.read.csv(f'hdfs:/user/{netID}/boats.txt')
    sailors = spark.read.json(f'hdfs:/user/{netID}/sailors.json')

    print('Printing boats inferred schema')
    boats.printSchema()
    print('Printing sailors inferred schema')
    sailors.printSchema()
    # Why does sailors already have a specified schema?

    print('Reading boats.txt and specifying schema')
    boats = spark.read.csv('boats.txt', schema='bid INT, bname STRING, color STRING')

    print('Printing boats with specified schema')
    boats.printSchema()

    # Give the dataframe a temporary view so we can run SQL queries
    boats.createOrReplaceTempView('boats')
    sailors.createOrReplaceTempView('sailors')
    # Construct a query
    print('Example 1: Executing SELECT count(*) FROM boats with SparkSQL')
    query = spark.sql('SELECT count(*) FROM boats')

    # Print the results to the console
    query.show()

    #####--------------YOUR CODE STARTS HERE--------------#####
    #####1.5 SQL queries and DataFrame methods

    #####Question 1: How would you express the following computation using SQL instead of the object interface?
    question_1_query = spark.sql("SELECT sid, sname, age FROM sailors WHERE age > 40")
    question_1_query.show()

    #####Question 2: How would you express the following using the object interface instead of SQL?
    reserves = spark.read.json(f'hdfs:/user/{netID}/reserves.json')
    reserves.createOrReplaceTempView('reserves')
    question_2_query = reserves.filter(reserves.bid!=101).groupBy(reserves.sid).agg(count(reserves.bid))
    question_2_query.show()

    #####Question 3:  Using a single SQL query, how many distinct boats did each sailor reserve? The resulting DataFrame 
    #####should include the sailor's id, name, and the count of distinct boats.
    # question_3_query = sailors.join(reserves, ["sid"]).groupBy(sailors.sid, sailors.sname).agg(countDistinct(reserves.bid).alias("distinct_boats")).sort(desc("distinct_boats"))
    # question_3_query.show()
    question_3_query = spark.sql("SELECT temp_reserves.sid, sailors.sname, temp_reserves.num FROM sailors LEFT JOIN (SELECT reserves.sid, COUNT(DISTINCT reserves.bid) AS num FROM reserves GROUP BY reserves.sid) temp_reserves ON sailors.sid = temp_reserves.sid")
    question_3_query.show()

    ####1.6 Bigger datasets
    #Load the datasets artist_term.csv and tracks.csv into DataFrame
    artist_term = spark.read.csv(f'hdfs:/user/{netID}/artist_term.csv')
    tracks = spark.read.csv(f'hdfs:/user/{netID}/tracks_copy.csv')

    #Predefine the schema
    artist_term = spark.read.csv('artist_term.csv', schema='artistID STRING , term STRING')
    tracks = spark.read.csv('tracks_copy.csv', schema='trackID STRING, title STRING, release STRING, year INT, duration DOUBLE, artistID STRING')
    # artist_term.printSchema()
    # tracks.printSchema()
    
    #Creat temp view 
    artist_term.createOrReplaceTempView('artist_term')
    tracks.createOrReplaceTempView('tracks')

    ####Question 4:  Implement a query using Spark transformations which finds for each artist term, compute the median year of release, 
    ####maximum track duration, and the total number of artists for that term (by ID).
    def find_median(values_list):
        try:
            median = np.median(values_list) #get the median of values in a list in each row
            return round(float(median),2)
        except Exception:
            return None

    def find_max(values_list):
        try:
            maximum = np.max(values_list) #get the max of values in a list in each row
            return round(float(maximum),2)
        except Exception:
            return None

    def find_num(values_list):
        try:
            num = np.unique(values_list) #get the unique number of values in a list in each row
            return len(num)
        except Exception:
            return None

    def find_mean(values_list):
        try:
            mean = np.mean(values_list) #get the mean of values in a list in each row
            return round(float(mean),2)
        except Exception:
            return None

    median_finder = F.udf(find_median,FloatType())
    max_finder = F.udf(find_max,FloatType())
    num_finder = F.udf(find_num,StringType())
    mean_finder = F.udf(find_mean,FloatType())

    question_4_query = artist_term.join(tracks, artist_term.artistID == tracks.artistID).groupBy(artist_term.term).agg(
        F.collect_list(tracks.year).alias("YearsOfRelease"),
        F.collect_list(tracks.duration).alias("Durations"),
        F.collect_list(tracks.artistID).alias("ArtistIDs"))

    question_4_query = question_4_query.withColumn("MedianYearOfRelease", median_finder("YearsOfRelease")).withColumn("MaximumDuration", max_finder("Durations")).withColumn("NumOfArtist", num_finder("ArtistIDs")).withColumn("MeanDuration", mean_finder("Durations")).sort(asc("MeanDuration"))
    question_4_query_final = question_4_query.drop("YearsOfRelease", "Durations", "ArtistIDs", "MeanDuration")
    question_4_query_final.show(10)


    ####Question 5: Create a query using Spark transformations that finds the number of distinct tracks associated (through artistID) to each term. 
    question_5_query = artist_term.join(tracks, artist_term.artistID == tracks.artistID).groupBy(artist_term.term).agg(countDistinct(tracks.trackID).alias("distinct_track")).sort(desc("distinct_track"))
    question_5_query.tail(10)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
