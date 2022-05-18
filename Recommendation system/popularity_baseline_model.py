#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Popularity baseline model with evaluation metric [precisionAtk, mean precision, ndcgAtk].
"""
import numpy as np
import pandas as pd
import pyspark.sql.functions as f

from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, desc

class PopularityRecommender():

	"""A baseline model designed to recommend the most popular items for users.
	"""

	def __init__(self, ratings_df_train, topn=100):
		"""
		Parameter: 
			ratings_df_train - a spark dataframe of training set 
			topn - int, the number of top movies we want to recommend for each userId
		"""
		self.ratings_df_train = ratings_df_train
		self.topn = topn

	def popularity_df(self):
		"""
		Returns a spark dataframe includes movies in training set sorted based on their average ratings.
		"""
		popularity_ratings_train_df = self.ratings_df_train\
								.groupBy("movieId")\
								.mean("rating")\
								.sort(desc("avg(rating)"))
		
		return popularity_ratings_train_df

	def get_seen(self):
		"""
		Get all the movies seen by each userId.
		"""
		seen = self.ratings_df_train.groupby("userId").agg(f.collect_set("movieId").alias("seen_movieIds"))
		return seen

	def recommending(self, popularity_df, seen_df, userId):
		"""
		Recommend top 100 movies for userId.
		Params:
			popularity_df -  a pandas dataframe contains the sorted movieId
			seen_df - a pandas dataframe contains the users and their seen movies
		"""
		seen_movies = seen_df[seen_df.userId==userId].iloc[0][1]# a list of seen movies by userId
		recommendations = list(popularity_df[~popularity_df['movieId'].isin(seen_movies)].iloc[:self.topn, 0])
		return recommendations

class Dataset():
	"""
	Load dataset.
	"""
	def __init__(self, spark_session, filepath):
		"""
		Construct spark dataset.
		"""
		self.spark_session = spark_session
		self.sc = spark_session.sparkContext

		#build spark data object
		self.filepath = filepath
		self.RDD = self.load_file_as_RDD(self.filepath)
		self.DF = self.load_file_as_DF(self.filepath)

	def load_file_as_RDD(self,filepath):
		ratings_RDD = self.sc.textFile(filepath)
		header = ratings_RDD.take(1)[0]

		return ratings_RDD\
			.filter(lambda line: line!=header)\
			.map(lambda line: line.split(","))\
			.map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2])))

	def load_file_as_DF(self, filepath):
		ratings_RDD = self.load_file_as_RDD(filepath)
		ratingsRDD = ratings_RDD.map(lambda tokens: Row(userId=int(tokens[0]), movieId=int(tokens[1]), rating=float(tokens[2]))) 

		return self.spark_session.createDataFrame(ratingsRDD)
	
	def split(self, df):
		"""
		Get stratified samples based on userId.
		"""
		unique_userId = [x.userId for x in df.select("userId").distinct().collect()]

		#set splitting fraction
		fraction_1 = {key:0.6 for key in unique_userId}
		fraction_2 = {key:0.5 for key in unique_userId}

		# return stratified subsets based on userId
		df_train = df.sampleBy("userId", fractions=fraction_1, seed=10)
		df_tmp = df.subtract(df_train)
		df_val = df_tmp.sampleBy("userId",fractions=fraction_2, seed=10)
		df_test = df_tmp.subtract(df_val)
	
		return df_train, df_val, df_test

class ModelEvaluator():
	"""
	Evaluates the accuracy of the top recommendations provided to a user, comparing to the items the user has actually interacted in val set or test set.
	"""
	def __init__(self, spark_session, ratings_df_test, model):
		"""
		Parameter:
			ratings_df_test - dataframe of test set
			model - an instance of PopularityRecommender()
		"""
		self.spark_session = spark_session
		self.sc = spark_session.sparkContext
		self.ratings_df_test = ratings_df_test
		self.model = model

	def evaluate(self):
		"""Return the precision_at_k/MAP/NDCG scores on test set."""
		label_df_test = self.ratings_df_test.groupby("userId").agg(f.collect_set("movieId").alias("seen_movieIds")).toPandas()
		labels = label_df_test["seen_movieIds"].tolist()#ground truth labels
		
    #make predictions 
		popularity_df = self.model.popularity_df().toPandas()
		seen_df = self.model.get_seen().toPandas()
		recommendations = label_df_test[["userId"]]#initialize recommendations
		recommendations['recommendations'] = recommendations['userId'].map(lambda x: self.model.recommending(popularity_df, seen_df, x))
		predictions = recommendations["recommendations"].tolist()
		predictionAndlabel = [(predictions[i], labels[i]) for i in range(len(predictions))]
		
    #make evaluations
		rdd = self.sc.parallelize(predictionAndlabel)
		metrics = RankingMetrics(rdd)
		print("The precision at {} for the popularity model is {}".format(self.model.topn, metrics.precisionAt(self.model.topn)))
		print("The mean average precision for the popularity model is {}".format(metrics.meanAveragePrecision))
		print("The NDCG at {}  for the popularity model is {}".format(self.model.topn, metrics.ndcgAt(self.model.topn)))

def main():
	#build spark session
	spark = SparkSession.builder.appName('part1').getOrCreate()

	# load data into DF
	dataset = Dataset(spark, "ratings.csv")
	ratings_df = dataset.DF
	ratings_df_train, ratings_df_val, ratings_df_test = dataset.split(ratings_df)

	#initialize a recommender instance
	recommender = PopularityRecommender(ratings_df_train, 100)

	#evaluate on val and test sets
	#val set
	print("-----------------Val Set---------------------------")
	evaluator = ModelEvaluator(spark, ratings_df_val, recommender)
	evaluator.evaluate()

	#test set
	print("-----------------Test Set---------------------------")
	evaluator = ModelEvaluator(spark, ratings_df_test, recommender)
	evaluator.evaluate()

if __name__ == "__main__":
	main()











