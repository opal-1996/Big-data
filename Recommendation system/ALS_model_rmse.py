#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALS model with evaluation metric RMSE.
"""
import gc

#pyspark.sql to get the spark session
from pyspark.sql import SparkSession, Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


class ALSModel():
	"""
	Collaborative filtering recommender with Alternating Least Square Matrix Factorization, implemented by Spark.
	"""
	def __init__(self, spark_session, ratings_df):
		"""
		Params:
			ratings_df - a spark dataframe for ratings.csv
		"""
		self.spark_session = spark_session
		self.sc = spark_session.sparkContext
		self.ratings_df = ratings_df

		#initialize the ALS model
		self.model = ALS(
						userCol="userId",
						itemCol="movieId",
						ratingCol="rating",
						nonnegative = True, 
						implicitPrefs = False,
						coldStartStrategy="drop"
						)	
  
	def train_val_test_split(self):
		"""
		Split orginal dataset into train/val/test datasets with split ratio: [0.6, 0.2, 0.2]
		Return:
			ratings_df_train - spark DF
			ratings_df_val - spark DF
			ratings_df_test - spark DF
		"""
		unique_userId = [x.userId for x in self.ratings_df.select("userId").distinct().collect()]
		#set splitting fraction
		fraction_1 = {key:0.6 for key in unique_userId}
		fraction_2 = {key:0.5 for key in unique_userId}

		# return stratified subsets based on userId
		ratings_df_train = self.ratings_df.sampleBy("userId", fractions=fraction_1, seed=10)
		ratings_df_tmp = self.ratings_df.subtract(ratings_df_train)
		ratings_df_val = ratings_df_tmp.sampleBy("userId",fractions=fraction_2, seed=10)
		ratings_df_test = ratings_df_tmp.subtract(ratings_df_val)
		return ratings_df_train, ratings_df_val, ratings_df_test

	def tune_model(self, regParams, ranks,maxIter ):
		"""
		Hyperparameter tuning for ALS model.
		Params:
			maxIter - int, the maximum number of learning iteration 
			regParams - list of float, regularization params
			ranks - list of float, number of latent factors
		"""
		#get train/val/test data
		df_train, df_val, df_test = self.train_val_test_split()

		#get the best model by tuning on hyperparameters
		self.model = tune_ALS(self.model, df_train, df_val, regParams, ranks, maxIter)

		#test the performance of the model on test set
		predictions = self.model.transform(df_test)
		evaluator = RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
		rmse = evaluator.evaluate(predictions)

		print("The hold-out dataset RMSE of the best tuned model is :", rmse)

		#clean up 
		del df_train, df_val, df_test, predictions, evaluator
		gc.collect()#garbage collector

class Dataset():
	"""
	Load dataset.
	"""
	def __init__(self, spark_session, filepath):
		"""
		Construct spark dataset.
		"""
		self.spark_session = spark_session
		self.sc = self.spark_session.sparkContext

		#build spark data object
		self.filepath = filepath
		self.RDD = self.load_file_as_RDD(self.filepath)
		self.DF = self.load_file_as_DF(self.filepath)

	def load_file_as_RDD(self, filepath):
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


#function for tuning the model, with metric RMSE
def tune_ALS(model, df_train, df_val, regParams, ranks, maxIter):
	"""
	Grid search method to select the best model based on RMSE of validation data.

	Params:
		model - ALS model
		df_train - spark DF with columns []
		df_val - spark DF with columns []
		maxIter - int, max number of learning iterations
		regParams - list of float, one dimension of hyper-param tuning grid
		ranks - list of float, one dimension of hyper-param tuning grid

	Return:
		The best fitted ALS model with lowest RMSE score on val data.
	"""

	#initialize
	min_error = float("inf")
	best_rank = -1
	best_regularization = 0
	best_maxIter = 0
	best_model = None

	for rank in ranks:
		for reg in regParams:
				for iter in maxIter:
					#get ALS model
					als = model.setRank(rank).setRegParam(reg).setMaxIter(iter)
					#train ALS model
					train_model = als.fit(df_train)
					#evaluate the model by computing the RMSE on the validation data
					predictions = train_model.transform(df_val)
					evaluator = RegressionEvaluator(metricName="rmse",
											labelCol="rating",
											predictionCol="prediction"
											)
					rmse = evaluator.evaluate(predictions)
					print("latent factors = {}, regularization = {}, maxIter = {}: " "validation RMSE  = {}".format(rank, reg, iter, rmse))

					if rmse < min_error:
						min_error = rmse
						best_rank = rank
						best_regularization = reg
						best_maxIter = iter
						best_model = train_model
            

	print("The best model has {} maxIter, {} latent factors and " "regularization = {}".format(best_maxIter, best_rank, best_regularization))
	return best_model 

def main():
	#initialize params
	ratings_path = "ratings.csv"

	#initialize spark 
	spark = SparkSession.builder.appName("part1").getOrCreate()

	#load dataset and initialize ALS recommender system
	ratings_DF = Dataset(spark, ratings_path).DF
	als = ALSModel(spark, ratings_DF)

	#add hyperparameters and their respective values to tune the model
	ranks = [10, 50, 100]
	regParams = [.01, .05, 0.1]
	maxIter = [10, 20]

	als.tune_model(regParams, ranks, maxIter)

if __name__ == "__main__":
	main()
