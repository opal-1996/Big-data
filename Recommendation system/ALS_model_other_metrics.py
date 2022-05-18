#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALS model with evaluation metric PrecisionAtk, NDCGAtk, and Mean Precision.
"""
import gc

from pyspark.sql import SparkSession, Row
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as f
from pyspark.sql import HiveContext
sc = spark.sparkContext
sqlContext = HiveContext(sc)

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

	def tune_model(self, regParams, ranks, maxIter, K):
		"""
		Hyperparameter tuning for ALS model.
		Params:
			maxIter - int, the maximum number of learning iteration 
			regParams - list of float, regularization params
			ranks - list of float, number of latent factors
            K - list of int, number of recommended movies
		"""
		#get train/val/test data
		df_train, df_val, df_test = self.train_val_test_split()

		#get the best model and optimal number of recommended movies by tuning on hyperparameters
		self.model, best_k = tune_ALS(self.model, df_train, df_val, regParams, ranks, maxIter, K)

		#test the performance of the model on test set
		predictions = self.model.transform(df_test)
		filtered_predictions = get_topN(predictions, "userId", "prediction", best_k)
		pre_df = filtered_predictions.groupby("userId").agg(f.collect_set("movieId").alias("movieIds"))
		pre_lst = pre_df.select('movieIds').rdd.map(lambda row : row[0]).collect()
		label_df = df_test.groupby("userId").agg(f.collect_set("movieId").alias("movieIds"))
		label_lst = label_df.select('movieIds').rdd.map(lambda row : row[0]).collect()
			 
		predictionAndlabel = [[pre_lst[i], label_lst[i]] for i in range(len(label_lst))]
							
		rdd = self.sc.parallelize(predictionAndlabel)
		metrics = RankingMetrics(rdd)
       
		precision_at_k = metrics.precisionAt(best_k)
		mean_precision = metrics.meanAveragePrecision
		ndcg_at_k = metrics.ndcgAt(best_k)

		print("The hold-out dataset precision at {} of the best tuned model is {}.".format(best_k, precision_at_k))
		print("The hold-out dataset mean precision of the best tuned model is {}.".format(mean_precision))
		print("The hold-out dataset ndcg at {} of the best tuned model is {}.".format(best_k, ndcg_at_k))

		#clean up 
		del df_train, df_val, df_test, predictions
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

def get_topN(df, group_by_columns, order_by_column, n=1):
    window_group_by_columns = Window.partitionBy(group_by_columns)
    ordered_df = df.select(df.columns + [
        f.row_number().over(window_group_by_columns.orderBy(col(order_by_column).desc())).alias('row_rank')])
    topN_df = ordered_df.filter(f"row_rank <= {n}").drop("row_rank")
    return topN_df

def tune_ALS(model, df_train, df_val, regParams, ranks, maxIter, K):
	"""
	Grid search method to select the best model based on RMSE of validation data.

	Params:
		model - ALS model
		df_train - spark DF with columns []
		df_val - spark DF with columns []
		maxIter - int, max number of learning iterations
		regParams - list of float, one dimension of hyper-param tuning grid
		ranks - list of float, one dimension of hyper-param tuning grid
    K - list of int, number of recommended movies

	Return:
		The best fitted ALS model with lowest RMSE score on val data.
	"""

	#initialize
	max_precision = 0
	best_rank = -1
	best_regularization = 0
	best_maxIter = 0
	best_k = 0
	best_model = None

	for rank in ranks:
		for reg in regParams:
				for iter in maxIter:
					for k in K:
							#get ALS model
							als = model.setRank(rank).setRegParam(reg).setMaxIter(iter)
							#train ALS model
							train_model = als.fit(df_train)
							#evaluate the model by computing the RMSE on the validation data
							predictions = train_model.transform(df_val)
					
							#only return the first k recommended movies
							filtered_predictions = get_topN(predictions, "userId", "prediction", k)
			 
							#evaluate
							pre_df = filtered_predictions.groupby("userId").agg(f.collect_set("movieId").alias("movieIds"))
							pre_lst = pre_df.select('movieIds').rdd.map(lambda row : row[0]).collect()
							label_df = df_val.groupby("userId").agg(f.collect_set("movieId").alias("movieIds"))
							label_lst = label_df.select('movieIds').rdd.map(lambda row : row[0]).collect()
			 
							predictionAndlabel = [[pre_lst[i], label_lst[i]] for i in range(len(label_lst))]

							rdd = sc.parallelize(predictionAndlabel)
							metrics = RankingMetrics(rdd)

							precision_at_k = metrics.precisionAt(k)
							mean_precision = metrics.meanAveragePrecision
							ndcg_at_k = metrics.ndcgAt(k)
							print("Rank={}|Reg={}|MaxIter={}|Num of recommended movies={}--->PrecisionAt{}={}".format(rank, reg, iter, k, k, precision_at_k))
							print("Rank={}|Reg={}|MaxIter={}|Num of recommended movies={}--->Mean Precision={}".format(rank, reg, iter, k, mean_precision))
							print("Rank={}|Reg={}|MaxIter={}|Num of recommended movies={}--->NDCG at {}={}".format(rank, reg, iter, k, k, ndcg_at_k))


							if precision_at_k > max_precision:
								max_precision = precision_at_k
								best_rank = rank
								best_regularization = reg
								best_maxIter = iter
								best_k = k 
								best_model = train_model
            

	print("The best model has {} maxIter, {} latent factors and " "regularization = {}".format(best_maxIter, best_rank, best_regularization))
	print("The optimal number of recommended movies is: {}".format(best_k))
	return best_model, best_k


def main():
	#initialize params
	ratings_path = "ratings.csv"

	#initialize spark 
	spark = SparkSession.builder.appName("part1").getOrCreate()

	#load dataset and initialize ALS recommender system
	ratings_df = Dataset(spark, ratings_path).DF
	als = ALSModel(spark, ratings_df)

	#tune the als model
	ranks = [10,20]
	regParams = [.01, 0.05]
	maxIter = [10, 20]
	K = [100]

	als.tune_model(regParams, ranks, maxIter, K)

if __name__ == "__main__":
	main()
