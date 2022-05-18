#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lenskit ALS model.
"""

#!pip install lenskit

import pandas as pd
import lenskit.crossfold as xf
import time

from lenskit.algorithms.als import BiasedMF
from contextlib import closing
from lenskit import Recommender
from lenskit.batch import train_isolated, predict
from lenskit.metrics.predict import user_metric, rmse, global_metric


class DataLoader():
  """
  Load and split data.
  """
  def __init__(self, filepath):
    self.filepath = filepath

  def load_data(self):
    "Return a pandas dataframe of original ratings."
    ratings = pd.read_csv(self.filepath)
    self.ratings = ratings.rename(columns={'userId': 'user', 'movieId': 'item'})
    return self
  
  def split(self):
    """
    Return splitted datasets based on user_id.
    """
    train_data = []
    tmp_data = []
    for train, tmp in xf.partition_users(self.ratings[['user', 'item', 'rating']], 1, xf.SampleFrac(0.5)):
      tmp_data.append(tmp)
      train_data.append(train)

    test_data = []
    val_data = []
    for test, val in xf.partition_users(tmp_data[0], 1, xf.SampleFrac(0.5)):
      test_data.append(test)
      val_data.append(val)
    return train_data[0], val_data[0], test_data[0]


def isolated_training(algo, train_ratings, test_ratings):
  """
  Execute isolated training.
  Params:
    algo - the algorithm to train
    train_ratings 
    test_ratings 
  """
  algo = Recommender.adapt(algo)
  with closing(train_isolated(algo, train_ratings)) as algo:
    preds = predict(algo, test_ratings)
  return preds

def evaluator_rmse(ranks, regs):
  best_precision = float("inf")
  best_rank = 0
  best_reg = 0
  best_model = None

  #load and split data
  train_ratings, val_ratings, test_ratings = DataLoader("ratings.csv").load_data().split()

  for rank in ranks:
    for reg in regs:
      algo = BiasedMF(features=rank,reg=reg)
      preds = isolated_training(algo, train_ratings, val_ratings)
      precision = global_metric(preds, metric=rmse)
      
      print("The rmse on val set when rank={} and reg={} is: {}".format(rank, reg, precision))
      
      if precision < best_precision:
          best_precision = precision
          best_rank = rank
          best_reg = reg

  #evaluate on testing data
  algo = BiasedMF(features=best_rank,reg=best_reg)
  preds_test = isolated_training(algo, train_ratings, test_ratings)
  precision_test = global_metric(preds_test, metric=rmse) 
  print("The best combination of hyperparameters is: reg={}|rank={}".format(best_reg, best_rank))
  print("The global RMSE on testing set is: {}".format(precision_test))


def main():
  #hyperparameters
  ranks = [10, 20, 50]
  regs = [0.01, 0.05, 0.1]

  #evaluating
  tic = time.perf_counter()
  evaluator_rmse(ranks, regs)
  toc = time.perf_counter()
  print(f"Training and evaluating in {toc - tic:0.4f} seconds!")

  
if __name__ == "__main__":
	main()
