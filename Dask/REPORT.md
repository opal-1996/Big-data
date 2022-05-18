# Lab 4: Dask

- Name: Qin Yang

- NetID: qy692

## Description of your solution

For the full set, I used cluster, along with bags and dask dataframes to help accelerate computing process.The key to optimization is filtering out those rows we need as early as possible before we do computation task, otherwise we might get stuck on it. For instance, I tried to convert each .dly file into one dask dataframe by using mapping, and then concatenate or append them together into single dataframe and compute on it , but it was too slow. So I turned to computing aggregation for each .dly file as intermediate results and combine them together to compute the final result, but still too slow to process. 
