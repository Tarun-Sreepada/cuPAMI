import pandas as pd
import cupy as cp
import cudf

def csv_to_parquet(csv_path, parquet_path, sep):
    file = []
    with open(csv_path, 'r') as f:
        file = f.readlines()
        
    file = [line.strip().split(sep) for line in file]
    
    max_len = max([len(line) for line in file])
    
    df = pd.DataFrame(file, columns=[f'col_{i}' for i in range(max_len)])
    
    df.to_parquet(parquet_path)
    
    cudf_df = cudf.read_parquet(parquet_path)
    print(cudf_df.head())
    
    
csv_to_parquet("/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.csv", "/home/tarun/cuPAMI/datasets/Transactional_T10I4D100K.parquet", '\t')