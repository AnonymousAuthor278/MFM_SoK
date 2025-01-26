import pandas as pd
import os

# 读取 Parquet 文件
file_path = './laion400m-meta/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet'
data = pd.read_parquet(file_path)

# 截取前 2,000,000 条数据
subset_data = data.head(2000000)

# 输出到新的 Parquet 文件
os.makedirs('./laion400m-sub-meta', exist_ok=True)
output_file_path = './laion400m-sub-meta/subset-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet'
subset_data.to_parquet(output_file_path, index=False)

print(f'Successfully saved the first 2,000,000 rows to {output_file_path}')
