import pandas as pd
import pymongo
from glob import glob
from pathlib import Path


def mongo_connect():
    client = pymongo.MongoClient('mongodb+srv://tuanlinh:tuanlinh123@bigdatatuanlinh.paxp2cq.mongodb.net')
    db = client['bigdatatuanlinh']
    return db

def read_csv(folder_name):
    files = glob(f'dataset/{folder_name}/*.csv')
    final_df = []
    for file in files:
        file_name = Path(file).stem
        region = file_name.split('_')[0]
        year = file_name.split('_')[-1]
        df = pd.read_csv(file)
        df['region'] = region
        df['year'] = year
        final_df.append(df)
    return pd.concat(final_df)


if __name__ == '__main__':
    mongo_db = mongo_connect()

    # Task 1:
    df_electricity = read_csv('Electricity')
    # df_electricity.to_csv('df_electricity.csv', index=False)
    # electricity_data_list = df_electricity.to_dict(orient='records')
    # mongo_db['electricity'].insert_many(electricity_data_list)
    #
    df_gas = read_csv('Gas')
    df_gas.to_csv('df_gas.csv', index=False)
    # for index, row in df_gas.iterrows():
    #     data_dict = row.to_dict()
    #     mongo_db['gas'].insert_one(data_dict)

