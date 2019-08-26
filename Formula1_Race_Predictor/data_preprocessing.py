import time
import random
import numpy as np

from datetime import datetime
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

def preprocess_df(df):

    # Iterate over columns removing/replacing NaN values
    for col in df.columns:

        # 34 will be the NaN class for final position
        if col == 'final_position':
            df['final_position'].fillna(34, inplace=True)

        # Replace all empty driver data with mean from that samples round & final position
        elif col == 'driver_season_points':
            df[col] = df[col].fillna(round(df.groupby(['final_position', 'round'])[col].transform('mean')))

        elif col == 'driver_season_position':
            df[col] = df[col].fillna(round(df.groupby(['final_position', 'round'])[col].transform('mean')))

        elif col == 'driver_season_wins':
            df[col] = df[col].fillna(round(df.groupby(['final_position', 'round'])[col].transform('mean')))

        # Transform text into integer classes
        elif col == 'driver_nationality' or col == 'constructor_nationality':
            df[col] = encoder.fit_transform(df[col])

        # Replace all empty constructor data with operations on driver data for that race
        elif col == 'constructor_season_points':
            df[col] = df[col].fillna(df.groupby(['year', 'round', 'constructor_id'])['driver_season_points'].transform('sum'))

        elif col == 'constructor_season_position':
            df[col] = df[col].fillna(df.groupby(['year', 'round', 'constructor_id'])['driver_season_position'].transform('min'))

        elif col == 'constructor_season_wins':
            df[col] = df[col].fillna(df.groupby(['year', 'round', 'constructor_id'])['driver_season_wins'].transform('sum'))

        elif col == 'q1_time':
            q1_ms = []
            q2_ms = []
            q3_ms = []

            df[col].fillna('00:00.00', inplace=True)
            df['q2_time'].fillna('00:00.00', inplace=True)
            df['q3_time'].fillna('00:00.00', inplace=True)

            # Converting time stamps into milleseconds
            for q1, q2, q3 in zip(df.q1_time, df.q2_time, df.q3_time):
                q1_obj = datetime.strptime(q1, '%M:%S.%f')
                q2_obj = datetime.strptime(q2, '%M:%S.%f')
                q3_obj = datetime.strptime(q3, '%M:%S.%f')

                q1_ms.append((q1_obj.minute * 60000) + (q1_obj.second * 1000) + (q1_obj.microsecond // 1000))
                q2_ms.append((q2_obj.minute * 60000) + (q2_obj.second * 1000) + (q2_obj.microsecond // 1000))
                q3_ms.append((q3_obj.minute * 60000) + (q3_obj.second * 1000) + (q3_obj.microsecond // 1000))

            df[col] = q1_ms
            df['q2_time'] = q2_ms
            df['q3_time'] = q3_ms

            # Replace all empty quali times with average quali time from that samples round & grid position
            df[col].replace({0 : np.nan}, inplace=True)
            df['q2_time'].replace({0 : np.nan}, inplace=True)
            df['q3_time'].replace({0 : np.nan}, inplace=True)
            df[col] = df[col].fillna(df.groupby(['grid_position', 'round'])[col].transform('mean'))

            # Replace missing q2 and q3 times with previous quali time
            df['q2_time'].fillna(df['q1_time'], inplace=True)
            df['q3_time'].fillna(df['q2_time'], inplace=True)

        else:
            pass

    # Move final position column to right end
    cols = list(df)
    cols.insert(21, cols.pop(cols.index('final_position')))
    df = df.loc[:, cols]
    
    # Final cleanup & save preprocessed df
    df.dropna(inplace=True)
    # df.to_csv("../Datasets/Formed/preprocessed.csv")        # Commented out after initial save

    # Return normalized values
    return scaler.fit_transform(df.values)

def generate_dnn_data(values):

    # Split intervals
    train_interval = round(values.shape[0] * 0.7)
    val_interval = round(values.shape[0] * 0.85)

    # Shuffle then split
    random.shuffle(values)
    train_x, train_y = values[:train_interval, :-1], values[:train_interval, -1]
    val_x, val_y = values[train_interval:val_interval, :-1], values[train_interval:val_interval, -1]
    test_x, test_y = values[val_interval:, :-1], values[val_interval:, -1]

    return train_x, train_y, val_x, val_y, test_x, test_y
