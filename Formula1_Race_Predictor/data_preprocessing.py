import time
import random
import numpy as np

from datetime import datetime
from sklearn import preprocessing
from sklearn.utils import class_weight

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
    
    # Sort for RNN
    df.sort_values(['year', 'round', 'final_position'], ascending=[True,True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Final cleanup
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    # Choose if want to predict all 34 position classes, or podium or points positions, etc (Comment out to predict all 34 positions)
    df['final_position'] = np.where(df['final_position'] > 10, 11, df['final_position'])
    
    # Save preprocessed dataset (commented out after initial save)
    #df.to_csv("../Datasets/Formed/preprocessed.csv")
    
    # Obtain balanced class weights for training
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(df['final_position'].values),
                                                     df['final_position'].values)
  
    # Return normalized values & class weigths
    return scaler.fit_transform(df.values), class_weights

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

def generate_rnn_data(values):

    # Use last 10% of data for RNN validation/test
    train_interval = round(values.shape[0] * 0.9)
    train_x, train_y = values[:train_interval, :-1], values[:train_interval, -1]
    val_x, val_y = values[train_interval:, :-1], values[train_interval:, -1]

    # Reshape to 3D input for RNN
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
    val_x = val_x.reshape(val_x.shape[0], 1, val_x.shape[1])    

    return train_x, train_y, val_x, val_y