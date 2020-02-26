import pandas as pd
import os
import numpy as np


def load_df_from_raw_files(path):
    dfs = []

    for dir_name, dirs, files in os.walk(path):
        for d in dirs:
            filename = os.listdir(os.path.join(dir_name, d))[0]
            full_path = os.path.join(dir_name, d, filename)
            df_aux = pd.read_csv(full_path)
            dfs.append(df_aux)
    return pd.concat(dfs)


def cast_df_raw_columns(df_):
    df = df_.copy()
    for c in df.columns:
        if 'float' in str(df[c].dtype):
            df[c] = pd.to_numeric(df[c], downcast='float')
        elif 'int' in str(df[c].dtype):
            df[c] = pd.to_numeric(df[c], downcast='integer')

    return df


def filter_airports(df):
    df['OD_PAIR'] = df['ORIGIN'] + '_' + df['DEST']
    df_odpair = pd.DataFrame(df.groupby(['ORIGIN', 'DEST', 'OD_PAIR'])['ARR_DELAY'].count()).reset_index(drop=False)
    df_odpair.columns = ['ORIGIN', 'DEST', 'OD_PAIR', 'COUNT']

    n_dates = df['FL_DATE'].unique().shape[0]
    df_valid_odpairs = df_odpair.loc[df_odpair['COUNT'] >= 10 * n_dates]
    valid_odpairs = df_valid_odpairs['OD_PAIR'].tolist()
    airports = (pd.concat([gb_cons['ORIGIN'], gb_cons['DEST']])).unique().tolist()

    print("Number of days in the analysis: ", n_dates)
    print("Original number of airports: ", pd.concat([df['ORIGIN'], df['DEST']]).unique().shape[0])
    print("Number of airports after filtering: ", len(airports))
    print("Orignal number of OD pairs: ", df['OD_PAIR'].unique().shape[0])
    print("Number of OD pairs after filtering", len(valid_odpairs))

    return df[df['OD_PAIR'].isin(valid_odpairs)].reset_index(drop=True), valid_odpairs


def get_season(month):
    if month in range(9,12):
        return 1                # September - November -- Low delays
    elif month in range(1, 6):
        return 2                # January - May -- Medium delays
    elif month in range(6, 9) or month == 12:
        return 3                # June - August or December -- High delays
    return month


def get_time_vars(df_, dates, hours, od_pairs):
    df = pd.DataFrame()
    df['FL_DATE'] = np.repeat(dates, hours.shape[0] * od_pairs.shape[0])
    df['HOUR'] = np.tile(np.repeat(hours, od_pairs.shape[0]), dates.shape[0])
    df['OD_PAIR'] = np.tile(od_pairs, dates.shape[0] * hours.shape[0])

    df = df.merge(df_, how='left', on=['FL_DATE', 'HOUR', 'OD_PAIR'])

    df['DAY'] = df['FL_DATE'].apply(lambda d: d.day)
    df['DAY_OF_WEEK'] = df['FL_DATE'].apply(lambda d: d.dayofweek)
    df['MONTH'] = df['FL_DATE'].apply(lambda d: d.month)
    df['QUARTER'] = df['FL_DATE'].apply(lambda d: d.quarter)
    df['YEAR'] = df['FL_DATE'].apply(lambda d: d.year)
    df['SEASON'] = df['MONTH'].apply(get_season)

    return df


def get_label(df, th, h, od_pairs)
    df['y_reg'] = df['MEDIAN_DEP_DELAY'].fillna(v).shift(-h*od_pairs.shape[0]).fillna(-1)
    df['y_clas'] = 1*(df_shifted.values >= th)

    return df
