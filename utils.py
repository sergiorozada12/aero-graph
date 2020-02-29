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


def filter_airports(df_):
    df = df_.copy()
    df['OD_PAIR'] = df['ORIGIN'] + '_' + df['DEST']
    df_odpair = pd.DataFrame(df.groupby(['ORIGIN', 'DEST', 'OD_PAIR'])['ARR_DELAY'].count()).reset_index(drop=False)
    df_odpair.columns = ['ORIGIN', 'DEST', 'OD_PAIR', 'COUNT']

    n_dates = df['FL_DATE'].unique().shape[0]
    df_valid_odpairs = df_odpair.loc[df_odpair['COUNT'] >= 10 * n_dates]
    valid_odpairs = df_valid_odpairs['OD_PAIR'].tolist()
    airports = (pd.concat([df_valid_odpairs['ORIGIN'], df_valid_odpairs['DEST']])).unique().tolist()

    print("Number of days in the analysis: ", n_dates)
    print("Original number of airports: ", pd.concat([df['ORIGIN'], df['DEST']]).unique().shape[0])
    print("Number of airports after filtering: ", len(airports))
    print("Orignal number of OD pairs: ", df['OD_PAIR'].unique().shape[0])
    print("Number of OD pairs after filtering", len(valid_odpairs))

    return df[df['OD_PAIR'].isin(valid_odpairs)].reset_index(drop=True), \
           np.array(sorted(valid_odpairs)), \
           np.array(sorted(airports))

def fill_list_na():
    return lambda x: x if type(x) == list else []

def merge_and_group(df_, shift, problem_type):
    df = df_.copy()

    col = 'NODE'
    if problem_type == 0:
        col = 'OD_PAIR'

    df_dep = pd.DataFrame(df.groupby(['FL_DATE', 'CRS_DEP_HOUR', col])['DEP_DELAY'].agg(list)).reset_index()
    df_arr = pd.DataFrame(df.groupby(['FL_DATE', 'CRS_ARR_HOUR', col])['ARR_DELAY'].agg(list)).reset_index()

    df_dep.columns = ['FL_DATE', 'HOUR', col, 'DEP_DELAY']
    df_arr.columns = ['FL_DATE', 'HOUR', col, 'ARR_DELAY']

    df_merged = df_dep.merge(df_arr, how='outer', on=['FL_DATE', 'HOUR', col])

    df_merged['ARR_DELAY'] = df_merged['ARR_DELAY'].apply(fill_list_na())
    df_merged['DEP_DELAY'] = df_merged['DEP_DELAY'].apply(fill_list_na())

    arr_delay_tm1 = df_merged['ARR_DELAY'].shift(shift).apply(fill_list_na())
    dep_delay_tm1 = df_merged['DEP_DELAY'].shift(shift).apply(fill_list_na())

    df_merged['ARR_DELAY'] = df_merged['ARR_DELAY'] + arr_delay_tm1
    df_merged['DEP_DELAY'] = df_merged['DEP_DELAY'] + dep_delay_tm1

    df_merged['MEAN_ARR_DELAY'] = df_merged['ARR_DELAY'].apply(np.mean)
    df_merged['MEAN_DEP_DELAY'] = df_merged["DEP_DELAY"].apply(np.mean)

    df_merged['FL_DATE'] = pd.to_datetime(df_merged['FL_DATE'])

    return df


def get_season(month):
    if month in range(9,12):
        return 1                # September - November -- Low delays
    elif month in range(1, 6):
        return 2                # January - May -- Medium delays
    elif month in range(6, 9) or month == 12:
        return 3                # June - August or December -- High delays
    return month


def get_time_vars_od(df_, dates, hours, od_pairs):
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


def get_time_vars_node(df_, dates, hours, nodes):
    df = pd.DataFrame()
    df['FL_DATE'] = np.repeat(dates, hours.shape[0] * nodes.shape[0])
    df['HOUR'] = np.tile(np.repeat(hours, nodes.shape[0]), dates.shape[0])
    df['NODE'] = np.tile(nodes, dates.shape[0] * hours.shape[0])

    df = df.merge(df_, how='left', on=['FL_DATE', 'HOUR', 'NODE'])

    return df


def get_label(df, th, h, od_pairs)
    df['y_reg'] = df['MEDIAN_DEP_DELAY'].fillna(v).shift(-h*od_pairs.shape[0]).fillna(-1)
    df['y_clas'] = 1*(df_shifted.values >= th)

    return df

def get_hour(df_):
    df = df_.copy()
    df['CRS_ARR_HOUR'] = df['CRS_ARR_TIME'].apply(lambda x: int(x // 100)).apply(lambda x: 0 if x == 24 else x)
    df['CRS_DEP_HOUR'] = df['CRS_DEP_TIME'].apply(lambda x: int(x // 100)).apply(lambda x: 0 if x == 24 else x)

    df['ARR_HOUR'] = df['ARR_TIME'].apply(lambda x: int(x // 100)).apply(lambda x: 0 if x == 24 else x)
    df['DEP_HOUR'] = df['DEP_TIME'].apply(lambda x: int(x // 100)).apply(lambda x: 0 if x == 24 else x)

    return df
