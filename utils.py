import pandas as pd
import os
import numpy as np
import zipfile

OD_PAIR = 0
NODE = 1


def load_df_from_raw_files(path):
    df = pd.DataFrame()

    for file in os.listdir(path):
        zfile = zipfile.ZipFile(path + file)
        df = pd.concat([df, pd.read_csv(zfile.open(zfile.infolist()[0].filename))], sort=True)

    cols_delay_type = ['CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']
    df[cols_delay_type] = df[cols_delay_type].fillna(0.0)

    print("Data loaded, shape: ", df.shape)
    print("----------------------------------------")

    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    return df.drop(columns=unnamed_cols)


def cast_df_raw_columns(df_):
    df = df_.copy()
    for c in df.columns:
        if 'float' in str(df[c].dtype):
            df[c] = pd.to_numeric(df[c], downcast='float')
        elif 'int' in str(df[c].dtype):
            df[c] = pd.to_numeric(df[c], downcast='integer')

    return df


def filter_od_pairs(df_):
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
    print("----------------------------------------")

    return df[df['OD_PAIR'].isin(valid_odpairs)].reset_index(drop=True), \
           np.array(sorted(valid_odpairs)), \
           np.array(sorted(airports))


def filter_airports(df_):
    df = df_.copy()

    df_dest = pd.DataFrame(df.groupby(['DEST'])['ORIGIN'].count()).reset_index(drop=False)
    df_origin = pd.DataFrame(df.groupby(['ORIGIN'])['DEST'].count()).reset_index(drop=False)

    df_origin.columns = ['ORIGIN', 'COUNT']
    df_dest.columns = ['DEST', 'COUNT']

    n_dates = df['FL_DATE'].unique().shape[0]
    airports = []

    for airport in df['ORIGIN'].unique():
        total = df_origin[df_origin['ORIGIN'] == airport]['COUNT'] + df_dest[df_dest['DEST'] == airport]['COUNT']
        if total.values[0]/n_dates >= 10:
            airports.append(airport)

    print("Number of days in the analysis: ", n_dates)
    print("Original number of airports: ", df['ORIGIN'].unique().shape[0])
    print("Number of airports after filtering: ", len(airports))
    print("----------------------------------------")

    return df[df['ORIGIN'].isin(airports) & df['DEST'].isin(airports)].dropna().reset_index(drop=True), \
           np.array(sorted(airports))


def fill_list_na():
    return lambda x: x if type(x) == list else []


def merge_and_group(df_, problem_type):
    df = df_.copy()

    dep_del_cols = ['DEP_DELAY', 'CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']
    arr_del_cols = ['ARR_DELAY', 'CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']

    dep_del_cols_ = ['DEP_' + col if 'DEP_' not in col else col for col in dep_del_cols]
    arr_del_cols_ = ['ARR_' + col if 'ARR_' not in col else col for col in arr_del_cols]

    if problem_type == OD_PAIR:
        df_dep = pd.DataFrame(df.groupby(['FL_DATE', 'CRS_DEP_HOUR', 'OD_PAIR'])[dep_del_cols].agg(list)).reset_index()
        df_arr = pd.DataFrame(df.groupby(['FL_DATE', 'CRS_ARR_HOUR', 'OD_PAIR'])[arr_del_cols].agg(list)).reset_index()

        df_dep.columns = ['FL_DATE', 'HOUR', 'OD_PAIR'] + dep_del_cols_
        df_arr.columns = ['FL_DATE', 'HOUR', 'OD_PAIR'] + arr_del_cols_

        df_merged = df_dep.merge(df_arr, how='outer', on=['FL_DATE', 'HOUR', 'OD_PAIR'])

    else:
        df_dep = pd.DataFrame(df.groupby(['FL_DATE', 'CRS_DEP_HOUR', 'ORIGIN'])[dep_del_cols].agg(list)).reset_index()
        df_arr = pd.DataFrame(df.groupby(['FL_DATE', 'CRS_ARR_HOUR', 'DEST'])[arr_del_cols].agg(list)).reset_index()

        df_dep.columns = ['FL_DATE', 'HOUR', 'NODE'] + dep_del_cols_
        df_arr.columns = ['FL_DATE', 'HOUR', 'NODE'] + arr_del_cols_

        df_merged = df_dep.merge(df_arr, how='outer', on=['FL_DATE', 'HOUR', 'NODE'])

    print("Merged and grouped")
    print("----------------------------------------")

    return df_merged


def obtain_avg_delay(df, shift):
    df_merged = df.copy()

    dep_del_cols = ['DEP_DELAY', 'CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']
    arr_del_cols = ['ARR_DELAY', 'CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']

    dep_del_cols_ = ['DEP_' + col if 'DEP_' not in col else col for col in dep_del_cols]
    arr_del_cols_ = ['ARR_' + col if 'ARR_' not in col else col for col in arr_del_cols]

    for i in range(len(dep_del_cols_)):
        general_col = 'DELAY' if i == 0 else dep_del_cols[i]
        arr_col = arr_del_cols_[i]
        dep_col = dep_del_cols_[i]

        print(general_col)

        df_merged_shifted = df_merged[[arr_col, dep_col]].shift(shift)

        df_merged[arr_col] = df_merged[arr_col].apply(fill_list_na())
        df_merged[dep_col] = df_merged[dep_col].apply(fill_list_na())

        df_merged_shifted[arr_col] = df_merged_shifted[arr_col].apply(fill_list_na())
        df_merged_shifted[dep_col] = df_merged_shifted[dep_col].apply(fill_list_na())

        df_merged[general_col] = df_merged[arr_col] + \
                                 df_merged[dep_col] + \
                                 df_merged_shifted[arr_col] + \
                                 df_merged_shifted[dep_col]
        df_merged['MEAN_' + general_col] = df_merged[general_col].apply(np.mean)

    df_merged['FL_DATE'] = pd.to_datetime(df_merged['FL_DATE'])

    print("Mean delays estimated")
    print("----------------------------------------")

    return df_merged


def get_season(month):
    if month in range(9,12):
        return 1                # September - November -- Low delays
    elif month in range(1, 6):
        return 2                # January - May -- Medium delays
    elif month in range(6, 9) or month == 12:
        return 3                # June - August or December -- High delays
    return month


def get_time_vars(df_, dates, hours, elements, col):
    df = pd.DataFrame()
    df['FL_DATE'] = np.repeat(dates, hours.shape[0] * elems.shape[0])
    df['HOUR'] = np.tile(np.repeat(hours, elems.shape[0]), dates.shape[0])
    df[col] = np.tile(elems, dates.shape[0] * hours.shape[0])

    df = df.merge(df_, how='left', on=['FL_DATE', 'HOUR', col])

    df['DAY'] = df['FL_DATE'].apply(lambda d: d.day)
    df['DAY_OF_WEEK'] = df['FL_DATE'].apply(lambda d: d.dayofweek)
    df['MONTH'] = df['FL_DATE'].apply(lambda d: d.month)
    df['QUARTER'] = df['FL_DATE'].apply(lambda d: d.quarter)
    df['YEAR'] = df['FL_DATE'].apply(lambda d: d.year)
    df['SEASON'] = df['MONTH'].apply(get_season)

    return df


def get_label(df, th, h, length):
    df['y_reg'] = df['MEAN_DELAY'].fillna(0.0).shift(-h*length).fillna(-1)
    df['y_clas'] = 1*(df['y_reg'].values >= th)

    return df


def get_hour(df_):
    df = df_.copy()
    df['CRS_ARR_HOUR'] = df['CRS_ARR_TIME'].apply(lambda x: int(x // 100)).apply(lambda x: 0 if x == 24 else x)
    df['CRS_DEP_HOUR'] = df['CRS_DEP_TIME'].apply(lambda x: int(x // 100)).apply(lambda x: 0 if x == 24 else x)

    df['ARR_HOUR'] = df['ARR_TIME'].apply(lambda x: int(x // 100)).apply(lambda x: 0 if x == 24 else x)
    df['DEP_HOUR'] = df['DEP_TIME'].apply(lambda x: int(x // 100)).apply(lambda x: 0 if x == 24 else x)

    return df


def create_od_pair_graph(od_pairs, path):

    adj = np.zeros([od_pairs.shape[0], od_pairs.shape[0]])

    for i, od_i in enumerate(od_pairs):
        for j, od_j in enumerate(od_pairs):
            if od_i.split('_')[0] == od_j.split('_')[0] | od_i.split('_')[0] == od_j.split('_')[1] | \
                    od_i.split('_')[1] == od_j.split('_')[0] | od_i.split('_')[1] == od_j.split('_')[1]:
                adj[i, j] = 1

    degree = np.diag((adj.T@adj).diagonal())
    laplacian = degree - adj

    np.save(path + "graph/Adj_OD", adj)
    np.save(path + "graph/Lap_OD", laplacian)


def create_airport_graph(df_, nodes, path):
    edges = df_[['ORIGIN', 'DEST']].drop_duplicates()
    weights = df_.groupby(['ORIGIN', 'DEST']).count()

    adj = np.zeros([nodes.shape[0], nodes.shape[0]])
    adj_weighted = np.zeros([nodes.shape[0], nodes.shape[0]])

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if any((edges['ORIGIN'] == node_i) & (edges['DEST'] == node_j)):
                adj[i, j] = 1
                adj_weighted[i, j] = weights.loc[node_i, node_j][0]

    degree = np.diag((adj.T@adj).diagonal())
    laplacian = degree - adj

    np.save(path + "graph/Adj_nodes", adj)
    np.save(path + "graph/Adj_w_nodes", adj_weighted)
    np.save(path + "graph/Lap_nodes", laplacian)

