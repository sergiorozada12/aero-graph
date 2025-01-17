import pandas as pd
import os
import numpy as np
import zipfile
import json

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Columns options
OD_PAIR = 0
NODE = 1

# Feature selection options
LR = 0
RF = 1

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

        print("Resumming column: ", general_col)

        df_merged_shifted = df_merged[[arr_col, dep_col]].shift(shift)

        df_merged[arr_col] = df_merged[arr_col].apply(fill_list_na())
        df_merged[dep_col] = df_merged[dep_col].apply(fill_list_na())

        df_merged_shifted[arr_col] = df_merged_shifted[arr_col].apply(fill_list_na())
        df_merged_shifted[dep_col] = df_merged_shifted[dep_col].apply(fill_list_na())

        df_merged['DEP_' + general_col] = df_merged[dep_col] + df_merged_shifted[dep_col]
        df_merged['ARR_' + general_col] = df_merged[arr_col] + df_merged_shifted[arr_col]

        df_merged[general_col] = df_merged['DEP_' + general_col] + df_merged['ARR_' + general_col]

        df_merged['MEAN_' + general_col] = df_merged[general_col].apply(np.mean)
        df_merged['MEDIAN_' + general_col] = df_merged[general_col].apply(np.median)

        if general_col == 'DELAY':
            df_merged['MEAN_DEP_DELAY'] = df_merged['DEP_DELAY'].apply(np.mean)
            df_merged['MEAN_ARR_DELAY'] = df_merged['ARR_DELAY'].apply(np.mean)

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
    df['FL_DATE'] = np.repeat(dates, hours.shape[0] * elements.shape[0])
    df['HOUR'] = np.tile(np.repeat(hours, elements.shape[0]), dates.shape[0])
    df[col] = np.tile(elements, dates.shape[0] * hours.shape[0])

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
    edges = []

    for i, od_i in enumerate(od_pairs):
        for j, od_j in enumerate(od_pairs):
            if od_i.split('_')[0] == od_j.split('_')[0] or od_i.split('_')[0] == od_j.split('_')[1] or \
                    od_i.split('_')[1] == od_j.split('_')[0] or od_i.split('_')[1] == od_j.split('_')[1]:
                adj[i, j] = 1
                edges.append([od_i, od_j])

    degree = np.diag((adj.T@adj).diagonal())
    laplacian = degree - adj

    np.save(path + "graph/Adj_od_pairs", adj)
    np.save(path + "graph/Lap_Ood_pairs", laplacian)
    np.save(path + "graph/edges_od_pairs", np.array(edges))


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
    np.save(path + "graph/edges_nodes", edges.values)



########################################################
########    UTILS FOR FEATURE ENGINEERING   ############
########################################################

def get_features_df(df_ods, df_nodes, od_pairs, nodes):
    avg_delays_od_dep = df_ods['MEAN_DEP_DELAY'].values.reshape(-1, od_pairs.shape[0]).astype(np.float32)
    avg_delays_od_arr = df_ods['MEAN_ARR_DELAY'].values.reshape(-1, od_pairs.shape[0]).astype(np.float32)

    cols_dep_od = [od + '_DEP' for od in od_pairs]
    cols_arr_od = [od + '_ARR' for od in od_pairs]

    df_od_avg_delays = pd.concat([pd.DataFrame(avg_delays_od_dep, columns=cols_dep_od),
                                  pd.DataFrame(avg_delays_od_arr, columns=cols_arr_od)], axis=1)

    avg_delays_nodes_dep = df_nodes['MEAN_DEP_DELAY'].values.reshape(-1, nodes.shape[0]).astype(np.float32)
    avg_delays_nodes_arr = df_nodes['MEAN_ARR_DELAY'].values.reshape(-1, nodes.shape[0]).astype(np.float32)

    cols_dep_node = [n + '_DEP' for n in nodes]
    cols_arr_node = [n + '_ARR' for n in nodes]

    df_node_avg_delays = pd.concat([pd.DataFrame(avg_delays_nodes_dep, columns=cols_dep_node),
                                    pd.DataFrame(avg_delays_nodes_arr, columns=cols_arr_node)], axis=1)

    return pd.concat([df_od_avg_delays, df_node_avg_delays], axis=1)

def get_delay_types_df(df_ods, df_nodes, d_types, od_pairs, nodes):
    dfs = []
    for d in d_types:
        cols_od = [od + '_' + d for od in od_pairs]

        d_type_df_od = pd.DataFrame(df_ods['MEAN_' + d].values.\
                        reshape(-1, od_pairs.shape[0]).astype(np.float32),
                        columns=cols_od)

        cols_nodes = [n + '_' + d for n in nodes]

        d_type_df_node = pd.DataFrame(df_nodes['MEAN_' + d].values.\
                            reshape(-1, nodes.shape[0]).astype(np.float32),
                            columns=cols_nodes)

        dfs.append(d_type_df_od)
        dfs.append(d_type_df_node)

    return pd.concat(dfs, axis=1)

def apply_clustering(df_, n_clust, n_dates, n_hours, random_state=None):
    df = df_.copy()
    kmeans_h = KMeans(n_clusters=n_clust, random_state=random_state)
    kmeans_h.fit(df.values)
    df['HOUR_CLUSTER'] = kmeans_h.labels_

    kmeans_d = KMeans(n_clusters=n_clust, random_state=random_state)
    kmeans_d.fit(df.values.reshape(n_dates, -1))
    df['DAY_CLUSTER'] = np.repeat(kmeans_d.labels_, n_hours)

    df['DAY_CLUSTER-1'] = df['DAY_CLUSTER'].shift(-n_hours, fill_value=1)

    return df

def treat_delay_type(d_type, df_, elems, col, edges):

    df = df_.copy()
    for el in elems:
        df[el + '_MEAN_' + d_type] = np.repeat(df.loc[df[col] == el, 'MEAN_' + d_type].values, elems.shape[0])
    
    print("Start calculating neighbours")

    neighbors = {el: [neigh + '_MEAN_' + d_type for neigh in edges[np.where(edges[:, 1] == el), 0][0]] for el in elems}

    return df.apply(lambda row: row[neighbors[row[col]]].sum(), axis=1).values


########################################################
########    UTILS FOR FEATURE SELECTION     ############
########################################################


def obtain_data(df, h, cols_to_select, col_delay, n_samples):
    df_ = df.loc[df[col_delay].shift(-h).fillna(0) != 0]

    df_delay = df_[df_['y_clas'] == 1]

    idx_delay_repeated = np.repeat(df_delay.index, (n_samples // len(df_delay)) + 1)

    idx_non_delay = np.random.choice(df_[df_['y_clas'] == 0].index, n_samples // 2, replace=False)
    idx_delay = np.random.choice(idx_delay_repeated, n_samples // 2, replace=False)
    idx = np.concatenate([idx_non_delay, idx_delay])

    X = df_.loc[idx, cols_to_select].values
    y = df_.loc[idx, 'y_clas'].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y


def make_rf_test(df, n_trees, cols_to_select, col_delay, h, n_samples):
    X_train, y_train = obtain_data(df, h=h, cols_to_select=cols_to_select, col_delay=col_delay, n_samples=n_samples)
    model_rf = RandomForestClassifier(random_state=0,
                                      criterion='gini',
                                      n_estimators=n_trees,
                                      max_depth=5,
                                      min_samples_leaf=100,
                                      max_features='auto').fit(X_train, y_train)

    return model_rf.feature_importances_


def make_lr_test(df, alpha, cols_to_select, col_delay, h, n_samples):
    X_train, y_train = obtain_data(df, h=h, cols_to_select=cols_to_select, col_delay=col_delay, n_samples=n_samples)
    model_lr = LogisticRegression(random_state=0,
                                  solver='liblinear',
                                  penalty='l1',
                                  C=1-alpha).fit(X_train, y_train)

    return np.reshape(np.absolute(model_lr.coef_), -1)


def get_feature_importance(df, type_selector, perm_cols, alt_cols, col_delay, h, n_samples):
    cols_to_select = perm_cols + alt_cols
    cols_to_select.remove('y_clas')

    feat_imp_arr = np.zeros((len(cols_to_select), 10))
    for j in range(10):
        feat_imp_arr[:, j] = make_rf_test(df, 100, cols_to_select, col_delay, h, n_samples) if type_selector == RF \
              else make_lr_test(df, .1, cols_to_select, col_delay, h, n_samples)

    feat_imp_avg = np.mean(feat_imp_arr, axis=1)
    idx = np.argsort(feat_imp_avg)[::-1]

    return np.array(cols_to_select)[idx].tolist(), feat_imp_avg[idx].tolist()


def feature_selection(df, entities, avg_delays, perm_cols, alt_cols, n_samples, col, out_path):

    features_lr = {}
    features_rf = {}
    for entity in entities:
        print("Entity: {} - ".format(entity), end="")

        df_ent = df[df[col] == entity].reset_index(drop=True)
        df_ent.drop(columns=[col], inplace=True)
        df_ent = pd.concat([df_ent, avg_delays], axis=1)

        # LOGISTIC REGRESSION
        cols, cols_imp = get_feature_importance(df_ent, LR, perm_cols, alt_cols, 'MEAN_DELAY', 2, n_samples)
        features_lr[entity] = {"cols": cols, "imp": cols_imp}

        # RANDOM FOREST
        cols, cols_imp = get_feature_importance(df_ent, RF, perm_cols, alt_cols, 'MEAN_DELAY', 2, n_samples)
        features_rf[entity] = {"cols": cols, "imp": cols_imp}

        print("DONE - Cols: {}".format(cols[len(perm_cols):]))

    with open(out_path + 'feat_lr_{}.json'.format(col.lower() + 's'), 'w') as fp:
        json.dump(features_lr, fp)

    with open(out_path + 'feat_rf_{}.json'.format(col.lower() + 's'), 'w') as fp:
        json.dump(features_rf, fp)

########################################################
########    UTILS FOR GENERAL EXPERIMENTS   ############
########################################################


def get_metrics_dict(metrics_arr):
    return {'acc': metrics_arr[0],
            'prec': metrics_arr[1],
            'rec': metrics_arr[2],
            'f1': metrics_arr[3]}


def get_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    if y_pred.sum() > 0 and y_test.sum() > 0:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    else:
        precision = 0.
        recall = 0.
        f1 = 0.

    return [accuracy,
            precision,
            recall,
            f1]


def predict_assumption(data, model):
    predictions = np.zeros(len(data.df_test))

    cond_0s = data.df_test[data.col_delay] <= 0

    df_pred = data.df_test.loc[~cond_0s]
    idx_p0 = df_pred.index

    X_test = df_pred.loc[:, data.df_test.columns.drop(data.col_labels)].values

    X_test = data.scale(X_test)

    y_pred = model.predict(X_test)
    predictions[idx_p0] = y_pred

    return predictions


class TrainingData:
    def __init__(self, df_train, df_test, h, col_delay, col_labels):
        self.df_train = df_train
        self.df_test = df_test
        self.h = h
        self.col_delay = col_delay
        self.col_labels = col_labels

        self.scaler = None

    def obtain_training_data(self, n_samples):
        df_ = self.df_train.loc[self.df_train[self.col_delay].shift(-self.h).fillna(0) != 0]

        idx = self.obtain_equal_idx(df_[df_[self.col_labels] == 0].index,
                                    df_[df_[self.col_labels] == 1].index,
                                    n_samples)

        X_train = df_.loc[idx, self.df_train.columns.drop(self.col_labels)].values
        y_train = df_.loc[idx, self.col_labels].values

        X_test = self.df_test.loc[:, self.df_test.columns.drop(self.col_labels)].values
        y_test = self.df_test.loc[:, self.col_labels].values

        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train)

        X_train = self.scale(X_train)
        X_test = self.scale(X_test)

        return X_train, y_train, X_test, y_test

    def scale(self, data):
        if self.scaler == None:
            raise RuntimeError("Scaler undefined")
        return self.scaler.transform(data)

    def obtain_equal_idx(self, idx_0, idx_1, n_samples):
        idx_1_repeated = np.repeat(idx_1, (n_samples // len(idx_1)) + 1)

        idx_non_delay = np.random.choice(idx_0, n_samples // 2, replace=False)
        idx_delay = np.random.choice(idx_1_repeated, n_samples // 2, replace=False)
        return np.concatenate([idx_non_delay, idx_delay])