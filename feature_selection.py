def obtain_data(df, h, cols_to_select, cols_delay, n_samples):
    df_ = df.loc[df[cols_delay].shift(-h).fillna(0) != 0]

    df_delay = df_[df_['y_clas'] == 1]

    idx_delay_repeated = np.repeat(df_delay.index, (n_samples // len(df_delay)) + 1)

    idx_non_delay = np.random.choice(data[data['y_clas'] == 0].index, n_samples // 2, replace=False)
    idx_delay = np.random.choice(idx_delay_repeated, n_samples // 2, replace=False)
    idx = np.concatenate([idx_non_delay, idx_delay])

    X = df_.loc[idx, cols_to_select].values
    y = df_.loc[idx, 'y_clas'].values

    scaler = MinMaxScaler()
    scaler.fit_transform(X)

    return X, y


def make_rf_test(df, n_trees, cols):
    X_train, y_train = obtain_data(df, h)
    model_rf = RandomForestClassifier(random_state=0,
                                      criterion='gini',
                                      n_estimators=n_trees,
                                      max_depth=5,
                                      min_samples_leaf=100,
                                      max_features='auto').fit(X_train, y_train)

    return model_rf.feature_importances


def make_lr_test(df, alpha, cols):
    X_train, y_train = obtain_data(df, cols)
    model_lr = LogisticRegression(random_state=0,
                                  solver='liblinear',
                                  penalty='l1',
                                  C=1-alpha).fit(X_train, y_train)

    return model_lr.coef


def get_feature_importance(df, type_selector, cols, perm_cols, alt_cols):
    # 100 more relevant features
    feat_imp = make_rf_test(df, 15, cols) if type_selector == 1 else make_lr_test(df, .1, cols)

    del_idx = [cols.index(c) for c in alt_cols]
    relevant_feat_idx = np.argsort(feat_imp[del_idx])[0:100]

    feats_considered = alt_cols[relevant_feat_idx]
    cols_ = perm_cols + list(feats_considered)

    df = df.loc[:, cols_]
    cols_.remove('y_clas')

    # 10 more relevant features
    feat_imp_arr = np.zeros((len(cols_), 10))
    for j in range(10):
        feat_imp_arr[:, i] = make_rf_test(df, 100, cols_) if type_selector == 1 else make_lr_test(df, .1, cols_)

    feat_imp_avg = np.mean(feat_imp_arr, axis=1)

    del_idx_ = [cols_.index(n) for n in feats_considered]
    relevant_feat_idx_ = np.argsort(feat_imp_avg[del_idx_])[0:10]

    feats_considered_ = np.array(feats_considered)[relevant_feat_idx_]
    cols_considered_ = perm_cols + list(feats_considered_)

    return cols_considered_, feat_imp_avg[[cols_.index(n) for n in cols_considered_]]


N_SAMPLES = 3000
COL = 'OD_PAIR'

cols_data = list(df_data.columns) + list(df_med_node_delays.columns) + list(df_med_od_delays.columns)
cols_data.remove('y')
cols_data.remove('OD_PAIR')

nodes_ods = np.array(list(od_pairs) + list(df_med_node_delays.columns))

PERM_COLS = ['HOUR',
             'DAY',
             'DAY_OF_WEEK',
             'MONTH', 
             'QUARTER',
             'SEASON',
             'MEAN_DELAY',
             'y_clas']

ALT_COLS = []

entities = []

features = {}
for i, entity in enumerate(entities):
    print("Entity: {}".format(entity), end="")

    df_ = df[df[COL] == entity].reset_index(drop=True)
    df_.drop(columns=[COL], inplace=True)
    df_ = pd.concat([df_od, df_med_od_delays, df_med_node_delays], axis=1)

    cols, cols_imp = get_feature_importance(df_)
    features[entity] = {"cols": cols, "imp": cols_imp}