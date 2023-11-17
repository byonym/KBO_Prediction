import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import get_model
import keras
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')
import tensorflow as tf
import random as python_random



def lgb_predict(data_set, j=26, types='train'):
    # 비율지표 및 세이버매트릭스 지표 생성
    train_sab = saber_matrix(data_set, j, types)

    # 학습
    X = train_sab.drop(['WR_NEXT', 'NEXT_WLS'], axis=1)
    y = train_sab.WR_NEXT

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=74, test_size=0.1)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    joblib.dump(scaler, './data/result/std_scaler.bin', compress=True)

    d_train = lgb.Dataset(X_train_std, label=y_train)
    d_test = lgb.Dataset(X_test_std, label=y_test)

    params = {'objective' : 'regression',
                            'learning_rate' : 0.02,
                           'boosting_type' : 'gbdt',
                            'metric': 'rmse',
                            'is_training_metric' : True,
                            'feature_fraction' : 0.5,
                            'bagging_fraction' : 0.7,
                                'bagging_freq' : 5,
                           'eval_metric' : 'l1',
                           'num_leaves' : 40,
                           'max_depth' : 30,
                           'num_iterations' :3000,
                            'min_gain_to_split ':10,
          'num_boost_round ':1000,
          'bagging_seed':26}

    model = lgb.train(params, d_train, 1000, d_test, verbose_eval=100, early_stopping_rounds=100)
    predict_test = model.predict(X_test_std)
    mse = mean_squared_error(y_test, predict_test)
    print('RMSE: ', np.sqrt(mse))
    joblib.dump(model, './data/result/lgb.pkl')
    return


def saber_matrix(df, j, types):
    df['ERA'] = (df.ER_teampit / (df.INN2_teampit / 3)) * 9  # 평균자책점
    df['ERA_SP'] = (df.ER_sp / (df.INN2_sp / 3)) * 9  # 선발 평균자책점

    df['AVG_bat'] = df.HIT_bat / df.AB_bat  # 타율
    df['AVG_pit'] = df.HIT_pit / df.AB_pit  # 피안타율

    df['OPS_bat'] = ((df.HIT_bat + df.BB_bat) +
                     ((df.HIT_bat - df.H2_bat - df.H3_bat - df.HR_bat)
                      + (df.H2_bat * 2) + (df.H3_bat * 3) + (df.HR_bat * 4))) / df.AB_bat
    df['OPS_pit'] = ((df.HIT_pit + df.BB_pit) +
                     ((df.HIT_pit - df.H2_pit - df.H3_pit - df.HR_pit)
                      + (df.H2_pit * 2) + (df.H3_pit * 3) + (df.HR_pit * 4))) / df.AB_pit

    df['wOBA_bat'] = ((0.69 * (df.BB_bat))
                      + (0.89 * (df.HIT_bat - df.H2_bat - df.H3_bat - df.HR_bat))
                      + (1.27 * df.H2_bat)
                      + (1.62 * df.H3_bat)
                      + (2.1 * df.HR_bat)) / (df.AB_bat + df.BB_bat)

    df['wOBA_pit'] = ((0.69 * (df.BB_pit))
                      + (0.89 * (df.HIT_pit - df.H2_pit - df.H3_pit - df.HR_pit))
                      + (1.27 * df.H2_pit)
                      + (1.62 * df.H3_pit)
                      + (2.1 * df.HR_pit)) / (df.AB_pit + df.BB_pit)

    df['FIP'] = (((-2 * df.KK_pit) + (3 * (df.BB_pit)) + (13 * df.HR_pit))
                 / (df.INN2_teampit / 3)) + 3.0

    df['WHIP'] = (df.HIT_pit + df.BB_pit) / (df.INN2_teampit / 3)

    df['ISO'] = (df.H2_bat + (2 * df.H3_bat) + (3 * df.HR_bat)) / df.AB_bat
    df['PE'] = (df.RUN ** 2) / ((df.RUN ** 2) + (df.R ** 2))

    if types == 'train':
        df['WR_NEXT'] = (df.NEXT_WLS) / j
    return df



def making_input(games_count, dataset):
    tmp_era = [
        'WLS_changed', 'INN2_teampit',
        'BF', 'AB_pit', 'HIT_pit', 'H2_pit', 'H3_pit', 'HR_pit', 'SB_pit',
        'BB_pit', 'KK_pit', 'GD_pit', 'R', 'ER_teampit', 'INN2_sp', 'ER_sp'
    ]

    tmp_avg = [
        'WLS_changed',
        'AB_bat', 'RBI', 'RUN', 'HIT_bat', 'H2_bat', 'H3_bat', 'HR_bat',
        'SB_bat', 'BB_bat', 'KK_bat', 'GD_bat', 'ERR', 'LOB',
        #         'type1', 'type2', 'type3', 'type4',
        'PE', '3hal'
    ]
    input_data_era = []
    input_data_avg = []
    labels_AVG = []
    labels_ERA = []
    #
    for i in ['HH', 'HT', 'KT', 'LG', 'LT', 'NC', 'OB', 'SK', 'SS', 'WO']:
        for j in [(20160101, 20161231), (20170101, 20171231), (20180101, 20181231), (20190101, 20191231), (20200101, 20201231)]:
            data = dataset[(dataset['T_ID'] == i) & (dataset['GDAY_DS'] < j[1]) & (dataset['GDAY_DS'] > j[0])].sort_values(by='G_ID')
            for k in range(len(data) - 52):
                era_features = data[tmp_era].iloc[k:k + 26, ].values
                avg_features = data[tmp_avg].iloc[k:k + 26, ].values

                era_features = np.append(era_features, data.iloc[k + 26:k + 52, ]['WLS_changed'].values.reshape(26, -1), axis=1)
                avg_features = np.append(avg_features, data.iloc[k + 26:k + 52, ]['WLS_changed'].values.reshape(26, -1), axis=1)

                input_data_era.append(era_features)
                input_data_avg.append(avg_features)

                labels_AVG.append(data[['HIT_bat']].iloc[k + 26:k + 26 + games_count, ].sum().values[0] /
                                  data[['AB_bat']].iloc[k + 26:k + 26 + games_count, ].sum().values[0])
                labels_ERA.append((data[['ER_teampit']].iloc[k + 26:k + 26 + games_count,].sum().values[0]) / (
                (data[['INN2_teampit']].iloc[k + 26:k + 26 + games_count,].sum().values[0] / 3)) * 9)
    return np.array(input_data_era), np.array(labels_ERA), np.array(input_data_avg), np.array(labels_AVG), tmp_era, tmp_avg

def Min_Max_scaling_3D(X_train, X_test, types):
    X_train = np.delete(X_train, -1, axis=2)
    X_test = np.delete(X_test, -1, axis=2)
    num_sample = X_train.shape[0]
    num_sequence = X_train.shape[1]
    num_feature = X_train.shape[2]

    scaler = MinMaxScaler()
    for ss in range(num_sequence):
        scaler.partial_fit(X_train[:, ss, :])

    results = []
    for ss in range(num_sequence):
        results.append(scaler.transform(X_train[:, ss, :]).reshape(num_sample, 1, num_feature))

    X_train_std = np.concatenate(results, axis=1)

    num_sample = X_test.shape[0]
    num_sequence = X_test.shape[1]
    num_feature = X_test.shape[2]

    # for ss in range(num_sequence):
    #     scaler.partial_fit(X_test[:, ss, :])

    results = []
    for ss in range(num_sequence):
        results.append(scaler.transform(X_test[:, ss, :]).reshape(num_sample, 1, num_feature))

    X_test_std = np.concatenate(results, axis=1)

    joblib.dump(scaler, f'./data/result/minmax_{types}.bin')
    return X_train_std, X_test_std

def train(X_input, y_input, types, X_test, y_test):
    # print('start train')
    np.random.seed(154)
    python_random.seed(154)
    tf.random.set_seed(154)
    model, early = get_model(types)
    model.fit(X_input, y_input,
              batch_size=128,
              epochs=2000,
              verbose=1,
              callbacks=[early],
              validation_split=0.05)
    model.save(f'./data/result/{types}_model.h5')
    predict_test = model.predict(X_test)
    mse = mean_squared_error(y_test, predict_test)
    print('RMSE: ', np.sqrt(mse))

def predict(types, predict_data):
    # print('start predict')
    loaded = keras.models.load_model(f'./data/result/{types}_model.h5')
    # print(predict_data.shape)
    result = loaded.predict(predict_data)
    return result

def for_lgb(X_test_era, y_test_era, y_predict_era,  X_test_avg, y_test_avg, y_predict_avg, era_columns, avg_columns):
    # print('start lgb_make')
    df_pred = pd.DataFrame()
    df_pred['ERA_LSTM_pred'] = y_predict_era
    df_pred['ERA_NEXT'] = y_test_era
    df_pred['AVG_LSTM_pred'] = y_predict_avg
    df_pred['AVG_NEXT'] = y_test_avg

    X_era = np.zeros(X_test_era.shape)

    for i in range(X_test_era.shape[0]):
        for j in range(X_test_era.shape[1]):
            for k in range(X_test_era.shape[2]):
                X_era[i][j][k] = X_test_era[i][j][k]
            X_era[i][j][-1] = X_test_era[i][j][-1]

    X_avg = np.zeros(X_test_avg.shape)
    for i in range(X_test_avg.shape[0]):
        for j in range(X_test_avg.shape[1]):
            for k in range(X_test_avg.shape[2]):
                X_avg[i][j][k] = X_test_avg[i][j][k]
            X_avg[i][j][-1] = X_test_avg[i][j][-1]

    dataset_era = X_era.sum(axis=1)
    dataset_avg = X_avg.sum(axis=1)
    df_era = pd.DataFrame(dataset_era, columns=era_columns + ['NEXT_WLS'])
    df_avg = pd.DataFrame(dataset_avg, columns=avg_columns + ['NEXT_WLS'])
    df_era['ERA_pred'] = df_pred['ERA_LSTM_pred']
    df_avg['AVG_pred'] = df_pred['AVG_LSTM_pred']
    df_wr = pd.merge(df_era, df_avg, how='inner', left_index=True, right_index=True,
                     on=['WLS_changed', 'NEXT_WLS'])
    df_wr = df_wr[['WLS_changed', 'INN2_teampit', 'BF', 'AB_pit', 'HIT_pit', 'H2_pit', 'H3_pit',
                   'HR_pit', 'SB_pit', 'BB_pit', 'KK_pit', 'GD_pit', 'R', 'ER_teampit',
                   'INN2_sp', 'ER_sp', 'AB_bat', 'RBI', 'RUN',
                   'HIT_bat', 'H2_bat', 'H3_bat', 'HR_bat', 'SB_bat', 'BB_bat', 'KK_bat',
                   'GD_bat', 'ERR', 'LOB', 'PE', '3hal', 'AVG_pred', 'ERA_pred', 'NEXT_WLS']]

    return df_wr

def main():
    data = pd.read_csv('./data/result/train_data.csv')
    X_era, y_era, X_avg, y_avg, era_columns, avg_columns = making_input(26, data)
    X_train_era, X_test_era, y_train_era, y_test_era = train_test_split(X_era, y_era, random_state=0, test_size=0.35)
    X_train_avg, X_test_avg, y_train_avg, y_test_avg = train_test_split(X_avg, y_avg, random_state=0, test_size=0.35)

    X_train_std_era, X_test_std_era =  Min_Max_scaling_3D(X_train_era, X_test_era, 'era')
    X_train_std_avg, X_test_std_avg = Min_Max_scaling_3D(X_train_avg, X_test_avg, 'avg')

    train(X_train_std_era, y_train_era, 'era', X_test_std_era, y_test_era)
    train(X_train_std_avg, y_train_avg, 'avg', X_test_std_avg, y_test_avg)

    era_predict = predict('era', X_test_std_era)
    avg_predict = predict('avg', X_test_std_avg)
    lgb_data = for_lgb(X_test_era, y_test_era, era_predict.reshape(-1,), X_test_avg, y_test_avg, avg_predict.reshape(-1,), era_columns, avg_columns)
    lgb_predict(lgb_data)


if __name__ == '__main__':
    main()

