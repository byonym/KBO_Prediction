import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def predict_wrate(data_set, j=1, types='test'):
    # 비율지표 및 세이버매트릭스 지표 생성
    test_sab = saber_matrix(data_set, 1, types)
    # 학습
    X = test_sab.values
    scaler = joblib.load('./data/result/std_scaler.bin')
    X_test_scaled= scaler.transform(X)
    model = joblib.load('./data/result/lgb.pkl')
    y_pred = model.predict(X_test_scaled)

    return y_pred


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



def eda_for_predict(data):
    global team_name
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

    for i in team_name:
        dataset = data[data['T_ID']==i]
        era_testset = dataset[tmp_era].values
        avg_testset = dataset[tmp_avg].values
        input_data_era.append(era_testset)
        input_data_avg.append(avg_testset)

    era_input_final = np.array(input_data_era)
    avg_input_final = np.array(input_data_avg)

    num_sample = era_input_final.shape[0]
    num_sequence = era_input_final.shape[1]
    num_feature = era_input_final.shape[2]

    scaler_era = joblib.load(f'./data/result/minmax_era.bin')
    results = []
    for ss in range(num_sequence):
        results.append(scaler_era.transform(era_input_final[:, ss, :]).reshape(num_sample, 1, num_feature))

    era_std_final = np.concatenate(results, axis=1)


    num_sample = avg_input_final.shape[0]
    num_sequence = avg_input_final.shape[1]
    num_feature = avg_input_final.shape[2]

    scaler_avg = joblib.load(f'./data/result/minmax_avg.bin')

    results = []
    for ss in range(num_sequence):
        results.append(scaler_avg.transform(avg_input_final[:, ss, :]).reshape(num_sample, 1, num_feature))

    avg_std_final = np.concatenate(results, axis=1)



    return era_input_final, era_std_final, avg_input_final, avg_std_final

def predict_era_avg(types, predict_data):
    loaded = keras.models.load_model(f'./data/result/{types}_model.h5')

    result = loaded.predict(predict_data)
    return result

def for_lgb_fin(era_predict_data, era_predict_result, avg_predict_data, avg_predict_result):
    X_era = era_predict_data[:]

    X_avg = avg_predict_data[:]

    dataset_era = X_era.sum(axis=1)
    dataset_avg = X_avg.sum(axis=1)
    era_columns = ['WLS_changed', 'INN2_teampit','BF', 'AB_pit', 'HIT_pit', 'H2_pit', 'H3_pit', 'HR_pit', 'SB_pit','BB_pit', 'KK_pit', 'GD_pit', 'R', 'ER_teampit', 'INN2_sp', 'ER_sp']
    avg_columns = ['WLS_changed','AB_bat', 'RBI', 'RUN', 'HIT_bat', 'H2_bat', 'H3_bat', 'HR_bat','SB_bat', 'BB_bat', 'KK_bat', 'GD_bat', 'ERR', 'LOB','PE', '3hal']


    df_era = pd.DataFrame(dataset_era, columns=era_columns)
    df_avg = pd.DataFrame(dataset_avg, columns=avg_columns)
    df_era['ERA_pred'] = era_predict_result
    df_avg['AVG_pred'] = avg_predict_result
    df_wr = pd.merge(df_era, df_avg, how='inner', left_index=True, right_index=True,
    on=['WLS_changed'])
    df_wr = df_wr[['WLS_changed', 'INN2_teampit', 'BF', 'AB_pit', 'HIT_pit', 'H2_pit', 'H3_pit',
                   'HR_pit', 'SB_pit', 'BB_pit', 'KK_pit', 'GD_pit', 'R', 'ER_teampit',
                   'INN2_sp', 'ER_sp', 'AB_bat', 'RBI', 'RUN',
                   'HIT_bat', 'H2_bat', 'H3_bat', 'HR_bat', 'SB_bat', 'BB_bat', 'KK_bat',
                   'GD_bat', 'ERR', 'LOB', 'PE', '3hal', 'AVG_pred', 'ERA_pred']]
    return df_wr

def main():
    global team_name
    test_data = pd.read_csv('./data/result/test_data.csv')
    era_input_final, era_std_final, avg_input_final, avg_std_final = eda_for_predict(test_data.sort_values(by=['G_ID']))


    era_result = predict_era_avg("era", era_std_final)
    avg_result = predict_era_avg("avg", avg_std_final)
    lgbdf = for_lgb_fin(era_input_final, era_result, avg_input_final, avg_result)
    # lgbdf['Team'] = np.array(team_name)
    wrate = predict_wrate(lgbdf)

    result = pd.DataFrame()
    result['Team'] = team_name
    result['ERA'] = era_result
    result['AVG'] = avg_result
    result['W_Rate'] = wrate

    result.sort_values(by='W_Rate', ascending=False).to_csv('./result/prediction.csv', index=False)



if __name__ == '__main__':
    #
    team_name = ['HH', 'HT', 'KT', 'LG', 'LT', 'NC', 'OB', 'SK', 'SS', 'WO']
    main()

