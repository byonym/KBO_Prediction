import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans

def rawdata_preprocess():
    # loading raw data
    indiv_hit_16 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2016.csv', encoding='cp949')
    indiv_hit_17 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2017.csv', encoding='cp949')
    indiv_hit_18 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2018.csv', encoding='cp949')
    indiv_hit_19 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2019.csv', encoding='cp949')
    indiv_hit_20 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2020.csv', encoding='cp949')
    indiv_pit_16 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2016.csv', encoding='cp949')
    indiv_pit_17 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2017.csv', encoding='cp949')
    indiv_pit_18 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2018.csv', encoding='cp949')
    indiv_pit_19 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2019.csv', encoding='cp949')
    indiv_pit_20 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2020.csv', encoding='cp949')

    team_hit_16 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2016.csv', encoding='cp949')
    team_hit_17 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2017.csv', encoding='cp949')
    team_hit_18 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2018.csv', encoding='cp949')
    team_hit_19 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2019.csv', encoding='cp949')
    team_hit_20 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2020.csv', encoding='cp949')
    team_pit_16 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2016.csv', encoding='cp949')
    team_pit_17 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2017.csv', encoding='cp949')
    team_pit_18 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2018.csv', encoding='cp949')
    team_pit_19 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2019.csv', encoding='cp949')
    team_pit_20 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2020.csv', encoding='cp949')

    # load crawling data
    cr_ind_hit = pd.read_csv('./data/crawling/crawl_ind_h.csv', encoding='cp949')
    cr_ind_pit = pd.read_csv('./data/crawling/crawl_ind_p.csv', encoding='cp949')
    cr_team_hit = pd.read_csv('./data/crawling/crawl_team_h.csv', encoding='cp949')
    cr_team_pit = pd.read_csv('./data/crawling/crawl_team_p.csv', encoding='cp949')


    # concat by theirselves
    indiv_pit = pd.concat([indiv_pit_16, indiv_pit_17, indiv_pit_18, indiv_pit_19, indiv_pit_20])
    indiv_hit = pd.concat([indiv_hit_16, indiv_hit_17, indiv_hit_18, indiv_hit_19, indiv_hit_20])
    team_hit = pd.concat([team_hit_16, team_hit_17, team_hit_18, team_hit_19, team_hit_20])
    team_pit = pd.concat([team_pit_16, team_pit_17, team_pit_18, team_pit_19, team_pit_20])

    # sum BB+IB+HP
    indiv_pit['BB'] = indiv_pit['BB'] + indiv_pit['IB'] + indiv_pit['HP']
    indiv_hit['BB'] = indiv_hit['BB'] + indiv_hit['IB'] + indiv_hit['HP']
    team_pit['BB'] = team_pit['BB'] + team_pit['IB'] + team_pit['HP']
    team_hit['BB'] = team_hit['BB'] + team_hit['IB'] + team_hit['HP']

    # sum HIT+H2+H3
    indiv_pit['HIT'] = indiv_pit['HIT'] + indiv_pit['H2'] + indiv_pit['H3']

    # concat all_data with crawling data
    indiv_pit_all = pd.concat([indiv_pit[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'P_ID', 'START_CK',
                                          'RELIEF_CK', 'INN2', 'BF', 'PA', 'AB', 'HIT', 'HR', 'KK', 'BB', 'R',
                                          'ER']],
                               cr_ind_pit[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'P_ID', 'START_CK',
                                           'RELIEF_CK', 'INN2', 'BF', 'PA', 'AB', 'HIT', 'HR', 'KK', 'BB', 'R',
                                           'ER']]])

    indiv_hit_all = pd.concat([indiv_hit[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'P_ID', 'AB', 'RBI',
                                          'RUN', 'HIT', 'H2', 'H3', 'HR', 'BB', 'KK']], cr_ind_hit])

    team_hit_all = pd.concat([team_hit[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'AB', 'RBI', 'RUN',
                                        'HIT', 'H2', 'H3', 'HR', 'SB', 'BB', 'KK', 'GD', 'ERR', 'LOB']], cr_team_hit])

    team_pit_all = pd.concat([team_pit[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'WLS', 'INN2', 'BF',
                                        'AB', 'HIT', 'H2', 'H3', 'HR', 'SB', 'BB', 'KK', 'GD', 'R', 'ER']],
                              cr_team_pit])

    indiv_pit_all = indiv_pit_all[indiv_pit_all['P_ID'] != 'No_Code']
    indiv_pit_all['P_ID'] = indiv_pit_all['P_ID'].astype('int64')
    indiv_hit_all = indiv_hit_all[indiv_hit_all['P_ID'] != 'No_Code']
    indiv_hit_all['P_ID'] = indiv_hit_all['P_ID'].astype('int64')
    team_pit_all['WLS_changed'] = team_pit_all['WLS'].apply(lambda x: 1 if x=='W' else 0)

    return indiv_pit_all, indiv_hit_all, team_hit_all, team_pit_all


def add_and_concat(ind_pit, team_pit, team_hit):
    pitcher_start = ind_pit[ind_pit['START_CK'] == 1].reset_index(drop=True)
    pitcher_start = pitcher_start[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'INN2', 'ER']]
    pitcher_team_all = pd.merge(team_pit, pitcher_start, how='inner',
                                on=['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC'],
                                suffixes=('_teampit', '_sp'))

    team_all = pd.merge(pitcher_team_all, team_hit, how='inner',
                        on=['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC'],
                        suffixes=('_pit', '_bat'))

    return team_all


def saber_stats(series):
    series['ERA'] = (series['ER_teampit'] / (series['INN2_teampit'] / 3)) * 9

    series['AVG_bat'] = series['HIT_bat'] / series['AB_bat']
    series['AVG_pit'] = series['HIT_pit'] / series['AB_pit']

    series['OPS_bat'] = ((series['HIT_bat'] + series['BB_bat'] + (
                (series['HIT_bat'] - series['H2_bat'] - series['H3_bat'] - series['HR_bat'])
                + (series['H2_bat'] * 2) + (series['H3_bat'] * 3) + (series['HR_bat'] * 4)))) / series['AB_bat']

    series['OPS_pit'] = ((series['HIT_pit'] + series['BB_pit'] + (
                (series['HIT_pit'] - series['H2_pit'] - series['H3_pit'] - series['HR_pit'])
                + (series['H2_pit'] * 2) + (series['H3_pit'] * 3) + (series['HR_pit'] * 4)))) / series['AB_pit']

    series['wOBA_bat'] = ((0.69 * (series['BB_bat']))
                          + (0.89 * (series['HIT_bat'] - series['H2_bat'] - series['H3_bat'] - series['HR_bat']))
                          + (1.27 * series['H2_bat'])
                          + (1.62 * series['H3_bat'])
                          + (2.1 * series['HR_bat'])) / (series['AB_bat'] + series['BB_bat'])

    series['wOBA_pit'] = ((0.69 * (series['BB_pit']))
                          + (0.89 * (series['HIT_pit'] - series['H2_pit'] - series['H3_pit'] - series['HR_pit']))
                          + (1.27 * series['H2_pit'])
                          + (1.62 * series['H3_pit'])
                          + (2.1 * series['HR_pit'])) / (series['AB_pit'] + series['BB_pit'])

    series['FIP'] = (((-2 * series['KK_pit']) + (3 * (series['BB_pit'])) + (13 * series['HR_pit']))
                     / (series['INN2_teampit'] / 3)) + 3.0

    series['WHIP'] = (series['HIT_pit'] + series['BB_pit']) / (series['INN2_teampit'] / 3)

    series['ISO'] = (series['H2_bat'] + (2 * series['H3_bat']) + (3 * series['HR_bat'])) / series['AB_bat']
    series['PE'] = (series['RUN'] ** 2) / ((series['RUN'] ** 2) + (series['R'] ** 2))
    return series

def rel_w_rate(data):
    new = [['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'wins', 'loses']]
    for k in ['LG', 'OB', 'SS', 'HH', 'HT', 'WO', 'KT', 'SK', 'LT', 'NC']:
        for j in ['LG', 'OB', 'SS', 'HH', 'HT', 'WO', 'KT', 'SK', 'LT', 'NC']:
            if k == j:
                continue
            for i in range(len(data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)])):
                if i == 0:
                    new.append([data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i].G_ID,
                                data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i,].GDAY_DS, \
                                k, j, 0, 0])

                elif 0 < i < 5:
                    win = list(data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[:i].WLS.values).count('W')
                    lose = list(data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[:i].WLS.values).count('L')
                    new.append([data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i].G_ID,
                                data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i].GDAY_DS, \
                                k, j, win, lose])

                else:
                    new.append([data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i].G_ID,
                                data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i].GDAY_DS, \
                                k, j, \
                                list(data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i - 5:i].WLS.values).count(
                                    'W'), \
                                list(data[(data['T_ID'] == k) & (data['VS_T_ID'] == j)].iloc[i - 5:i].WLS.values).count(
                                    'L')])
    return pd.DataFrame(new[1:], columns=new[0])

def count_of_avg_3(data):
    batter = data[:]
    player_list = list(batter['P_ID'].unique())
    batter = batter[['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'P_ID', 'AB', 'HIT']]
    make = pd.DataFrame(columns=['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'P_ID', 'AB', 'HIT', 'cum_AB', 'cum_HIT', 'AVG'])
    for i in player_list:
        player = batter[batter['P_ID'] == i]
        player['cum_AB'] = player['AB'].cumsum()
        player['cum_HIT'] = player['HIT'].cumsum()
        player['AVG'] = player['cum_HIT'] / player['cum_AB']
        make = pd.concat([make, player])

    make['AVG'] = make['AVG'].fillna(0)

    team_3hal = pd.DataFrame(columns=['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', '3hal'])
    for t in ['NC', 'LT', 'OB', 'SK', 'SS', 'HT', 'HH', 'WO', 'KT', 'LG']:
        temp = [['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', '3hal']]
        for j in list(make[make['T_ID'] == t].G_ID.unique()):
            ttemp = [j, make[make['G_ID'] == j].GDAY_DS.unique()[0], t,
                     make[(make['G_ID'] == j) & (make['T_ID'] == t)].VS_T_ID.unique()[0]]
            ttemp.append(len(make[(make['G_ID'] == j) & (make['AVG'] >= 0.3) & (make['T_ID'] == t)]))
            temp.append(ttemp)
        temp = pd.DataFrame(temp[1:], columns=temp[0])
        team_3hal = pd.concat([team_3hal, temp])
    return team_3hal

def money():
    #load data
    sal_2016 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2016.csv', encoding='cp949').dropna()
    sal_2017 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2017.csv', encoding='cp949').dropna()
    sal_2018 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2018.csv', encoding='cp949').dropna()
    sal_2019 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2019.csv', encoding='cp949').dropna()
    sal_2020 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2020.csv', encoding='cp949').dropna()


    absal_2016 = sal_2016['MONEY'].str.slice(0,-2).astype(int)
    pcode_2016 = sal_2016[['PCODE','T_ID']]
    result_2016 = pd.concat([pcode_2016,absal_2016], axis=1)

    absal_2017 = sal_2017['MONEY'].str.slice(0,-2).astype(int)
    pcode_2017 = sal_2017[['PCODE','T_ID']]
    result_2017 = pd.concat([pcode_2017,absal_2017], axis=1)

    absal_2018 = sal_2018['MONEY'].str.slice(0,-2).astype(int)
    pcode_2018 = sal_2018[['PCODE','T_ID']]
    result_2018 = pd.concat([pcode_2018,absal_2018], axis=1)

    absal_2019 = sal_2019['MONEY'].str.slice(0,-2).astype(int)
    pcode_2019 = sal_2019[['PCODE','T_ID']]
    result_2019 = pd.concat([pcode_2019,absal_2019], axis=1)

    absal_2020 = sal_2020['MONEY'].str.slice(0,-2).astype(int)
    pcode_2020 = sal_2020[['PCODE','T_ID']]
    result_2020 = pd.concat([pcode_2020,absal_2020], axis=1)

    data_17 = pd.merge(result_2016,result_2017, on ='PCODE', suffixes =('_2016','_2017')).dropna()
    data_17['fluctuation'] = data_17['MONEY_2017']/data_17['MONEY_2016']
    data_18 = pd.merge(result_2017,result_2018, on ='PCODE', suffixes =('_2017','_2018')).dropna()
    data_18['fluctuation'] = data_18['MONEY_2018']/data_18['MONEY_2017']
    data_19 = pd.merge(result_2018,result_2019, on ='PCODE', suffixes =('_2018','_2019')).dropna()
    data_19['fluctuation'] = data_19['MONEY_2019']/data_19['MONEY_2018']
    data_20 = pd.merge(result_2019,result_2020, on ='PCODE', suffixes =('_2019','_2020')).dropna()
    data_20['fluctuation'] = data_20['MONEY_2020']/data_20['MONEY_2019']

    df17 = pd.DataFrame(data_17.groupby('T_ID_2017')['fluctuation'].mean()).reset_index()
    df18 = pd.DataFrame(data_18.groupby('T_ID_2018')['fluctuation'].mean()).reset_index()
    df19 = pd.DataFrame(data_19.groupby('T_ID_2019')['fluctuation'].mean()).reset_index()
    df20 = pd.DataFrame(data_20.groupby('T_ID_2020')['fluctuation'].mean()).reset_index()

    jsn2017 = dict()
    jsn2018 = dict()
    jsn2019 = dict()
    jsn2020 = dict()

    for i in np.array(df17):
        jsn2017[i[0]] = i[1]

    for i in np.array(df18):
        jsn2018[i[0]] = i[1]

    for i in np.array(df19):
        jsn2019[i[0]] = i[1]

    for i in np.array(df20):
        jsn2020[i[0]] = i[1]
    return jsn2017, jsn2018, jsn2019, jsn2020

def kmeans():
    players = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_선수_2020.csv', encoding='cp949').dropna()
    indiv_hit_18 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2018.csv', encoding='cp949')
    indiv_hit_19 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2019.csv', encoding='cp949')
    indiv_hit_20 = pd.read_csv('./data/raw_data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2020.csv', encoding='cp949')
    frame = [indiv_hit_18, indiv_hit_19, indiv_hit_20]
    batter_ind_all = pd.concat(frame).reset_index().drop('index', axis=1)
    players = players[['PCODE', 'NAME']]
    players.rename(columns={'PCODE': 'P_ID'}, inplace=True)
    players.drop_duplicates(subset="P_ID", inplace=True)
    batter_ind_all = batter_ind_all[['P_ID','START_CK', 'BAT_ORDER_NO', 'PA', 'AB', 'RBI', 'RUN', 'HIT', 'H2', 'H3','HR', 'SB', 'CS', 'SH', 'SF', 'BB', 'IB', 'HP', 'KK', 'GD', 'ERR',
                        'LOB', 'P_HRA_RT', 'P_AB_CN', 'P_HIT_CN']]
    df = pd.DataFrame(batter_ind_all.groupby('P_ID')[['PA', 'START_CK', 'AB', 'RBI', 'RUN', 'HIT', 'H2', 'H3',
                                                      'HR', 'SB', 'CS', 'SH', 'SF', 'BB', 'IB', 'HP', 'KK', 'GD',
                                                      'ERR']].apply(sum))
    df = df[df['PA'] >= 900]
    df = df.apply(lambda x: create_saber_stats_for_Kmeans(x), axis=1)
    df['BB'] = df.apply(lambda x: x.BB + x.IB + x.HP, axis=1)
    df = df.drop(['IB', 'HP'], axis=1)
    df['START_PA'] = df.apply(lambda x: x.START_CK / x.PA, axis=1)  # 스타팅으로 출전하는 빈도
    df['SB_attempt'] = df.apply(lambda x: (x.SB + x.CS) / (x.HIT - x.H2 - x.H3 - x.HR + x.BB), axis=1)  # 1루 진출 시 도루 시도
    df.loc[:, 'RBI':'ERR'] = df.loc[:, 'RBI':'ERR'].div(df.PA, axis=0)

    df_kmeans = df[['AVG', 'ISO', 'HR_HIT', 'PA_BB', 'SB_attempt']]
    X = df_kmeans.values

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=5, max_iter=30, random_state=0)
    pred = kmeans.fit_predict(X_std)
    df['type'] = pred
    df.reset_index(inplace=True)
    df_new = pd.merge(df, players, how='inner', on='P_ID')
    return df_new

def create_saber_stats_for_Kmeans(row):
    row['AVG'] = row.HIT / row.AB  # 타율

    row['OPS'] = ((row.HIT + row.BB + row.HP) + ((row.HIT - row.H2 - row.H3 - row.HR)
                                                 + (row.H2 * 2) + (row.H3 * 3) + (row.HR * 4))) / row.AB

    row['wOBA'] = ((0.69 * (row.BB - row.IB)) + (0.72 * row.HP) + (0.89 * (row.HIT - row.H2 - row.H3 - row.HR))
                   + (1.27 * row.H2) + (1.62 * row.H3) + (2.1 * row.HR)) / (row.AB + row.BB - row.IB + row.SF + row.HP)

    row['ISO'] = (row.H2 + (2 * row.H3) + (3 * row.HR)) / row.AB  # 타자의 파워

    row['PA_BB'] = row.PA / (row.BB + row.IB + row.HP)  # 단위 볼넷 당 타석 수

    row['HR_HIT'] = row.HR / row.HIT  # 단위 안타 당 홈런 개수
    return row


def foriegn_players():
    pitcher16 = pd.read_csv('./data_csv/pitcher/pitcher2016.csv', encoding='cp949')
    pitcher17 = pd.read_csv('./data_csv/pitcher/pitcher2017.csv', encoding='cp949')
    pitcher18 = pd.read_csv('./data_csv/pitcher/pitcher2018.csv', encoding='cp949')
    pitcher19 = pd.read_csv('./data_csv/pitcher/pitcher2019.csv', encoding='cp949')
    hitter16 = pd.read_csv('./data_csv/hitter/hitter2016.csv', encoding='cp949')
    hitter17 = pd.read_csv('./data_csv/hitter/hitter2017.csv', encoding='cp949')
    hitter18 = pd.read_csv('./data_csv/hitter/hitter2018.csv', encoding='cp949')
    hitter19 = pd.read_csv('./data_csv/hitter/hitter2019.csv', encoding='cp949')
    pitcher20 = pd.read_csv('./data_csv/2020_ind_pit.csv')
    hitter20 = pd.read_csv('./data_csv/2020_ind_hit.csv')
    pitcher20 = pitcher20[pitcher20['P_ID'] != 'No_Code']
    hitter20 = hitter20[hitter20['P_ID'] != 'No_Code']
    pitcher20['P_ID'] = pitcher20['P_ID'].astype('int')
    hitter20['P_ID'] = hitter20['P_ID'].astype('int')

    # 외국인 명단
    pit16 = {'WO': [65331, 62322, 66323, 66324], 'OB': [61240, 66226], 'LT': [65543, 65546],
             'SS': [66402, 66423, 66440, 66446], 'HH': [66748, 66750, 65742, 66742], \
             'HT': [66643, 66628], 'KT': [67845, 66032, 65331, 66049, 66050], 'LG': [66154, 66138, 62698],
             'NC': [65931, 63938], 'SK': [63810, 66825, 65856]}
    hit16 = {'WO': [66306], 'OB': [66244], 'LT': [65523, 66523], 'SS': [66452], 'HH': [66740], 'HT': [64699],
             'KT': [65005], \
             'LG': [65103], 'NC': [64914], 'SK': [66805]}

    pit17 = {'WO': [67312, 67313, 62322], 'OB': [61240, 66226], 'LT': [67559, 65546, 65543], 'SS': [67423, 67435],
             'HH': [67748, 67742], \
             'HT': [67645, 66643], 'KT': [67033, 65331], 'LG': [66138, 62698], 'NC': [67948, 63938],
             'SK': [65856, 67815]}
    hit17 = {'WO': [66306, 67394], 'OB': [66244], 'LT': [67598], 'SS': [67450], 'HH': [66740], 'HT': [67650],
             'KT': [67024, 67025], \
             'LG': [65103, 67134], 'NC': [67935], 'SK': [67827, 67872]}

    pit18 = {'WO': [65742, 63938, 67313], 'OB': [65543, 68240], 'LT': [65546, 68526], 'SS': [68435, 68400],
             'HH': [68748, 68742, 68794], \
             'HT': [67645, 66643], 'KT': [65331, 61240], 'LG': [62698, 68135], 'NC': [68953, 68948],
             'SK': [65856, 68815]}
    hit18 = {'WO': [67394, 68345], 'OB': [68244, 68245], 'LT': [67598], 'SS': [67450], 'HH': [68730], 'HT': [67650],
             'KT': [67025], \
             'LG': [68103], 'NC': [67935], 'SK': [67872]}

    pit19 = {'WO': [67313, 69343], 'OB': [65543, 68240], 'LT': [65546, 69550], 'SS': [69435, 69439, 69413],
             'HH': [69744, 69748], \
             'HT': [69640, 69656], 'KT': [69045, 69032], 'LG': [69103, 68135], 'NC': [69940, 69934, 69953],
             'SK': [69861, 62698, 68815]}
    hit19 = {'WO': [68345], 'OB': [69209], 'LT': [69530, 69569], 'SS': [67450], 'HH': [68730], 'HT': [69605, 69652],
             'KT': [67025], \
             'LG': [69150, 69165], 'NC': [69950, 69901], 'SK': [67872]}

    pit20 = {'WO': [67313, 69343], 'OB': [50234, 69045], 'LT': [50558, 50524], 'SS': [69439, 50404],
             'HH': [69744, 69748], \
             'HT': [50636, 50640], 'KT': [69032, 50040], 'LG': [69103, 68135], 'NC': [69940, 50912],
             'SK': [50815, 50835]}
    hit20 = {'WO': [50300, 99998], 'OB': [69209], 'LT': [50506], 'SS': [50468, 99999], 'HH': [68730, 50730],
             'HT': [69652], 'KT': [67025], \
             'LG': [50165], 'NC': [50923], 'SK': [67872, 99997]}

    # 16년도 외국인 용병 수치 구하기 (투수)- 17, 18, 19, 20년도 숫자만 바꿔주면 가능
    dt = [['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'FR_INN2', 'FR_ER']]
    for j in ['NC', 'LG', 'WO', 'HH', 'HT', 'SK', 'KT', 'OB', 'LT', 'SS']:
        for i in list(pitcher16[pitcher16['T_ID'] == j].G_ID.unique()):
            if len(pitcher16[(pitcher16.G_ID == i) & pitcher16.P_ID.isin(pit16[j])]):
                ER = pitcher16[(pitcher16.G_ID == i) & pitcher16.P_ID.isin(pit16[j])].ER.sum()
                INN = pitcher16[(pitcher16.G_ID == i) & pitcher16.P_ID.isin(pit16[j])].INN2.sum()
            else:
                ER = 0
                INN = 0
            dt.append(
                [pitcher16[(pitcher16.G_ID == i)].G_ID.values[0], pitcher16[(pitcher16.G_ID == i)].GDAY_DS.values[0], \
                 j, pitcher16[(pitcher16.G_ID == i) & (pitcher16.T_ID == j)].VS_T_ID.values[0], INN, ER])
    dt_16 = pd.DataFrame(dt[1:], columns=dt[0])
    dt_16.sort_values(by=['T_ID', 'GDAY_DS']).to_csv('./data/result/fr_pit_2016.csv', index=False)

    # 16년도 외국인 용병 수치 구하기 (타자) - 17, 18, 19, 20년도 숫자만 바꿔주면 가능
    df = [['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID', 'FR_AB', 'FR_HIT', 'FR_H2', 'FR_H3', 'FR_HR']]

    for j in ['NC', 'LG', 'WO', 'HH', 'HT', 'SK', 'KT', 'OB', 'LT', 'SS']:
        for i in list(hitter16[hitter16['T_ID'] == j].G_ID.unique()):
            if len(hitter16[(hitter16.G_ID == i) & hitter16.P_ID.isin(hit16[j])]):
                AB = hitter16[(hitter16.G_ID == i) & hitter16.P_ID.isin(hit16[j])].AB.sum()
                HIT = hitter16[(hitter16.G_ID == i) & hitter16.P_ID.isin(hit16[j])].HIT.sum()
                H2 = hitter16[(hitter16.G_ID == i) & hitter16.P_ID.isin(hit16[j])].H2.sum()
                H3 = hitter16[(hitter16.G_ID == i) & hitter16.P_ID.isin(hit16[j])].H3.sum()
                HR = hitter16[(hitter16.G_ID == i) & hitter16.P_ID.isin(hit16[j])].HR.sum()
            else:
                AB = 0
                HIT = 0
                H2 = 0
                H3 = 0
                HR = 0
            df.append([hitter16[(hitter16.G_ID == i)].G_ID.values[0], hitter16[(hitter16.G_ID == i)].GDAY_DS.values[0], \
                       j, hitter16[(hitter16.G_ID == i) & (hitter16.T_ID == j)].VS_T_ID.values[0], AB, \
                       HIT, H2, H3, HR])
    df_16 = pd.DataFrame(df[1:], columns=df[0])
    df_16.sort_values(by=['T_ID', 'GDAY_DS']).to_csv('fr_hit_2016.csv', index=False)



def train_test_split(team_data, team_three):
    team_merge = pd.merge(team_data, team_three, on=['G_ID', 'GDAY_DS', 'T_ID', 'VS_T_ID'], how='left')
    team_merge = team_merge.sort_values(by=['T_ID', 'G_ID'])
    team_train = pd.DataFrame(columns=team_merge.columns)
    team_test = pd.DataFrame(columns=team_merge.columns)
    for i in ['NC', 'KT', 'WO', 'LT', 'SK', 'SS', 'HT', 'HH', 'OB', 'LG']:
        team_train = pd.concat([team_train, team_merge[team_merge['T_ID'] == i].iloc[:-26].sort_values(by='G_ID')])
        team_test = pd.concat([team_test, team_merge[team_merge['T_ID'] == i].iloc[-1:-27:-1].sort_values(by='G_ID')])
    return team_train, team_test



def main():
    indiv_pit_all, indiv_hit_all, team_hit_all, team_pit_all = rawdata_preprocess()

    team_all = add_and_concat(indiv_pit_all, team_pit_all, team_hit_all)
    team_saber_all = saber_stats(team_all)

    # add_rel = rel_w_rate(team_saber_all) # 상대전적 구하는 함수인데 성능에 안 좋은 영향을 주어 안 썼습니다.
    # div_kmeans = kmeans()  # 선수들의 유형을 군집화하는 함수인데 성능이 약간 낮아져 안썼습니다.
    # foriegn_players()  # 외국인 선수들의 각 경기별 지표인데, 좋은 활용방안을 못 찾아 최종으로 쓰지 않았습니다.
    # sal17, sal18, sal19, sal20 = money() # 팀마다의 연봉 상승률을 고려하여 제작했지만, 성능에 궁정적 영향을 주지않아 사용하지 않았습니다.


    team_3hal = count_of_avg_3(indiv_hit_all)
    train_data, test_data = train_test_split(team_saber_all, team_3hal)
    train_data.to_csv('./data/result/train_data.csv', index=False)
    test_data.to_csv('./data/result/test_data.csv', index=False)

if __name__ == '__main__':
    main()