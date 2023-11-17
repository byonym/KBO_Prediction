from selenium import webdriver
import selenium
from bs4 import BeautifulSoup
import time
import pandas as pd
import numpy as np
import os

def crawling(game_num):
    driver = webdriver.Chrome('./crawling/chromedriver')
    driver.implicitly_wait(3)
    new = []
    driver.get(f'https://sports.daum.net/gamecenter/{game_num}/highlight')
    html = driver.page_source
    time.sleep(2)

    soup = BeautifulSoup(html, 'html.parser')
    pitch = soup.select('#gameResultPitcherRecordWrapper')
    hit = soup.select('#gameResultBatterRecordWrapper')

    overall = soup.select('#default_4 > div.gc_cont.gc_record > div')
    date =soup.select('#gameScoreboardWrap > div > div > div.info_game > span.txt_time')

    res_ov = making_overall(overall, date)
    home_p, away_p = making_pitch(pitch)
    home_t, away_t = making_hit(hit)
    team_p = making_team_pitch(res_ov, away_t, home_t, away_p, home_p)
    team_h = making_team_hit(res_ov, away_t, home_t)
    return team_p, team_h

def making_overall(data, date):
    res = []
    team_list = ['LG', 'SK', '삼성', '키움', '한화', 'NC', 'KIA', '두산', 'KT', '롯데']
    new2 = ['득점']
    title = ['팀']
    team_cnt = 0
    for j in data:
        new = j.text.split('\n')
    for i in new:
        if team_cnt ==2:
            res.append(title)
            team_cnt+=1

        if i in team_list:
            title.append(i)
            team_cnt+=1
            continue

        if i=='':
            continue
        if '(' in i or 'vs' in i or '더보기' in i:
            continue
        if len(new2) == 3:
            res.append(new2)
            new2 = []
        new2.append(i)
    over = pd.DataFrame(res)
    o = over.T
    date = date[0].text[:5].replace('.', '')
    o[10] = np.array(['날짜', date, date])
    headers_o = o.iloc[0]
    o  = pd.DataFrame(o.values[1:], columns=headers_o)
    o[['득점', '안타', '홈런', '타율', '탈삼진', '도루', '실책', '병살', '잔루']] = o[['득점', '안타', '홈런', '타율', '탈삼진', '도루', '실책', '병살', '잔루']].astype('float32')
    return o

def making_pitch(data):
    new2 = []
    team_list = ['LG', 'SK', '삼성', '키움', '한화', 'NC', 'KIA', '두산', 'KT', '롯데']
    for j in data:
        new = j.text.split('\n')
    temp = []
    count = 0
    team_cnt = 0


    for i in new:
        if i in team_list:
            team_cnt+=1

        if team_cnt == 2:
            away_p = pd.DataFrame(new2)
            headers_a = away_p.iloc[0]
            away_p  = pd.DataFrame(away_p.values[1:], columns=headers_a)
            away_p['이닝'] =away_p['이닝'].apply(inning_change)
            away_p[['타자', '투구수', '타수', '피안타', '피홈런', '탈삼진', '사사구', '실점', '자책', 'ERA']] = away_p[['타자', '투구수', '타수', '피안타', '피홈런', '탈삼진', '사사구', '실점', '자책', 'ERA']].astype('float64')
            new2 = []
            team_cnt+=1

        if i == '':
            continue
        if '기록' in i:
            continue
        if i in ['승', '패', '홀', '세']:
            continue
        count+=1
        if count != 13:
            temp.append(i)
        else:
            new2.append(temp)
            temp=[]
            count=0
    home_p = pd.DataFrame(new2)
    headers_h = home_p.iloc[0]
    home_p  = pd.DataFrame(home_p.values[1:], columns=headers_h)
    home_p['이닝'] =home_p['이닝'].apply(inning_change)
    home_p[['타자', '투구수', '타수', '피안타', '피홈런', '탈삼진', '사사구', '실점', '자책', 'ERA']] = home_p[['타자', '투구수', '타수', '피안타', '피홈런', '탈삼진', '사사구', '실점', '자책', 'ERA']].astype('float64')
    return away_p, home_p

def inning_change(x):
    inni = {'0 ⅓': 1/3, '1 ⅓': 4/3, '2 ⅓': 7/3, '3 ⅓': 10/3, '4 ⅓': 13/3, '5 ⅓': 16/3, '6 ⅓': 19/3, '7 ⅓': 22/3, '8 ⅓':25/3,\
           '0 ⅔':2/3, '1 ⅔':5/3, '2 ⅔':8/3, '3 ⅔':11/3, '4 ⅔':14/3, '5 ⅔':17/3, '6 ⅔':20/3, '7 ⅔':23/3, '8 ⅔':26/3,\
           '0': 0, '1': 1, '2': 2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
    x = inni[x]
    return x

def making_team_pitch(res_ov, away_t, home_t, away_p, home_p):
    team_code = {'LG': 'LG', 'SK': 'SK', 'KT':'KT', '한화': 'HH', '키움':'WO', '삼성': 'SS', '두산':'OB', 'KIA':'HT', '롯데':'LT', 'NC':'NC'}

    if res_ov['득점'][0] > res_ov['득점'][1]:
        aw = 'W'
        ho = 'L'
    elif res_ov['득점'][1] > res_ov['득점'][0]:
        aw = 'L'
        ho = 'W'
    else:
        aw = 'D'
        ho = 'D'

    aw_p = [int('20'+res_ov['날짜'][0]), team_code[res_ov['팀'][0]], team_code[res_ov['팀'][1]], 'T', aw, round(away_p['이닝'].sum())*3, away_p['투구수'].sum(),\
            home_t['타수'].sum(), home_t['안타'].sum(), home_t['2타'].sum(), home_t['3타'].sum(), home_t['홈런'].sum(),\
            res_ov['도루'][1], away_p['사사구'].sum(),away_p['탈삼진'].sum(), res_ov['병살'][1], away_p['실점'].sum(), away_p['자책'].sum()]


    ho_p = [int('20'+res_ov['날짜'][0]), team_code[res_ov['팀'][1]], team_code[res_ov['팀'][0]], 'B', ho, round(home_p['이닝'].sum())*3, home_p['투구수'].sum(),\
            away_t['타수'].sum(), away_t['안타'].sum(), away_t['2타'].sum(), away_t['3타'].sum(), away_t['홈런'].sum(),\
            res_ov['도루'][0], home_p['사사구'].sum(),home_p['탈삼진'].sum(), res_ov['병살'][0], home_p['실점'].sum(), home_p['자책'].sum()]

    team_pitch = pd.DataFrame([ho_p, aw_p], columns=['GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'WLS', 'INN2', 'BF', 'AB','HIT', 'H2', 'H3', 'HR',\
                                                    'SB', 'BB', 'KK', 'GD', 'R', 'ER'])
    return team_pitch


def making_hit(data):
    new2 = []
    team_list = ['LG', 'SK', '삼성', '키움', '한화', 'NC', 'KIA', '두산', 'KT', '롯데']
    for j in data:
        new = j.text.split('\n')
    temp = []
    count = 0
    team_cnt = 0

    for i in new:
        if i in team_list:
            team_cnt += 1
        if '(' in i:
            continue
        if team_cnt == 2:
            away_t = pd.DataFrame(new2)
            headers_a = away_t.iloc[0]
            away_t = pd.DataFrame(away_t.values[1:], columns=headers_a)
            away_t[['타수', '안타', '2타', '3타', '홈런', '득점', '타점', '삼진', '사사구']] = away_t[
                ['타수', '안타', '2타', '3타', '홈런', '득점', '타점', '삼진', '사사구']].astype('float64')
            new2 = []
            team_cnt += 1

        if i == '':
            continue
        if '기록' in i:
            continue
        if i in ['승', '패', '홀', '세']:
            continue
        count += 1
        if count != 11:
            temp.append(i)
        else:
            new2.append(temp)
            temp = []
            count = 0
    home_t = pd.DataFrame(new2)
    headers_h = home_t.iloc[0]
    home_t = pd.DataFrame(home_t.values[1:], columns=headers_h)
    home_t[['타수', '안타', '2타', '3타', '홈런', '득점', '타점', '삼진', '사사구']] = home_t[
        ['타수', '안타', '2타', '3타', '홈런', '득점', '타점', '삼진', '사사구']].astype('float64')

    return away_t, home_t

def making_team_hit(res_ov, away_t, home_t):
    team_code = {'LG': 'LG', 'SK': 'SK', 'KT':'KT', '한화': 'HH', '키움':'WO', '삼성': 'SS', '두산':'OB', 'KIA':'HT', '롯데':'LT', 'NC':'NC'}

    aw_t = [int('20'+res_ov['날짜'][0]), team_code[res_ov['팀'][0]], team_code[res_ov['팀'][1]], 'T', away_t['타수'].sum(), away_t['타점'].sum(),away_t['득점'].sum(),\
        away_t['안타'].sum(), away_t['2타'].sum(), away_t['3타'].sum(), away_t['홈런'].sum(), res_ov.iloc[0]['도루'], away_t['사사구'].sum(),\
        away_t['삼진'].sum(), res_ov.iloc[0]['병살'], res_ov.iloc[0]['실책'], res_ov.iloc[0]['잔루']]

    ho_t = [int('20'+res_ov['날짜'][0]), team_code[res_ov['팀'][1]], team_code[res_ov['팀'][0]], 'B', home_t['타수'].sum(), home_t['타점'].sum(),home_t['득점'].sum(),\
        home_t['안타'].sum(), home_t['2타'].sum(), home_t['3타'].sum(), home_t['홈런'].sum(), res_ov.iloc[1]['도루'], home_t['사사구'].sum(),\
        home_t['삼진'].sum(), res_ov.iloc[1]['병살'], res_ov.iloc[1]['실책'], res_ov.iloc[1]['잔루']]

    team_hit = pd.DataFrame([ho_t, aw_t], columns=['GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'AB', 'RBI', 'RUN', 'HIT', 'H2', 'H3', 'HR', 'SB', 'BB', 'KK', 'GD', 'ERR', 'LOB'])
    return team_hit

def main():
    

    team_pitch = pd.DataFrame(
        columns=['GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'WLS', 'INN2', 'BF', 'AB', 'HIT', 'H2', 'H3', 'HR', \
                 'SB', 'BB', 'KK', 'GD', 'R', 'ER'])

    team_hit = pd.DataFrame(
        columns=['GDAY_DS', 'T_ID', 'VS_T_ID', 'TB_SC', 'AB', 'RBI', 'RUN', 'HIT', 'H2', 'H3', 'HR', 'SB', 'BB', 'KK',
                 'GD', 'ERR', 'LOB'])

    for i in range(80033024, 80033025):
        try:
            team_p, team_h = crawling(i)
            team_pitch = pd.concat([team_pitch, team_p])
            team_hit = pd.concat([team_hit, team_h], ignore_index=True)
        except:
            pass
    
    team_pitch.to_csv('./crawling/result/test1.csv', index=False)
    team_hit.to_csv('./crawling/result/test2.csv', index=False)


if __name__ == '__main__':
    main()
    

