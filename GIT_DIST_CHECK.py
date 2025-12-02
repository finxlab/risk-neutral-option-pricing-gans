import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import copy
import os
import random
import joblib
import pickle
import time
from timeit import default_timer as timer
from datetime import datetime, timedelta
from tqdm import tqdm
import gc
import warnings
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
from statsmodels.tsa.stattools import acf


def compute_acf_score(real_series, generated_series_list, max_lag=20,
                      funcs=[lambda x: x, lambda x: x ** 2, lambda x: np.abs(x)]):

    M = generated_series_list.shape[1]

    acf_scores = []

    for f in funcs:
        real_transformed = f(real_series)
        gen_transformed = f(generated_series_list)

        real_acf = np.array([acf(real_transformed[:,gen], nlags=max_lag, fft=False) for gen in range(real_transformed.shape[1])])
        print(real_acf.shape)
        real_acf = np.mean(real_acf, axis=0)

        gen_acfs = np.array([acf(gen_transformed[:,gen], nlags=max_lag, fft=False) for gen in range(M)])
        mean_gen_acf = np.mean(gen_acfs, axis=0)

        acf_score = np.linalg.norm(real_acf - mean_gen_acf)
        acf_scores.append(acf_score)

    return acf_scores



sp_price = pd.read_csv('data/raw/sp500.csv')
sp_price.index = pd.DatetimeIndex(sp_price['Date'])
sp_price = sp_price[['Close']]
sp_price.columns = ['close']
sp_price = sp_price[sp_price.index >= '2000-01-01']


sp_ret = np.log(sp_price/sp_price.shift(1)).dropna()


## load all generated path
# path_SP500 = []
#
# for model_ in ['TimeGAN', 'QuantGAN','SigCWGAN']:
#     dir_ = 'noise_path/' + model_ +'/'
#     pathlist = os.listdir(dir_)
#     pathlist = [x for x in pathlist if '.pkl' in x]
#     pathlist = sorted(pathlist, key=lambda x: int(x.split('_')[-1].split('.')[0][5:]))
#     if model_ == 'QuantGAN':
#         pathlist = [joblib.load((dir_ + x)).T.squeeze(1)[:63] for x in pathlist]
#     else:
#         pathlist = [joblib.load((dir_ + x)).T[:63] for x in pathlist]
#     path_SP500.append(pathlist)





def dist_check_rev(ret_data, sim_path, windows_):

    np.random.seed(42)
    noises = np.random.normal(0, 1, (63, 10000))
    rb_list = pd.date_range(start = '2014-01-01', end='2023-09-01', freq='3MS')

    w_distance = []

    acf1 = []
    acf2 = []
    acf3 = []

    for rbidx in tqdm(range(len(rb_list))):
        temp_w = []
        temp_acf1 = []
        temp_acf2 = []
        temp_acf3 = []
        reb_date = rb_list[rbidx]

        act_noise = ret_data[ret_data.index <= reb_date].iloc[-3520:]

        act_noise = act_noise - act_noise.mean()[0]
        act_noise = act_noise/act_noise.std()[0]

        act_noise = pd.DataFrame(act_noise).rolling(window = windows_).sum().dropna()
        act_noise = act_noise.values


        temp_MC = pd.DataFrame(noises).rolling(window = windows_).sum().dropna()

        temp_acf = compute_acf_score(act_noise, temp_MC.values, max_lag = 20)
        temp_acf1.append(temp_acf[0])
        temp_acf2.append(temp_acf[1])
        temp_acf3.append(temp_acf[2])
        temp_MC = temp_MC.values.flatten()

        temp_w.append(stats.wasserstein_distance(act_noise.flatten(), temp_MC))
        for m_ in range(len(sim_path)):
            cur_model_path = sim_path[m_][rbidx]
            cur_model_path = pd.DataFrame(cur_model_path).rolling(window = windows_).sum().dropna()
            temp_acf = compute_acf_score(act_noise, cur_model_path.values, max_lag=20)

            temp_acf1.append(temp_acf[0])
            temp_acf2.append(temp_acf[1])
            temp_acf3.append(temp_acf[2])
            cur_model_path = cur_model_path.values.flatten()
            temp_w.append(stats.wasserstein_distance(act_noise.flatten(), cur_model_path))

        acf1.append(temp_acf1)
        acf2.append(temp_acf2)
        acf3.append(temp_acf3)

        w_distance.append(temp_w)

    return w_distance, acf1, acf2, acf3


def results_summary(results_df):

    was_ = pd.DataFrame(results_df[0])
    was_.columns = ['MC','TGAN','QGAN','SGAN']
    acf1_ = pd.DataFrame(results_df[1])
    acf1_.columns = ['MC','TGAN','QGAN','SGAN']
    acf2_ = pd.DataFrame(results_df[2])
    acf2_.columns = ['MC','TGAN','QGAN','SGAN']
    acf3_ = pd.DataFrame(results_df[3])
    acf3_.columns = ['MC','TGAN','QGAN','SGAN']

    return was_, acf1_, acf2_, acf3_



def check_summary(checks):
    res_ = pd.DataFrame(columns = ['MC', 'TGAN', 'QGAN', 'SGAN'],
                        index = ['WD(1)', 'WD(5)','WD(20)',
                                 'ACF(Raw)', 'ACF(Raw)','ACF(Raw)',
                                 'ACF(squared)','ACF(squared)','ACF(squared)',
                                 'ACF(Abs)','ACF(Abs)','ACF(Abs)'])
    res2 = res_.copy()
    print(res_.shape)
    i=0
    for mn in range(4):
        for c_ in range(3):

            res_.iloc[i] = list(checks[c_][mn].mean())
            res2.iloc[i] = list(checks[c_][mn].std())
            i+=1
    res_ = res_[['MC', 'TGAN', 'QGAN', 'SGAN']]
    return res_.astype(float).round(5), res2.astype(float).round(5)


r_1_m2 = dist_check_rev(sp_ret, path_SP500[:3], 1)
r_5_m2 = dist_check_rev(sp_ret, path_SP500[:3], 5)
r_20_m2 = dist_check_rev(sp_ret, path_SP500[:3], 20)


check1 = results_summary(r_1_m2)
check2 = results_summary(r_5_m2)
check3 = results_summary(r_20_m2)

checks = [check1, check2, check3]

res_mean, res_std  = check_summary(checks)
res_mean, res_std = res_mean.iloc[[0,1,2,3,6,9]], res_std.iloc[[0,1,2,3,6,9]]


res_mean.to_excel("results/generation_results/genres_mean.xlsx")
res_std.to_excel("results/generation_results/genres_std.xlsx")







np.random.seed(42)
noises = np.random.normal(0, 1, (63, 100000))
rb_list = pd.date_range(start = '2014-01-01', end='2023-09-01', freq='3MS')

ws_ = 20

raw_path = []
for rbidx in tqdm(range(len(rb_list))):
    reb_date = rb_list[rbidx]
    sample_data = sp_ret[sp_ret.index <= reb_date].iloc[-3520:]
    sample_data = sample_data - sample_data.mean()[0]
    sample_data = sample_data/sample_data.std()[0]
    sample_data = sample_data.rolling(window= ws_).sum().dropna().astype(float).round(5).values
    raw_path.extend(sample_data.flatten())


flatten_path = []

for i in range(3):
    cur_model_ = path_SP500[i]
    all_model_path = []
    for j in tqdm(range(len(cur_model_))):
        cur_paths = cur_model_[j][:63]
        cur_paths = pd.DataFrame(cur_paths).rolling(window=ws_).sum().dropna().astype(float).round(5).values
        all_model_path.extend(cur_paths.flatten())


    flatten_path.append(all_model_path)
    del(all_model_path, cur_model_)


all_model_path = []
simulations = pd.DataFrame(noises).rolling(window=ws_).sum().dropna().astype(float).round(5).values
all_model_path.append(simulations.flatten())

flatten_path.append(all_model_path)

fig, axs = plt.subplots(1, 4, figsize=(25, 5))
axs = axs.flatten()

label_name = ['TGAN','QGAN','SGAN' ,'$\mathbf{Z}$']
for i in range(4):
    # axs[i].set_xlim(-5.0 - 0.002, 5.0 + 0.002)
    # axs[i].set_ylim(0, 1.0)

    # axs[i].set_xlim(-10 - 0.002, 10.0 + 0.002)
    # axs[i].set_ylim(0, 0.4)

    axs[i].set_xlim(-15.0 - 0.002, 15.0 + 0.002)
    axs[i].set_ylim(0, 0.25)

    axs[i].tick_params(axis='x', labelsize=12)
    axs[i].tick_params(axis='y', labelsize=12)


    # axs[i].set_xticks(np.arange(-5.0, 5.0+0.0001, 5.0))
    # axs[i].set_xticks(np.arange(-10.0, 10.0+0.0001, 5.0))
    axs[i].set_xticks(np.arange(-15.0, 15.0+0.0001, 5.0))


    axs[i].hist(raw_path, bins=100, alpha=1.0, color='navy', edgecolor='white', label='Real',
             density=True)
    axs[i].hist(flatten_path[i], bins=100, alpha=0.4, color='#a00000', edgecolor='white', label=label_name[i], density=True)
    axs[i].legend(fontsize = 15, loc = 'upper right')
plt.tight_layout()
plt.subplots_adjust(wspace=0.6, hspace=0.6)


plt.savefig('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_GEN/' + 'gen_dist_1.png', dpi=800, transparent = True)
plt.savefig('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_GEN/' + 'gen_dist_1low.png', transparent = True)
plt.close()


plt.savefig('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_GEN/' + 'gen_dist_5.png', dpi=800, transparent = True)
plt.savefig('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_GEN/' + 'gen_dist_5low.png', transparent = True)
plt.close()

plt.savefig('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_GEN/' + 'gen_dist_20.png', dpi=800, transparent = True)
plt.savefig('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_GEN/' + 'gen_dist_20low.png', transparent = True)
plt.close()











price_ = pd.read_csv('C:/Users/user/Downloads/HEDGEPAPER/DATA/sp500.csv')
price_.index = pd.DatetimeIndex(price_['Date'])
price_ = price_[['Close']]
price_.columns = ['close']
price_ = price_[price_.index >= '2000-01-01']


div = pd.read_csv('C:/Users/user/Downloads/HEDGEPAPER/DATA/sp_div.csv', index_col = 0)
div.index = pd.DatetimeIndex(div.index)
div = div.sort_index()
div = div[div.index.year>=2000]
div = np.log(1+div/100)/252
div = div.asfreq('D').fillna(method = 'ffill')

rf = pd.read_csv('C:/Users/user/Downloads/HEDGEPAPER/DATA/rfcurve.csv',index_col = 0)
rf.index = pd.DatetimeIndex(rf.index)
rf = rf.asfreq('D').fillna(method = 'ffill')
rf = np.log((1+rf))/252


opdat = joblib.load('C:/Users/user/Downloads/HEDGEPAPER/REVISION/sp_fil.pkl')
opdat = opdat.loc[opdat['impl_volatility'].dropna().index]
opdat['impl_volatility'] = opdat['impl_volatility']/np.sqrt(252)


price_index = price_.index
start_dates_shifted = pd.to_datetime(opdat['date']) + timedelta(days=1)
start_locs = price_index.searchsorted(start_dates_shifted)
end_locs = price_index.searchsorted(pd.to_datetime(opdat['exdate']))
opdat['steps'] = end_locs - start_locs

del(price_index,start_dates_shifted, start_locs, end_locs)

rets = np.log(price_/price_.shift(1)).dropna()
r_length = len(rets[rets.index <= '2014-01-01'])

price_ = price_.asfreq('D').fillna(method = 'ffill')


opdat = opdat[opdat['Edate'] <= 91]

iv_ = opdat[['date', 'impl_volatility']].set_index('date')
iv_.index = pd.DatetimeIndex(iv_.index)
# 결과를 담을 빈 리스트
iv_temp_list = []
# 반복문을 돌며 shifted된 데이터 시리즈 생성
for hd_ in tqdm(range(10)):
    # 새로운 인덱스를 가진 시리즈 생성 후 리스트에 추가
    temp_series = iv_['impl_volatility'].copy()
    temp_series.index = temp_series.index + timedelta(days=hd_)
    iv_temp_list.append(temp_series)
iv_series = pd.concat(iv_temp_list)
iv_series.index = pd.DatetimeIndex(iv_series.index)
iv_series = iv_series[iv_series.index.year >= 2000]
iv_series = iv_series.groupby(level=0).mean()

opdat['div']  = div.loc[opdat['date']].values
opdat['iv_mean'] = iv_series.shift(1).fillna(0.1).loc[opdat['date']].values

df1 = opdat[['date','steps']]
df2 = rf.copy().dropna()
row_locs = df2.index.get_indexer(df1['date'], method='pad')
col_locs = df1['steps'].values-2
df2_values = df2.values
retrieved_values = df2_values[row_locs, col_locs]
opdat['df'] = retrieved_values

del(df1, row_locs,col_locs, df2_values,retrieved_values)

opdat = opdat[pd.to_datetime(opdat['date'])>= '20140101']


cumret = rets.loc['20140101':].rolling(window = 63).sum().sort_values(by = 'close').dropna()
bullday = price_.loc['20200402':].index[0] #model 25
bearday = rets.index[rets.index.get_loc(cumret.index[100]) - 63 + 6] #model 7
sideday = rets.index[rets.index.get_loc(abs(cumret['close']).sort_values().index[2]) - 63] #model4

noises = np.random.normal(0, 1, (63, 10000))
# plt.plot(sp_ret.loc[bullday:].iloc[:63].cumsum().values)
# plt.plot(sp_ret.loc[bearday:].iloc[:63].cumsum().values)
# plt.plot(sp_ret.loc[sideday:].iloc[:63].cumsum().values)
#MC, TGAN, QGAN, SGAN
situation_names = ['Bear', 'Sideway', 'Bull']
model_names = ['TGAN', 'QGAN', 'SGAN', '$\mathbf{Z}$']

model_nums = [7, 4, 25]
checkdays = [bearday, sideday, bullday]
fig, axs = plt.subplots(3, 4, figsize=(20, 12))
axs = axs.flatten()
i=0
for sday_ in range(3):
    # axs[i].set_xlim(-0.2, 0.2)
    # axs[i].set_ylim(0, 20)
    # axs[i].tick_params(axis='x', labelsize=12)
    # axs[i].tick_params(axis='y', labelsize=12)
    yticklist = []
    mn_ = model_nums[sday_]
    gir_dat = rets[rets.index <= checkdays[sday_]].iloc[-3520:]

    mean = rf.loc[checkdays[sday_], str(63)]
    cur_div = div.loc[checkdays[sday_]][0]
    std_dev = iv_series.loc[checkdays[sday_]]
    rpath = rets.loc[checkdays[sday_]:].iloc[:63].cumsum().values


    if i <4:
        axs[i].set_title(model_names[i],
                     fontsize=18,
                     # fontweight='bold',
                     color='darkslategray',
                     y=1.08)
    if i %4 == 0:
        axs[i].set_ylabel(situation_names[sday_],
                      fontsize=18 ,color='darkslategray',rotation='vertical', labelpad=30)

    for m_ in range(3):
        simulations = path_SP500[m_]

        simulations = simulations[mn_][:63]
        simulations = ((mean - cur_div) - ((std_dev ** 2) / 2)) + simulations * std_dev
        simulations = simulations.cumsum(0)
        yticklist.append(abs(simulations.min()))
        yticklist.append(abs(simulations.max()))
        random.seed(43)
        for p_ in random.sample(range(0, 10000), 100):
            axs[i].plot(simulations[:, p_], linewidth=0.8, alpha=0.4, color='grey')
        axs[i].plot(rpath, linewidth=1.4, color='red')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        if i < 4:
            axs[i].set_title(model_names[i],
                             fontsize=18,
                             # fontweight='bold',
                             color='darkslategray',
                             y=1.08)
        i += 1

    simulations = ((mean - cur_div) - ((std_dev ** 2) / 2)) + noises * std_dev
    simulations = simulations.cumsum(0)
    # simulations = noises.cumsum(0)
    yticklist.append(abs(simulations.min()))
    yticklist.append(abs(simulations.max()))
    random.seed(42)
    for p_ in random.sample(range(0, simulations.shape[1]), 100):
        axs[i].plot(simulations[:, p_], linewidth=0.8, alpha=0.4, color='grey')
    axs[i].plot(rpath, linewidth=1.4, color='red')
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    if i <4:
        axs[i].set_title(model_names[i],
                     fontsize=18,
                     # fontweight='bold',
                     color='darkslategray',
                     y=1.08)
    i += 1

    for t_ in range(4):
        axs[i - (t_ + 1)].set_ylim(-max(yticklist) * 0.7, max(yticklist) * 0.7)
        # axs[i - (t_ + 1)].set_ylim(-0.2, 0.2)


    gc.collect()

plt.tight_layout()
plt.subplots_adjust(
    top=0.9,      # Figure 상단에서 서브플롯 그리드 시작까지의 여백 (전체 높이의 10% 확보)
    bottom=0.05,  # Figure 하단 여백
    left=0.10,    # Figure 좌측에서 서브플롯 그리드 시작까지의 여백 (전체 너비의 15% 확보)
    right=0.90,   # Figure 우측 여백
    wspace=0.4,   # 서브플롯 가로 간격
    hspace=0.4)

plt.savefig('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_GEN/' + 'gen_sam.png', dpi=800, transparent=True)
plt.savefig('C:/Users/user/Downloads/HEDGEPAPER/REVISION/RES_GEN/' + 'gen_sam_low.png', transparent=True)
plt.close()



