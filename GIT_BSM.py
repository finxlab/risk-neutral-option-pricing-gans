import pandas as pd
import numpy as np
import joblib
from scipy.stats import norm
import warnings
warnings.filterwarnings(action='ignore')




def bsm_vectorized(S, K, T, r, q, sigma, option_type):
    # 이 함수는 Series 또는 배열을 입력으로 받습니다.

    # 만기(T)가 0 이하인 경우를 처리 (np.where를 사용해 벡터화)
    d1 = np.where(T > 0, (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)), np.nan)
    d2 = np.where(T > 0, d1 - sigma * np.sqrt(T), np.nan)

    # option_type에 따라 콜/풋 가격 계산
    call_price = np.where(option_type == 'C', S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
                          np.nan)
    put_price = np.where(option_type == 'P', K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1),
                         np.nan)

    # option_type에 따라 콜/풋 델타 계산
    call_delta = np.where(option_type == 'C', np.exp(-q * T) * norm.cdf(d1), np.nan)
    put_delta = np.where(option_type == 'P', np.exp(-q * T) * (norm.cdf(d1) - 1), np.nan)

    # 최종 결과 합치기
    final_price = np.where(option_type == 'C', call_price, put_price)
    final_delta = np.where(option_type == 'C', call_delta, put_delta)

    return final_price, final_delta



def bsm_vectorized_mu(S, K, T, r, q, sigma, option_type, mu):
    # 이 함수는 Series 또는 배열을 입력으로 받습니다.

    # 만기(T)가 0 이하인 경우를 처리 (np.where를 사용해 벡터화)
    d1 = np.where(T > 0, (np.log(S / K) + (mu - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)), np.nan)
    d2 = np.where(T > 0, d1 - sigma * np.sqrt(T), np.nan)

    # option_type에 따라 콜/풋 가격 계산
    call_price = np.where(option_type == 'C', S * np.exp( (mu-q-r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
                          np.nan)
    put_price = np.where(option_type == 'P', K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp( (mu-q-r) * T) * norm.cdf(-d1),
                         np.nan)

    # option_type에 따라 콜/풋 델타 계산
    call_delta = np.where(option_type == 'C', np.exp(-q * T) * norm.cdf(d1), np.nan)
    put_delta = np.where(option_type == 'P', np.exp(-q * T) * (norm.cdf(d1) - 1), np.nan)

    # 최종 결과 합치기
    final_price = np.where(option_type == 'C', call_price, put_price)
    final_delta = np.where(option_type == 'C', call_delta, put_delta)

    return final_price, final_delta


price_ = pd.read_csv('data/raw/sp500.csv')
price_.index = pd.DatetimeIndex(price_['Date'])
price_ = price_[['Close']]
price_.columns = ['close']
price_ = price_[price_.index >= '2000-01-01']

div = pd.read_csv('data/raw/sp_div.csv', index_col = 0)
div.index = pd.DatetimeIndex(div.index)
div = div.sort_index()
div = div[div.index.year>=2000]
div = np.log(1+div/100)/252
div = div.asfreq('D').fillna(method = 'ffill')


rf = pd.read_csv('data/processed/rfcurve.csv',index_col = 0)
rf.index = pd.DatetimeIndex(rf.index)
rf = rf.asfreq('D').fillna(method = 'ffill')
rf = np.log((1+rf))/252


opdat = joblib.load('data/processed/opdat.pkl')


rets = np.log(price_/price_.shift(1)).dropna()
r_length = len(rets[rets.index <= '2014-01-01'])
price_ = price_.asfreq('D').fillna(method = 'ffill')



prcs_, deltas_ = bsm_vectorized(opdat['index_price'].values, opdat['strike_price'].values/1000,(opdat['steps']/252).values,
                      opdat['df'].values * 252, opdat['div'].values * 252, opdat['iv_mean'].values * np.sqrt(252), opdat['cp_flag'].values)


prcs_ = pd.DataFrame(prcs_)
prcs_.index = opdat.index
prcs_.columns = ['BSM']

deltas_ = pd.DataFrame(deltas_)
deltas_.index = opdat.index
deltas_.columns = ['BSM']

joblib.dump(prcs_[opdat['cp_flag'] == 'C'], 'results/pricing_results/Price/BSM/CALL/res.pkl')
joblib.dump(prcs_[opdat['cp_flag'] == 'P'], 'results/pricing_results/Price/BSM/PUT/res.pkl')

joblib.dump(deltas_[opdat['cp_flag'] == 'C'], 'results/pricing_results/Delta/BSM/CALL/res.pkl')
joblib.dump(deltas_[opdat['cp_flag'] == 'P'], 'results/pricing_results/Delta/BSM/PUT/res.pkl')

