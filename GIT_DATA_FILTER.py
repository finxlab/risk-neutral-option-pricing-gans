import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

price_ = pd.read_csv('data/raw/sp500.csv')
price_.index = pd.DatetimeIndex(price_['Date'])
price_ = price_[['Close']]
price_.columns = ['close']
price_ = price_[price_.index >= '2000-01-01']


opdat = joblib.load('data/raw/opdat_raw.pkl')
opdat['impl_volatility'] = opdat['impl_volatility']/np.sqrt(252)

rf = pd.read_csv('data/processed/rfcurve.csv',index_col = 0)
rf.index = pd.DatetimeIndex(rf.index)
rf = np.log((1+rf))/252

div = pd.read_csv('data/raw/sp_div.csv', index_col = 0)
div.index = pd.DatetimeIndex(div.index)
div = div.sort_index()
div = div[div.index.year>=2000]
div = np.log(1+div/100)/252
div = div.asfreq('D').fillna(method = 'ffill')

def find_nearest_date(target_date):
    target_date = np.datetime64(target_date)
    idx = np.searchsorted(date_range_np, target_date, side='right') - 1
    return idx

def data_filtering(df, price):

    df = df.drop(['secid','symbol_flag','last_date','volume','open_interest','gamma','vega','theta',
                  'cfadj','am_settlement','contract_size','ss_flag','forward_price','expiry_indicator','root','suffix','cusip',
                  'sic','index_flag','exchange_d','class','issue_type','industry_group','issuer','div_convention', 'exercise_style',
                  'am_set_flag'], axis=1)
    df = df.loc[df['impl_volatility'].dropna().index]

    df['price'] = (df['best_bid'] + df['best_offer']) / 2
    df['index_price'] = price.loc[df['date']].values
    df['Edate'] = (pd.to_datetime(df['exdate']) - pd.to_datetime(df['date'])).dt.days
    df['spread'] = (df['best_offer'] - df['best_bid']) / df['price']
    # moneyness
    mask_c = df['cp_flag'] == 'C'
    df.loc[mask_c, 'moneyness'] = (df.loc[mask_c, 'index_price'] * 1000) / df.loc[mask_c, 'strike_price']
    mask_p = df['cp_flag'] == 'P'
    df.loc[mask_p, 'moneyness'] = df.loc[mask_p, 'strike_price'] / (df.loc[mask_p, 'index_price'] * 1000)
    # arbitrage
    mask_c = df['cp_flag'] == 'C'
    df.loc[mask_c, 'arb'] = df.loc[mask_c, 'price'] - (
                df.loc[mask_c, 'index_price'] - df.loc[mask_c, 'strike_price'] / 1000)
    mask_p = df['cp_flag'] == 'P'
    df.loc[mask_p, 'arb'] = df.loc[mask_p, 'price'] - (
                df.loc[mask_p, 'strike_price'] / 1000 - df.loc[mask_p, 'index_price'])

    df = df[df['Edate'] >= 6]
    df = df[df['Edate'] <= 91]
    # lower price
    df = df[df['price'] > 0.125]
    # spread
    df = df[df['spread'] <= 0.05]
    # arbitrage
    df = df[df['arb'] > 0]

    df = df[df['moneyness']<=1.15]
    df = df[df['moneyness']>=0.85]



    return df


opdat = data_filtering(opdat, price_)

price_index = price_.index
start_dates_shifted = pd.to_datetime(opdat['date']) + timedelta(days=1)
start_locs = price_index.searchsorted(start_dates_shifted)
end_locs = price_index.searchsorted(pd.to_datetime(opdat['exdate']))
opdat['steps'] = end_locs - start_locs

iv_ = opdat[['date', 'impl_volatility']].set_index('date')
iv_.index = pd.DatetimeIndex(iv_.index)

iv_temp_list = []

for hd_ in range(10):
    temp_series = iv_['impl_volatility'].copy()
    temp_series.index = temp_series.index + timedelta(days=hd_)
    iv_temp_list.append(temp_series)
iv_series = pd.concat(iv_temp_list)
iv_series.index = pd.DatetimeIndex(iv_series.index)
iv_series = iv_series.groupby(level=0).mean()

opdat['div'] = div.loc[opdat['date']].values
opdat['iv_mean'] = iv_series.shift(1).loc[opdat['date']].values

df1 = opdat[['date', 'steps']]
df2 = rf.copy().dropna()
row_locs = df2.index.get_indexer(df1['date'], method='pad')
col_locs = df1['steps'].values - 2
df2_values = df2.values
retrieved_values = df2_values[row_locs, col_locs]
opdat['df'] = retrieved_values


date_range = pd.date_range(start = '2014-01-01', end='2023-09-01', freq='3MS')
date_range_np = date_range.values


model_idx = opdat[['date']].copy()
model_idx['model_idx'] = pd.to_datetime(model_idx['date']).apply(find_nearest_date)

opdat['model_idx'] = model_idx['model_idx']

joblib.dump(opdat, 'data/processed/opdat.pkl')