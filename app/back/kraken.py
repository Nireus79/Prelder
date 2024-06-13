from pykrakenapi import KrakenAPI
import krakenex
import requests
import urllib.parse
import hashlib
import hmac
import base64
from ta.trend import macd_diff
from ta.momentum import stoch, rsi
import numpy as np
import pandas as pd
import os
import time

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

api_url = "https://api.kraken.com"
api_key = os.environ['API_KEY_KRAKEN']
api_sec = os.environ['API_SEC_KRAKEN']
minRet = 0.01


def bbands(price, window=None, width=None, numsd=None):
    """ returns average, upper band, and lower band"""
    ave = price.rolling(window).mean()
    sd = price.rolling(window).std(ddof=0)
    if width:
        upband = ave * (1 + width)
        dnband = ave * (1 - width)
        return price, np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)
    if numsd:
        upband = ave + (sd * numsd)
        dnband = ave - (sd * numsd)
        return price, np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)


def simple_crossing(df, col1, col2, col3):
    # crit1 = df[col1].shift(1) < df[col2].shift(1)
    crit2 = df[col1] > df[col2]
    up_cross = df[col1][crit2]
    side_up = pd.Series(1, index=up_cross.index)

    # crit3 = df[col1].shift(1) > df[col3].shift(1)
    crit4 = df[col1] < df[col3]
    down_cross = df[col1][crit4]
    side_down = pd.Series(-1, index=down_cross.index)

    return pd.concat([side_up, side_down]).sort_index()


def crossing3(df, col1, col2, col3):
    crit1 = df[col1].shift(1) < df[col2].shift(1)
    crit2 = df[col1] > df[col2]
    up_cross = df[col1][crit1 & crit2]
    side_up = pd.Series(1, index=up_cross.index)

    crit3 = df[col1].shift(1) > df[col3].shift(1)
    crit4 = df[col1] < df[col3]
    down_cross = df[col1][crit3 & crit4]
    side_down = pd.Series(-1, index=down_cross.index)

    return pd.concat([side_up, side_down]).sort_index()


def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    roc = pd.Series(((M / N) * 100), name='ROC_' + str(n))
    return roc


def MOM(df, n):
    mom = pd.Series(df.diff(n), name='Momentum_' + str(n))
    return mom


def getDailyVol(close, span0, delta):
    """
    Daily Volatility Estimator [3.1]
    daily vol re-indexed to close
    Original df0 = df0[df0 > 0] does not include first day indexes
    was changed to df0 = df0[df0 >= 0]
    :param delta:
    :param close:
    :param span0:
    :return:
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(hours=delta))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - delta], index=close.index[close.shape[0] - df0.shape[0]:]))
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - delta  # daily rets
    except Exception as e:
        print(f'error: {e}\nplease confirm no duplicate indices')
    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0


def getTEvents(gRaw, h):
    """Symmetric CUSUM Filter [2.5.2.1]
    T events are the moments that a shift in
    the mean value of a measured quantity away from a target value.

    The getTEvents function is used to identify "T events" in a time series. T events are moments when there is a
    significant shift in the mean value of a measured quantity away from a target value. This function uses a
    symmetric CUSUM (cumulative sum) filter to detect these events. Here are the steps and the formula involved in
    this function:

Initialize Variables: Initialize three empty lists, tEvents, sPos, and sNeg to keep track of the identified events
and the cumulative sums of positive and negative changes.

Calculate Differences: Compute the differences between consecutive values of the logarithm of the input series gRaw.
This is done using np.log(gRaw).diff().dropna(). The diff variable now contains the log returns.

Iterate Through Differences: Iterate through the differences in log returns, starting from the second index (index 1)
because the first value in diff is NaN.

Cumulative Sum of Positive and Negative Changes:

sPos represents the cumulative sum of positive changes in log returns. sNeg represents the cumulative sum of negative
changes in log returns. At each step, sPos and sNeg are updated by adding the current value of the log return. The
float function is used to convert the cumulative sums to floats. The max(0., pos) and min(0., neg) functions ensure
that these cumulative sums never go below zero. If a positive cumulative sum becomes negative, it's set to zero,
and if a negative cumulative sum becomes positive, it's set to zero as well. Event Detection:

Check if sNeg goes below a threshold -h.loc[i]. The threshold is relative to h, which is a pandas Series containing
some form of volatility measurement. If sNeg crosses below the threshold, it indicates a downward shift in the mean
value, so sNeg is reset to zero, and the current index i is added to the tEvents list. Similarly, if sPos goes above
the threshold, it indicates an upward shift in the mean value, and sPos is reset to zero, and i is added to the
tEvents list. Return T Events: The function returns a pd.DatetimeIndex object containing the timestamps of the
identified T events.

In summary, the function detects T events in a time series by monitoring changes in log returns. When the cumulative
sums of these changes cross certain thresholds (h.loc[i]), it signifies a shift in the mean value,
and the corresponding timestamp is recorded as a T event. This is a common technique used in event-driven finance and
signal processing to detect significant changes in time series data.
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw.astype('float64')).diff().dropna()
    for i in diff.index[1:]:
        try:
            pos, neg = float(sPos + diff.loc[i]), float(sNeg + diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos + diff.loc[i], type(sPos + diff.loc[i]))
            print(sNeg + diff.loc[i], type(sNeg + diff.loc[i]))
            break
        sPos, sNeg = max(0., pos), min(0., neg)
        if sNeg < -h.loc[i]:  # .loc[i] # gives threshold relative to data['Volatility'].rolling(window).mean()
            sNeg = 0
            tEvents.append(i)
        elif sPos > h.loc[i]:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def get_kraken_signature(url, data, secret):
    post_data = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + post_data).encode()
    message = url.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sig_digest = base64.b64encode(mac.digest())
    return sig_digest.decode()


def kraken_request(uri_path, data, key, sec):
    headers = {'API-Key': key, 'API-Sign': get_kraken_signature(uri_path, data, sec)}
    req = requests.post((api_url + uri_path), headers=headers, data=data)
    return req


def get_private_balance():
    resp = kraken_request('/0/private/Balance', {
        "nonce": str(int(1000 * time.time()))
    }, api_key, api_sec)
    return resp.json()


def get_condition(crypto_currency, fiat_currency, closing_price):
    """
    Takes the private user's account balance and gives condition to buy or sell to the bot.
    If balance has less crypto value than fiat returns condition "buy"
    for the bot to look for good crypto buy conditions.
    If balance has more crypto value than fiat returns condition "sell"
    for the bot to look for good crypto sell conditions.
    :return: Conditions to buy/sell crypto.
    """
    fiat_currency = 'Z' + fiat_currency
    crypto_currency = 'X' + crypto_currency
    balance = get_private_balance()
    crypto_balance = float(balance['result'][crypto_currency])
    crypto_value = crypto_balance * closing_price
    fiat_balance = float(balance['result'][fiat_currency])
    if crypto_value < fiat_balance:
        return 'Buy', crypto_balance, fiat_balance
    elif crypto_value >= fiat_balance:
        return 'Sell', crypto_balance, fiat_balance
    else:
        log = 'No balance found. Please select existing assets in your account.'
        print(log)


def add_order(ordertype, cond, vol, price, crypto_currency, fiat_currency):
    """
    https://support.kraken.com/hc/en-us/articles/360022839631-Open-Orders
    :param crypto_currency:
    :param fiat_currency:
    :param ordertype: market / limit
    :param cond: buy / sell
    :param vol: vol of fiat or crypto
    :param price: order price
    :return: order response
    """
    pair = fiat_currency + crypto_currency
    resp = kraken_request('/0/private/AddOrder', {
        "nonce": str(int(1000 * time.time())),
        "ordertype": ordertype,
        "type": cond,
        "volume": vol,
        "pair": pair,
        "price": price
    }, key=api_key, sec=api_sec)
    return resp.json()


def cancel_order():
    """
    Cancel all orders
    :return:
    """
    resp = kraken_request('/0/private/CancelAll', {
        "nonce": str(int(1000 * time.time()))
    }, api_key, api_sec)
    """
    or cancel specific order (needs txid)
    resp = kraken_request('/0/private/CancelOrder', {
    "nonce": str(int(1000*time.time())),
    "txid": "OG5V2Y-RYKVL-DT3V3B" EXAMPLE!!! (Give txid as argument)
    }, api_key, api_sec)
    """
    return resp.json()


def get_order_info(txid):
    resp = kraken_request('/0/private/QueryOrders', {
        "nonce": str(int(1000 * time.time())),
        "txid": txid,
        "trades": True
    }, api_key, api_sec)
    return resp.json()


def time_stamp():
    """
    Takes unix time and gives datetime format.
    :return: Readable local datetime format.
    """
    curr_time = time.localtime()
    curr_clock = time.strftime("%Y-%m-%d %H:%M:%S", curr_time)
    return curr_clock


def indicators(ldf, mdf, hdf):
    hdf['EMA20'] = hdf['close'].rolling(20).mean()
    hdf['EMA3'] = hdf['close'].rolling(3).mean()
    hdf['TrD20'] = hdf.apply(lambda x: x['close'] - x['EMA20'], axis=1)
    hdf['TrD3'] = hdf.apply(lambda x: x['close'] - x['EMA3'], axis=1)
    hdf.fillna(0, inplace=True)
    mdf['%K'] = stoch(mdf['high'], mdf['low'], mdf['close'], window=14, smooth_window=3, fillna=False)
    mdf['%D'] = mdf['%K'].rolling(3).mean()
    mdf['mac4'] = macd_diff(mdf['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    mdf.fillna(0, inplace=True)
    ldf['datetime'] = pd.to_datetime(ldf['time'], unit='s')
    ldf['price'], ldf['ave'], ldf['upper'], ldf['lower'] = bbands(ldf['close'], window=20, numsd=2)
    ldf['bb_l'] = ldf.apply(lambda x: (x['upper'] - x['close']) / (x['close'] - x['lower']) if x['close'] != x['lower']
    else 0, axis=1)
    ldf['EMA6'] = ldf['close'].rolling(6).mean()
    ldf['Tr6'] = ldf.apply(lambda x: x['close'] - x['EMA6'], axis=1)
    ldf['bb_cross'] = simple_crossing(ldf, 'close', 'upper', 'lower')
    ldf['Volatility'] = getDailyVol(ldf['close'], 100, 24)
    ldf['Vol_Volatility'] = getDailyVol(ldf['Volatility'], 100, 24)
    ldf['roc30'] = ROC(ldf['close'], 30)
    ldf['rsi'] = rsi(ldf['close'], window=14, fillna=False)
    ldf['mom10'] = MOM(ldf['close'], 10)
    t = getTEvents(ldf['close'], ldf['Volatility'])
    ldf['event'] = ldf['Volatility'].loc[t]
    ldf['event'] = ldf['Volatility'][ldf['Volatility'] > minRet]
    # ldf['tEvent'] = ldf.apply(lambda x: True if x.datetime in t else False, axis=1)
    ldf.fillna(0, inplace=True)
    return ldf, mdf, hdf


def high_data(frame):
    """
        FOR USE IN APEXCHARTS
        Takes a pandas dataframe and creates a tuple of lists with
        the needed values for the apexcharts candle charts.
        Axis X values are all multiplied by 1000.
        JavaScript Date object however uses milliseconds since 1 January 1970 UTC.
        Therefore, you should multiply your timestamps by 1000 prior to assign the data to the chart configuration.
        :param frame: The modified pandas dataframe from high_frame_indicators.
        :return: json and last running indicators to be displayed.
    """
    front_df = frame.fillna(0)
    front_df['time'] = front_df['time'].apply(lambda x: (x * 1000) + 10800000)  # *1000 javascript time + 3hours
    data_list = front_df.values.tolist()
    candle_data = []
    ema20_data = []
    ema3_data = []
    TrD20 = round(data_list[-1][10], 4)
    TrD3 = round(data_list[-1][11], 4)
    for i in data_list:
        candle_data.append({
            'x': i[0],
            'y': [i[1], i[2], i[3], i[4]]
        })
    for i in data_list:
        if i[8] != 0:
            ema20_data.append({
                'x': i[0],
                'y': round(i[8], 4)
            })
    for i in data_list:
        if i[9] != 0:
            ema3_data.append({
                'x': i[0],
                'y': round(i[9], 4)
            })
    return candle_data, ema20_data, ema3_data, TrD20, TrD3


def mid_data(frame):
    """
    Same as above (high) FOR USE IN APEXCHARTS
    """
    front_df = frame.fillna(0)
    front_df['time'] = front_df['time'].apply(lambda x: (x * 1000) + 10800000)
    data_list = front_df.values.tolist()
    candle_data = []
    stoch_k_data = round(data_list[-1][8], 4)
    stoch_d_data = round(data_list[-1][9], 4)
    macd = round(data_list[-1][10], 4)
    for i in data_list:
        candle_data.append({
            'x': i[0],
            'y': [i[1], i[2], i[3], i[4]]
        })
    return candle_data, stoch_k_data, stoch_d_data, macd


def low_data(frame):
    """
    Same as above (mid) FOR USE IN APEXCHARTS
    """
    front_df = frame.fillna(0)
    front_df['time'] = front_df['time'].apply(lambda x: (x * 1000) + 10800000)
    data_list = front_df.values.tolist()
    candle_data = []
    ave = []
    upper = []
    lower = []
    Tr6 = round(data_list[-1][15], 4)
    volatility = round(data_list[-1][17], 4)
    for i in data_list:
        candle_data.append({
            'x': i[0],
            'y': [i[1], i[2], i[3], i[4]]
        })
    for i in data_list:
        if i[10] != 0:
            ave.append({
                'x': i[0],
                'y': round(i[10], 4)
            })
    for i in data_list:
        if i[11] != 0:
            upper.append({
                'x': i[0],
                'y': round(i[11], 4)
            })
    for i in data_list:
        if i[12] != 0:
            lower.append({
                'x': i[0],
                'y': round(i[12], 4)
            })
    return candle_data, ave, upper, lower, Tr6, volatility


class Api:
    """
    crypto: the name of crypto
    fiat: the name of monetary currency
    interval: candle in min
    frame_len: the length of frame (how many candles)
    """

    def __init__(self, crypto, fiat, interval, frame_len):
        self.frame_len = frame_len
        self.crypto = crypto
        self.fiat = fiat
        self.interval = interval  # candle in minutes
        self.pair = self.crypto + self.fiat
        self.candle = interval * 60  # candle in seconds
        self.last_candle = int(time.time()) - self.candle  # 1 candle in seconds
        self.frame = int(time.time()) - self.candle * self.frame_len
        self.key = os.environ['API_KEY_KRAKEN']
        self.sec = os.environ['API_SEC_KRAKEN']

    def get_frame(self):
        api = krakenex.API()
        k = KrakenAPI(api)
        ohlc, last = k.get_ohlc_data(self.pair, interval=self.interval, since=self.frame)
        return ohlc.iloc[::-1]  # reverse rows

# i24h = Api('DOT', 'EUR', 1440, 3)
# i4h = Api('DOT', 'EUR', 240, 16)
# i30m = Api('DOT', 'EUR', 30, 700)
#
#
# i24h_frame = i24h.get_frame()
# i4h_frame = i4h.get_frame()
# i30m_frame = i30m.get_frame()
#
# hf = high_frame_indicators(i24h_frame)
# mf = mid_frame_indicators(i4h_frame)
# lf = low_frame_indicators(i30m_frame)
# print('hf---------------------------')
# print(hf)
# print('mf---------------------------')
# print(mf)
# print('lf---------------------------')
# print(lf)