from pykrakenapi import KrakenAPI
import krakenex
import requests
import urllib.parse
import hashlib
import hmac
import base64
from ta.momentum import stoch, rsi
from ta.trend import macd_diff
from ta.volatility import average_true_range
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
span = 100
delta = 1  # Days


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
    side_up = pd.Series(1, index=up_cross.index, dtype=int)  # Ensure integer dtype

    # crit3 = df[col1].shift(1) > df[col3].shift(1)
    crit4 = df[col1] < df[col3]
    down_cross = df[col1][crit4]
    side_down = pd.Series(-1, index=down_cross.index, dtype=int)  # Ensure integer dtype

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


def getDailyVol(close, span0, time_delta):
    """
    Daily Volatility Estimator [3.1]
    daily vol re-indexed to close
    Original df0 = df0[df0 > 0] does not include first day indexes
    was changed to df0 = df0[df0 >= 0]
    :param time_delta:
    :param close:
    :param span0:
    :return:
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=time_delta))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - time_delta], index=close.index[close.shape[0] - df0.shape[0]:]))
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - time_delta  # daily rets
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
the threshold, it indicates an upward shift in the mean value, and sPos is reset to zero, and 'i' is added to the
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


def kraken_request_t(uri_path, data, key, sec, max_retries=5, backoff_factor=60):
    headers = {'API-Key': key, 'API-Sign': get_kraken_signature(uri_path, data, sec)}
    retries = 0

    while retries < max_retries:
        try:
            req = requests.post((api_url + uri_path), headers=headers, data=data)
            req.raise_for_status()  # Raise an exception for 4XX/5XX status codes)
            return req

        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Connection error: {e}, retrying... ({retries + 1}/{max_retries})")
            retries += 1
            time.sleep(backoff_factor * retries)  # Exponential backoff

        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            break

    # If max retries are exceeded, return None or raise an error
    print("Max retries exceeded. Could not complete request.")
    return None


def get_private_balance():
    resp = kraken_request_t('/0/private/Balance', {
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
    if crypto_currency == 'ETH':
        crypto_currency = crypto_currency + '.F'
    elif crypto_currency == 'BTC':
        crypto_currency = 'XXBT'
    elif crypto_currency == 'DOGE':
        crypto_currency = 'XXDG'
    # DOT = DOT
    fiat_currency = fiat_currency + '.F'
    balance = get_private_balance()
    crypto_balance = float(balance['result'][crypto_currency])
    crypto_value = crypto_balance * closing_price
    fiat_balance = float(balance['result'][fiat_currency])
    if crypto_value < fiat_balance:
        return 'buy', crypto_balance, fiat_balance
    elif crypto_value >= fiat_balance:
        return 'sell', crypto_balance, fiat_balance
    else:
        return 'No balance found. Please select existing assets in your account.', crypto_balance, fiat_balance


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
    pair = crypto_currency + fiat_currency
    resp = kraken_request_t('/0/private/AddOrder', {
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
    resp = kraken_request_t('/0/private/CancelAll', {
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
    resp = kraken_request_t('/0/private/QueryOrders', {
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
    hdf['HF_EMA20'] = hdf['close'].rolling(20).mean()
    hdf['HF_EMA13'] = hdf['close'].rolling(13).mean()
    hdf['HF_EMA9'] = hdf['close'].rolling(9).mean()
    hdf['HF_EMA6'] = hdf['close'].rolling(6).mean()
    hdf['HF_EMA3'] = hdf['close'].rolling(3).mean()
    hdf['HF_Tr20'] = hdf.apply(lambda x: x['close'] - x['HF_EMA20'], axis=1)
    hdf['HF_Tr13'] = hdf.apply(lambda x: x['close'] - x['HF_EMA13'], axis=1)
    hdf['HF_Tr9'] = hdf.apply(lambda x: x['close'] - x['HF_EMA9'], axis=1)
    hdf['HF_Tr6'] = hdf.apply(lambda x: x['close'] - x['HF_EMA6'], axis=1)
    hdf['HF_Tr3'] = hdf.apply(lambda x: x['close'] - x['HF_EMA3'], axis=1)
    hdf['HF_vema3'] = hdf['volume'].rolling(3).mean()
    hdf['HF_Vtr3'] = hdf.apply(lambda x: x['volume'] - x['HF_vema3'], axis=1)
    hdf['HF_Volatility'] = getDailyVol(hdf['close'], span, delta)
    hdf['HF_MAV'] = hdf['HF_Volatility'].rolling(20).mean()
    hdf['HF_MAV_sig'] = hdf.apply(lambda x: x.HF_Volatility - x.HF_MAV, axis=1)
    hdf.fillna(0, inplace=True)
    mdf['MF_%K'] = stoch(mdf['high'], mdf['low'], mdf['close'], window=14, smooth_window=3, fillna=False)
    mdf['MF_%D'] = mdf['MF_%K'].rolling(3).mean()
    mdf['MF_St'] = mdf.apply(lambda x: x['MF_%K'] - x['MF_%D'], axis=1)
    mdf['MF_atr'] = average_true_range(mdf['high'], mdf['low'], mdf['close'], window=14, fillna=False)
    mdf['MF_rsi'] = rsi(mdf['close'], window=14, fillna=False)
    mdf['MF_atr'] = average_true_range(mdf['high'], mdf['low'], mdf['close'], window=14, fillna=False)
    mdf['MF_macd'] = macd_diff(mdf['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    mdf['MF_Volatility'] = getDailyVol(mdf['close'], span, delta)
    mdf['MF_MAV'] = mdf['MF_Volatility'].rolling(20).mean()
    mdf['MF_MAV_sig'] = mdf.apply(lambda x: x.MF_Volatility - x.MF_MAV, axis=1)
    mdf['MF_Vol_Vol'] = getDailyVol(mdf['MF_Volatility'], span, delta)
    mdf.fillna(0, inplace=True)
    ldf['datetime'] = pd.to_datetime(ldf['time'], unit='s')
    ldf['price'], ldf['ave'], ldf['upper'], ldf['lower'] = bbands(ldf['close'], window=20, numsd=2)
    ldf['bb_cross'] = simple_crossing(ldf, 'close', 'upper', 'lower')
    ldf['LF_Volatility'] = getDailyVol(ldf['close'], span, delta)
    ldf['LF_Vol_Vol'] = getDailyVol(ldf['LF_Volatility'], span, delta)
    t = getTEvents(ldf['close'], ldf['LF_Volatility'])
    ldf['event'] = ldf['LF_Volatility'].loc[t]
    ldf['event'] = ldf['LF_Volatility'][ldf['LF_Volatility'] > minRet]
    ldf['LF_MAV'] = ldf['LF_Volatility'].rolling(20).mean()
    ldf['LF_MAV_sig'] = ldf.apply(lambda x: x.LF_MAV - x.LF_Volatility, axis=1)
    ldf['LF_%K'] = stoch(ldf['high'], ldf['low'], ldf['close'], window=14, smooth_window=3, fillna=False)
    ldf['LF_%D'] = ldf['LF_%K'].rolling(3).mean()
    ldf['LF_roc10'] = ROC(ldf['close'], 10)
    ldf['LF_roc20'] = ROC(ldf['close'], 20)
    ldf['LF_mom10'] = MOM(ldf['close'], 10)
    ldf['LF_mom20'] = MOM(ldf['close'], 20)
    ldf['LF_mom30'] = MOM(ldf['close'], 30)
    ldf['LF_momi'] = ldf.apply(lambda x: x.LF_mom30 - x.LF_mom10, axis=1)
    ldf['LF_atr'] = average_true_range(ldf['high'], ldf['low'], ldf['close'], window=14, fillna=False)
    ldf['LF_vrsi'] = rsi(ldf['volume'], window=14, fillna=False)
    ldf['LF_vema13'] = ldf['volume'].rolling(13).mean()
    ldf['LF_vema6'] = ldf['volume'].rolling(6).mean()
    ldf['LF_Vtr13'] = ldf.apply(lambda x: x['volume'] - x['LF_vema13'], axis=1)
    ldf['LF_Vtr6'] = ldf.apply(lambda x: x['volume'] - x['LF_vema6'], axis=1)
    ldf['LF_EMA3'] = ldf['close'].rolling(3).mean()
    ldf['LF_Tr3'] = ldf.apply(lambda x: x['close'] - x['LF_EMA3'], axis=1)
    ldf['LF_EMA6'] = ldf['close'].rolling(6).mean()
    ldf['LF_Tr6'] = ldf.apply(lambda x: x['close'] - x['LF_EMA6'], axis=1)
    ldf['LF_EMA9'] = ldf['close'].rolling(9).mean()
    ldf['LF_Tr9'] = ldf.apply(lambda x: x['close'] - x['LF_EMA9'], axis=1)
    ldf['LF_EMA13'] = ldf['close'].rolling(13).mean()
    ldf['LF_Tr13'] = ldf.apply(lambda x: x['close'] - x['LF_EMA13'], axis=1)
    ldf['LF_EMA20'] = ldf['close'].rolling(20).mean()
    ldf['LF_Tr20'] = ldf.apply(lambda x: x['close'] - x['LF_EMA20'], axis=1)
    ldf['LF_St'] = ldf.apply(lambda x: x['LF_%K'] - x['LF_%D'], axis=1)
    ldf['LF_vema3'] = ldf['volume'].rolling(3).mean()
    ldf['LF_Vtr3'] = ldf.apply(lambda x: x['volume'] - x['LF_vema3'], axis=1)
    ldf['LF_vmacd'] = macd_diff(ldf['volume'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    ldf['LF_rsi'] = rsi(ldf['close'], window=14, fillna=False)
    ldf['LF_macd'] = macd_diff(ldf['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    ldf['LF_atr'] = average_true_range(ldf['high'], ldf['low'], ldf['close'], window=14, fillna=False)
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
    HF_candle_data = []
    HF_ema20_data = []
    HF_ema3_data = []
    for i in data_list:
        HF_candle_data.append({
            'x': i[0],
            'y': [i[1], i[2], i[3], i[4]]
        })
    for i in data_list:
        if i[8] != 0:
            HF_ema20_data.append({
                'x': i[0],
                'y': round(i[8], 4)
            })
    for i in data_list:
        if i[11] != 0:
            HF_ema3_data.append({
                'x': i[0],
                'y': round(i[9], 4)
            })
    return HF_candle_data, HF_ema20_data, HF_ema3_data


def mid_data(frame):
    """
    Same as above (high) FOR USE IN APEXCHARTS
    """
    front_df = frame.fillna(0)
    front_df['time'] = front_df['time'].apply(lambda x: (x * 1000) + 10800000)
    data_list = front_df.values.tolist()
    MF_candle_data = []
    for i in data_list:
        MF_candle_data.append({
            'x': i[0],
            'y': [i[1], i[2], i[3], i[4]]
        })
    return MF_candle_data


def low_data(frame):
    """
    Same as above (mid) FOR USE IN APEXCHARTS
    """
    front_df = frame.fillna(0)
    front_df['time'] = front_df['time'].apply(lambda x: (x * 1000) + 10800000)
    data_list = front_df.values.tolist()
    LF_candle_data = []
    LF_ave = []
    LF_upper = []
    LF_lower = []
    close = front_df.close.iloc[-1]
    LF_roc10 = front_df.LF_roc10.iloc[-1]
    for i in data_list:
        LF_candle_data.append({
            'x': i[0],
            'y': [i[1], i[2], i[3], i[4]]
        })
    for i in data_list:
        if i[10] != 0:
            LF_ave.append({
                'x': i[0],
                'y': round(i[10], 4)
            })
    for i in data_list:
        if i[11] != 0:
            LF_upper.append({
                'x': i[0],
                'y': round(i[11], 4)
            })
    for i in data_list:
        if i[12] != 0:
            LF_lower.append({
                'x': i[0],
                'y': round(i[12], 4)
            })
    return LF_candle_data, LF_ave, LF_upper, LF_lower, close, LF_roc10


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
