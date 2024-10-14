import logging
import time
import threading
import winsound
from app.back.kraken import (Api, time_stamp, add_order, cancel_order, get_condition, high_data, mid_data, low_data,
                             indicators, minRet)
from app.back.spring import activation, check
from sklearn.preprocessing import Normalizer, normalize
import pandas as pd
import numpy as np
import pickle
import joblib

pd.set_option('future.no_silent_downcasting', True)
logging.basicConfig(level=logging.INFO)

condition = None
crypto_currency = None
fiat_currency = None
crypto_balance = 0
fiat_balance = 0

limits = {
    'ETHEUR':{'limit': None, 'stop': None, 'timestamp': 0},
    'DOTEUR':{'limit': None, 'stop': None, 'timestamp': 0},
    'BTCEUR':{'limit': None, 'stop': None, 'timestamp': 0}
          }

ret = 0
roc10 = 0
order_type = 'market'
low_closing_price = 0
low_limit = None
low_stop = None

market_return = 0.04
market_reset = 0.03
kraken_fee = 0.004001

high_chart_data = None
mid_chart_data = None
low_chart_data = None
high_ema20 = None
high_ema3 = None
low_ave = None
low_upper = None
low_lower = None
event = 0
bb_cross = None
prime_prediction = None
meta_prediction = None
first_candle_time = 0

log = 'Please set control parameters to start.'
logs = []
trades = []
break_event = threading.Event()

def normalizer(data):
    """
    Normalization is a good technique to use when you do not know the distribution of your data or when you know the
    distribution is not Gaussian (a bell curve). Normalization is useful when your data has varying scales and the
    algorithm you are using does not make assumptions about the distribution of your data, such as k-nearest
    neighbors and artificial neural networks. data: :return:
    """
    scaler = Normalizer().fit(data)
    normalized = pd.DataFrame(scaler.fit_transform(data), index=data.index)
    normalized.columns = data.columns
    return normalized


def signal_handler():
    """
    Setting exit event to allow process end
    within while loops before program stop.
    """
    break_event.set()


def beeper(cond):
    """
    Triggers sound signal user notifications on important trading events.
    :param cond: sell / buy / start (events)
    """
    if cond == 'buy':
        for _ in range(5):
            winsound.Beep(440, 500)
            time.sleep(0.5)
    elif cond == 'sell':
        for _ in range(5):
            winsound.Beep(1000, 500)
            time.sleep(0.1)
    elif cond == 'start':
        winsound.Beep(440, 2000)
    elif cond == 'break':
        winsound.Beep(240, 2000)


def chart_data(p, high_frame, mid_frame, low_frame):
    global high_chart_data, high_ema20, high_ema3,\
        mid_chart_data, low_chart_data, low_ave, low_upper, low_lower, low_limit, low_stop, low_closing_price
    limit_data = []
    stop_data = []
    high_candles, ema20, ema3 = high_data(high_frame)
    mid_candles = mid_data(mid_frame)
    low_candles, ave, upper, lower, low_closing_price = low_data(low_frame)
    for i in low_candles:
        limit_data.append({
            'x': i['x'],
            'y': limits[p]['limit']
        })
        stop_data.append({
            'x': i['x'],
            'y': limits[p]['stop']
        })
    high_chart_data = high_candles[-20:]
    high_ema20 = ema20[-20:]
    high_ema3 = ema3[-20:]
    mid_chart_data = mid_candles[-20:]
    low_chart_data = low_candles[-20:]
    low_ave = ave[-20:]
    low_lower = lower[-20:]
    low_upper = upper[-20:]
    low_limit = limit_data[-20:]
    low_stop = stop_data[-20:]


def log_action(message):
    logging.info(message)
    logs.append(message)
    logs.append('<br>')
    return message

def sell_evaluation(asset, high_frame_indicated, mid_frame_indicated, low_frame_indicated, pms, mms):
    global prime_prediction, meta_prediction
    TrD9 = high_frame_indicated.iloc[-1]['TrD9']
    TrD6 = high_frame_indicated.iloc[-1]['TrD6']
    TrD3 = high_frame_indicated.iloc[-1]['TrD3']
    St4H = mid_frame_indicated.iloc[-1]['St4H']
    macd4 = mid_frame_indicated.iloc[-1]['4Hmacd']
    atr4 = mid_frame_indicated.iloc[-1]['4H_atr']
    rsi4 = mid_frame_indicated.iloc[-1]['4H_rsi']
    K4 = mid_frame_indicated.iloc[-1]['4H_rsi']
    D = low_frame_indicated.iloc[-1]['%D']
    MAV_signal = low_frame_indicated.iloc[-1]['MAV_signal']
    atr = low_frame_indicated.iloc[-1]['atr']
    roc = low_frame_indicated.iloc[-1]['roc10']
    bbc = low_frame_indicated.iloc[-1]['bb_cross']
    Vtr3 = low_frame_indicated.iloc[-1]['Vtr3']
    Tr13 = low_frame_indicated.iloc[-1]['Tr13']
    roc20 = low_frame_indicated.iloc[-1]['roc20']
    Volatility = low_frame_indicated.iloc[-1]['Volatility']
    momi = low_frame_indicated.iloc[-1]['momi']
    vrsi = low_frame_indicated.iloc[-1]['vrsi']
    if asset == 'ETH':
        featuresS = [[TrD9, TrD6, TrD3, St4H, MAV_signal, D, atr, roc]]
        featuresS = normalize(featuresS)
        prime_predictionS = pms.predict(featuresS)
        featuresMS = featuresS
        featuresMS = np.insert(featuresMS, len(featuresS), prime_predictionS)
        meta_predictionS = mms.predict([featuresMS])
        prime_prediction, meta_prediction = prime_predictionS[0], meta_predictionS[0]
        return prime_predictionS[0], meta_predictionS[0]
    if asset == 'BTC':
        featuresS = [[TrD9, TrD3, macd4, rsi4, K4, D, Vtr3, roc, vrsi]]
        featuresS = normalize(featuresS)
        featuresS = np.insert(featuresS, len(featuresS[0]), bbc)
        prime_predictionS = pms.predict([featuresS])
        featuresMS = featuresS
        featuresMS = np.insert(featuresMS, len(featuresS), prime_predictionS)
        meta_predictionS = mms.predict([featuresMS])
        prime_prediction, meta_prediction = prime_predictionS[0], meta_predictionS[0]
        return prime_predictionS[0], meta_predictionS[0]
    if asset == 'DOT':
        featuresS = [[TrD9, TrD6, atr4, Tr13, roc20, roc, Volatility, momi]]
        featuresS = normalize(featuresS)
        featuresS = np.insert(featuresS, len(featuresS[0]), bbc)
        prime_predictionS = pms.predict([featuresS])
        featuresMS = featuresS
        featuresMS = np.insert(featuresMS, len(featuresS), prime_predictionS)
        meta_predictionS = mms.predict([featuresMS])
        prime_prediction, meta_prediction = prime_predictionS[0], meta_predictionS[0]
        return prime_predictionS[0], meta_predictionS[0]


def buy_evaluation(asset, high_frame_indicated, mid_frame_indicated, low_frame_indicated, pmb, mmb):
    TrD20 = high_frame_indicated.iloc[-1]['TrD20']
    TrD13 = high_frame_indicated.iloc[-1]['TrD13']
    TrD9 = high_frame_indicated.iloc[-1]['TrD9']
    TrD6 = high_frame_indicated.iloc[-1]['TrD6']
    TrD3 = high_frame_indicated.iloc[-1]['TrD3']
    atr4H = mid_frame_indicated.iloc[-1]['atr4H']
    macd4 = mid_frame_indicated.iloc[-1]['4Hmacd']
    rsi4 = mid_frame_indicated.iloc[-1]['4H_rsi']
    Vtr13 = low_frame_indicated.iloc[-1]['Vtr13']
    Vtr6 = low_frame_indicated.iloc[-1]['Vtr6']
    roc = low_frame_indicated.iloc[-1]['roc10']
    MAV_signal = low_frame_indicated.iloc[-1]['MAV_signal']
    StD = low_frame_indicated.iloc[-1]['StD']
    D = low_frame_indicated.iloc[-1]['%D']
    vmacd = low_frame_indicated.iloc[-1]['vmacd']
    rsi = low_frame_indicated.iloc[-1]['rsi']
    macd = low_frame_indicated.iloc[-1]['macd']
    MAV = low_frame_indicated.iloc[-1]['MAV']
    mom20 = low_frame_indicated.iloc[-1]['mom20']
    atr = low_frame_indicated.iloc[-1]['atr']
    vrsi = low_frame_indicated.iloc[-1]['vrsi']
    global prime_prediction, meta_prediction
    if asset == 'ETH':
        featuresB = [[TrD20, TrD3, atr4H, Vtr13, Vtr6, MAV_signal, vrsi, roc]]
        featuresB = normalize(featuresB)
        prime_predictionB = pmb.predict(featuresB)
        featuresMB = featuresB
        featuresMB = np.insert(featuresMB, len(featuresMB), prime_predictionB)
        meta_predictionB = mmb.predict([featuresMB])
        prime_prediction, meta_prediction = prime_predictionB[0], meta_predictionB[0]
        return prime_predictionB[0], meta_predictionB[0]
    if asset == 'BTC':
        featuresB = [[TrD20, TrD13, TrD6, TrD3, rsi4, StD, D, vmacd, rsi, vrsi]]
        featuresB = normalize(featuresB)
        prime_predictionB = pmb.predict(featuresB)
        featuresMB = featuresB
        featuresMB = np.insert(featuresMB, len(featuresMB), prime_predictionB)
        meta_predictionB = mmb.predict([featuresMB])
        prime_prediction, meta_prediction = prime_predictionB[0], meta_predictionB[0]
        return prime_predictionB[0], meta_predictionB[0]
    if asset == 'DOT':
        featuresB = [[TrD20, TrD13, TrD9, TrD6, TrD3, macd4, macd, MAV, mom20, atr]]
        featuresB = normalize(featuresB)
        prime_predictionB = pmb.predict(featuresB)
        featuresMB = featuresB
        featuresMB = np.insert(featuresMB, len(featuresMB), prime_predictionB)
        meta_predictionB = mmb.predict([featuresMB])
        prime_prediction, meta_prediction = prime_predictionB[0], meta_predictionB[0]
        return prime_predictionB[0], meta_predictionB[0]


def ret_evaluation(asset, high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr):
    TrD20 = high_frame_indicated.iloc[-1]['TrD20']
    TrD13 = high_frame_indicated.iloc[-1]['TrD13']
    TrD9 = high_frame_indicated.iloc[-1]['TrD9']
    TrD6 = high_frame_indicated.iloc[-1]['TrD6']
    TrD3 = high_frame_indicated.iloc[-1]['TrD3']
    macd4 = mid_frame_indicated.iloc[-1]['4Hmacd']
    rsi4 = mid_frame_indicated.iloc[-1]['4H_rsi']
    K4 = mid_frame_indicated.iloc[-1]['4H_rsi']
    Tr6 = low_frame_indicated.iloc[-1]['Tr6']
    D = low_frame_indicated.iloc[-1]['%D']
    Vtr3 = low_frame_indicated.iloc[-1]['Vtr3']
    vmacd = low_frame_indicated.iloc[-1]['vmacd']
    atr4 = mid_frame_indicated.iloc[-1]['4H_atr']
    Vtr13 = low_frame_indicated.iloc[-1]['Vtr13']
    Vtr6 = low_frame_indicated.iloc[-1]['Vtr6']
    MAV_signal = low_frame_indicated.iloc[-1]['MAV_signal']
    roc = low_frame_indicated.iloc[-1]['roc10']
    vrsi = low_frame_indicated.iloc[-1]['vrsi']
    macd = low_frame_indicated.iloc[-1]['macd']
    MAV = low_frame_indicated.iloc[-1]['MAV']
    mom20 = low_frame_indicated.iloc[-1]['mom20']
    atr = low_frame_indicated.iloc[-1]['atr']
    if asset == 'ETH':
        features = [[TrD20, TrD3, atr4, Vtr13, Vtr6, MAV_signal,vrsi, roc]]
        features = normalize(features)
        ret_prediction = mr.predict(features)
        return ret_prediction[0], 1, 1
    elif asset == 'BTC':
        features = [[TrD9, TrD3, macd4, rsi4, K4, Tr6, D, Vtr3, vmacd, roc, vrsi]]
        features = normalize(features)
        ret_prediction = mr.predict(features)
        return ret_prediction[0], 1, 1
    elif asset == 'DOT':
        features = [[TrD20, TrD13, TrD9, TrD6, TrD3, macd4, macd, MAV, mom20, atr]]
        features = normalize(features)
        ret_prediction = mr.predict(features)
        return ret_prediction[0], 1, 1

def set_ptsl(pair, price, s, t, pt, sl):
    global limits
    if s == 'M':
        limits[pair]['limit'] = price
        limits[pair]['stop'] = price * (1 - (abs(roc10) / 100) * sl)
        limits[pair]['timestamp'] = t
        pickle.dump(limits, open('limits.pkl', 'wb'))
    else:
        limits[pair]['limit'] = price * (1 + (ret * pt))
        limits[pair]['stop'] = price * (1 - (ret * sl))
        limits[pair]['timestamp'] = t
        pickle.dump(limits, open('limits.pkl', 'wb'))


def reset_ptsl(pair):
    global limits
    limits[pair]['limit'] = None
    limits[pair]['stop'] = None
    limits[pair]['timestamp'] = 0
    pickle.dump(limits, open('limits.pkl', 'wb'))

def action(mode, crypto, fiat, price):
    global log, condition
    if mode == 'simulator':
        log = log_action('{} Simulating {} at {}'.format(time_stamp(), condition, price))
        trades.append(log)
        if condition == 'buy':
            condition = 'sell'
        elif condition == 'sell':
            condition = 'buy'
    elif mode == 'consulting':
        log = log_action('{} Consulting {} at {}.'.format(time_stamp(), condition, price))
        trades.append(log)
    elif mode == 'trading':
        log = log_action('{} {} at {}.'.format(time_stamp(),condition, price))
        trades.append(log)
        if condition == 'buy':
            asset_vol = (fiat_balance - fiat_balance * kraken_fee) / price
            tx = add_order(order_type, condition, asset_vol, price, crypto, fiat)
            trades.append(tx)
            log = log_action(tx)
        elif condition == 'sell':
            asset_vol = crypto_balance
            tx = add_order(order_type, condition, asset_vol, price, crypto, fiat)
            trades.append(tx)
            log = log_action(tx)
            if not tx['error']:
                reset_ptsl(crypto + fiat)


def reset_predictions():
    global prime_prediction, meta_prediction, ret
    prime_prediction = None
    meta_prediction = None
    ret = 0

def raw_data(crypto, fiat):
    i30m = Api(crypto, fiat, 30, 700)
    i4H = Api(crypto, fiat, 240, 100)
    i24H = Api(crypto, fiat, 1440, 100)
    return i30m.get_frame(), i4H.get_frame(), i24H.get_frame()

def multiPrelderbot(mode, assets):
    global condition, limits, ret, log, crypto_balance, fiat_balance, event, bb_cross, \
        prime_prediction, meta_prediction, roc10, first_candle_time, crypto_currency, fiat_currency
    licence = True  # check()['license_active']  # TODO activate licence check
    if licence:
        for crypto_currency, fiat_currency, pmb, mmb, pms, mms, mr in assets:
            pair = crypto_currency+fiat_currency
            log = log_action('Evaluating {}-{}.'.format(crypto_currency, fiat_currency))
            low_frame, mid_frame, high_frame = raw_data(crypto_currency, fiat_currency)
            low_frame_indicated, mid_frame_indicated, high_frame_indicated = indicators(low_frame, mid_frame, high_frame)
            chart_data(pair, high_frame_indicated, mid_frame_indicated, low_frame_indicated)
            closing_price = low_frame_indicated.iloc[-1]['close']
            event = low_frame_indicated.iloc[-1]['event']
            bb_cross = low_frame_indicated.iloc[-1]['bb_cross']
            roc10 = low_frame_indicated.iloc[-1]['roc10']
            new_timestamp = first_candle_time = low_frame.iloc[-1]['time']
            condition, crypto_balance, fiat_balance = get_condition(crypto_currency, fiat_currency, closing_price)
            limit, stop = limits[pair]['limit'], limits[pair]['stop']
            if mode == 'simulator':
                condition = 'buy'
            log = log_action('Event is: {}. BB crossing is: {}. Condition is: {}'.format(event, bb_cross, condition))
            if condition == 'sell':
                if limit is None and stop is None:
                    log = log_action('{} Limit and stop loss parameters are not set. This may be result of program restart.'
                                     .format(time_stamp()))
                    limits = joblib.load('limits.pkl')
                    limit, stop = limits[pair]['limit'], limits[pair]['stop']
                    if limit is None and stop is None:  # Case of manual buy
                        set_ptsl(pair, closing_price, 'M', new_timestamp, 1, 1)
                    limit, stop = limits[pair]['limit'], limits[pair]['stop']
                    log = log_action('{} Limit recovered {}. Stop loss recovered {}.'
                                     .format(time_stamp(), limit, stop))
                    trades.append(log)
                if closing_price < stop:
                    log = log_action('{} Closing price {} < stop {}.'.format(time_stamp(), closing_price, stop))
                    action(mode, crypto_currency, fiat_currency, closing_price)
                if closing_price > limit or new_timestamp >= limits[pair]['timestamp'] + 86400: # 86400 one day in seconds
                    log = log_action('{} Closing price {} > limit {} or prediction outdated.'
                                     .format(time_stamp(), closing_price, limit))
                    if event > minRet and bb_cross != 0 and roc10 > 0:
                        prime_predictionS, meta_predictionS = sell_evaluation(crypto_currency, high_frame_indicated,
                                                                                  mid_frame_indicated,
                                                                                  low_frame_indicated,
                                                                                  pms, mms)
                        log = log_action('Prime Prediction: {} Meta Prediction {}.'
                                             .format(prime_predictionS, meta_predictionS))
                        if prime_predictionS != meta_predictionS:
                            action(mode, crypto_currency, fiat_currency, closing_price)
                        else:
                            ret, pt, sl = ret_evaluation(crypto_currency, high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                            if ret > market_reset and roc10 > 0:
                                set_ptsl(pair, closing_price, 'R', new_timestamp, pt, sl)
                                log = log_action('{} Limit reset to {}. Stop reset to {}.'
                                                     .format(time_stamp(), limit, stop))
                                trades.append(log)
                    else:
                        reset_predictions()
            elif condition == 'buy':
                if event > minRet and bb_cross != 0 and roc10 > 0:
                    prime_predictionB, meta_predictionB = buy_evaluation(crypto_currency, high_frame_indicated,
                                                                         mid_frame_indicated,
                                                                         low_frame_indicated,
                                                                         pmb, mmb)
                    ret, pt, sl = ret_evaluation(crypto_currency, high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                    log = log_action('{} Prime prediction {}. Meta prediction {}. Ret {}. ROC10 {}.'
                                     .format(time_stamp(), prime_predictionB, meta_predictionB, ret, roc10))
                    if prime_predictionB == meta_predictionB and ret > market_return and roc10 > 0:
                        set_ptsl(pair, closing_price, 'S', new_timestamp, pt, sl)
                        log = log_action('{} Limit set {}. Stop loss set {}.'.format(time_stamp(), limit, stop))
                        trades.append(log)
                        action(mode, crypto_currency, fiat_currency, closing_price)
                else:
                    reset_predictions()
            time.sleep(10)
        while True:
            if break_event.is_set():  # thread "kill" by user
                cancel_order()
                log = log_action('{} Breaking operation.'.format(time_stamp()))
                break
            low_frame, mid_frame, high_frame = raw_data(assets[0][0], assets[0][1])
            new_candle_time = low_frame.iloc[-1]['time']
            if new_candle_time > first_candle_time:
                for crypto_currency, fiat_currency, pmb, mmb, pms, mms, mr in assets:
                    pair = crypto_currency+fiat_currency
                    log = log_action('Evaluating {}-{}.'.format(crypto_currency, fiat_currency))
                    low_frame, mid_frame, high_frame = raw_data(crypto_currency, fiat_currency)
                    low_frame_indicated, mid_frame_indicated, high_frame_indicated = indicators(low_frame, mid_frame, high_frame)
                    chart_data(pair, high_frame_indicated, mid_frame_indicated, low_frame_indicated)
                    closing_price = low_frame_indicated.iloc[-1]['close']
                    event = low_frame_indicated.iloc[-1]['event']
                    bb_cross = low_frame_indicated.iloc[-1]['bb_cross']
                    roc10 = low_frame_indicated.iloc[-1]['roc10']
                    new_timestamp = low_frame.iloc[-1]['time']
                    limit, stop = limits[pair]['limit'], limits[pair]['stop']
                    if mode != 'simulator':
                        condition, crypto_balance, fiat_balance = get_condition(crypto_currency, fiat_currency, closing_price)
                    log = log_action('Event is: {}. BB crossing is: {}. Condition is: {}'.format(event, bb_cross, condition))
                    if condition == 'sell':
                        if limit is None and stop is None:
                            log = log_action(
                                '{} Limit and stop loss parameters are not set. This may be result of program restart.'
                                .format(time_stamp()))
                            limits = joblib.load('limits.pkl')
                            limit, stop = limits[pair]['limit'], limits[pair]['stop']
                            if limit is None and stop is None:  # Case of manual buy
                                set_ptsl(pair, closing_price, 'M', new_timestamp, 1, 1)
                            limit, stop = limits[pair]['limit'], limits[pair]['stop']
                            log = log_action('{} Limit recovered {}. Stop loss recovered {}.'
                                             .format(time_stamp(), limit, stop))
                            trades.append(log)
                        if closing_price < stop:
                            log = log_action('{} Closing price {} < stop {}.'.format(time_stamp(), closing_price, stop))
                            action(mode, crypto_currency, fiat_currency, closing_price)
                        elif closing_price > limit or new_timestamp >= limits[pair]['timestamp'] + 86400: # 86400 one day in seconds
                            log = log_action('{} Closing price {} > limit {} or prediction outdated.'
                                             .format(time_stamp(), closing_price, limit))
                            if event > minRet and bb_cross != 0 and roc10 > 0:
                                prime_predictionS, meta_predictionS = sell_evaluation(crypto_currency, high_frame_indicated,
                                                                                          mid_frame_indicated,
                                                                                          low_frame_indicated,
                                                                                          pms, mms)
                                log = log_action('Prime Prediction: {} Meta Prediction {}.'
                                                     .format(prime_predictionS, meta_predictionS))
                                if prime_predictionS != meta_predictionS:
                                    action(mode, crypto_currency, fiat_currency, closing_price)
                                else:
                                    ret, pt, sl = ret_evaluation(crypto_currency, high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                                    if ret > market_reset and roc10 > 0:
                                        set_ptsl(pair, closing_price, 'R', new_timestamp, pt, sl)
                                        log = log_action('{} Limit reset to {}. Stop reset to {}.'
                                                             .format(time_stamp(), limit, stop))
                                        trades.append(log)
                            else:
                                reset_predictions()
                    elif condition == 'buy':
                        if event > minRet and bb_cross != 0 and roc10 > 0:
                            prime_predictionB, meta_predictionB = buy_evaluation(crypto_currency, high_frame_indicated,
                                                                                     mid_frame_indicated,
                                                                                     low_frame_indicated,
                                                                                     pmb, mmb)
                            ret, pt, sl = ret_evaluation(crypto_currency, high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                            log = log_action('{} Prime prediction {}. Meta prediction {}. Ret {}. ROC10 {}.'
                                                 .format(time_stamp(), prime_predictionB, meta_predictionB, ret, roc10))
                            if prime_predictionB == meta_predictionB and ret > market_return and roc10 > 0:
                                set_ptsl(pair, closing_price, 'S', new_timestamp, pt, sl)
                                log = log_action('{} Limit set {}. Stop loss set {}.'.format(time_stamp(), limit, stop))
                                trades.append(log)
                                action(mode, crypto_currency, fiat_currency, closing_price)
                        else:
                            reset_predictions()
                    time.sleep(10)
                log = log_action('{} Waiting next candle close.'.format(time_stamp()))
                time.sleep(1800 - (11*len(assets)))
            else:
                log = log_action('{} Waiting candle close.'.format(time_stamp()))
                time.sleep(59)  # wait one 1min - 1 second for first candle to close
    else:
        activation()
        log = log_action('Your product licence is not active. Please restart, activate or contact technical support. '
                         'Hermes_algotrading@proton.me')
        exit()


def data_feed():
    if len(logs) > 300:
        del logs[:len(logs) - 299]
    return {
        'log': log,
        'logs': logs,
        'trades': trades,
        'condition': condition,
        'crypto_balance': float(crypto_balance),
        'fiat_balance': float(fiat_balance),
        'high_chart_data': high_chart_data,
        'mid_chart_data': mid_chart_data,
        'high_ema20': high_ema20,
        'high_ema3': high_ema3,
        'low_chart_data': low_chart_data,
        'price': low_closing_price,
        'low_limit': low_limit,
        'low_stop': low_stop,
        'low_ave': low_ave,
        'low_upper': low_upper,
        'low_lower': low_lower,
        'event': 'True' if event > minRet else 'False',
        'bb_cross': bb_cross,
        'prime_prediction': 'True' if prime_prediction == 1 else ('False' if prime_prediction == 0 else None),
        'meta_prediction': 'True' if meta_prediction == 1 else ('False' if meta_prediction == 0 else None),
        'ret': round(ret, 4) if ret != 0 else None,
        'roc10': round(roc10, 4),
        'crypto_currency': crypto_currency,
        'fiat_currency': fiat_currency
    }

# def Prelderbot(mode, crypto_currency, fiat_currency, pmb, mmb, pms, mms, mr):
#     global condition, limits, ret, log, crypto_balance, fiat_balance, closing_price, event, bb_cross, \
#         prime_prediction, meta_prediction, roc10, timestamp, first_candle_time
#     licence = True  # check()['license_active']
#     pair = crypto_currency + fiat_currency
#     if licence:
#         log = log_action('Your product licence is active. Thank you for using Hermes.')
#         log = log_action('{} Operation start. Mode is {}.'.format(time_stamp(), mode))
#         low_frame, mid_frame, high_frame = raw_data(crypto_currency, fiat_currency)
#         low_frame_indicated, mid_frame_indicated, high_frame_indicated = indicators(low_frame, mid_frame, high_frame)
#         chart_data(pair, high_frame_indicated, mid_frame_indicated, low_frame_indicated)
#         closing_price = low_frame_indicated.iloc[-1]['close']
#         event = low_frame_indicated.iloc[-1]['event']
#         bb_cross = low_frame_indicated.iloc[-1]['bb_cross']
#         roc10 = low_frame_indicated.iloc[-1]['roc10']
#         new_timestamp = first_candle_time = low_frame_indicated.iloc[-1]['time']
#         condition, crypto_balance, fiat_balance = get_condition(crypto_currency, fiat_currency, closing_price)
#         limit, stop = limits[pair]['limit'], limits[pair]['stop']
#         if mode == 'simulator':
#             condition = 'buy'
#         log = log_action('Event is: {}. BB crossing is: {}. Condition is: {}'.format(event, bb_cross, condition))
#         if condition == 'sell':
#             if limit is None and stop is None:
#                 log = log_action('{} Limit and stop loss parameters are not set. This may be result of program restart.'
#                                  .format(time_stamp()))
#                 limits = joblib.load('limits.pkl')
#                 limit, stop, timestamp = limits[pair]['limit'], limits[pair]['stop'], limits[pair]['timestamp']
#                 if limit is None and stop is None:  # Case of manual buy
#                     set_ptsl(pair, 'M', new_timestamp, 1, 1)
#                 limit, stop, timestamp = limits[pair]['limit'], limits[pair]['stop'], limits[pair]['timestamp']
#                 log = log_action('{} Limit recovered {}. Stop loss recovered {}.'
#                                  .format(time_stamp(), limit, stop))
#                 trades.append(log)
#             if closing_price < stop:
#                 log = log_action('{} Closing price < stop.'.format(time_stamp()))
#                 action(mode, crypto_currency, fiat_currency, closing_price)
#             if closing_price > limit or new_timestamp >= timestamp + 86400: # 86400 one day in seconds
#                 log = log_action('{} Closing price > limit or prediction outdated.'.format(time_stamp()))
#                 if event > minRet and bb_cross != 0 and roc10 > 0:
#                     prime_predictionS, meta_predictionS = sell_evaluation(crypto_currency, high_frame_indicated,
#                                                                               mid_frame_indicated,
#                                                                               low_frame_indicated,
#                                                                               pms, mms)
#                     log = log_action('Prime Prediction: {} Meta Prediction {}.'
#                                          .format(prime_predictionS, meta_predictionS))
#                     if prime_predictionS != meta_predictionS:
#                         action(mode, crypto_currency, fiat_currency, closing_price)
#                     else:
#                         ret, pt, sl = ret_evaluation(crypto_currency, high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
#                         if ret > market_reset and roc10 > 0:
#                             set_ptsl(pair, 'R', new_timestamp, pt, sl)
#                             log = log_action('{} Limit reset to {}. Stop reset to {}.'
#                                                  .format(time_stamp(), limit, stop))
#                             trades.append(log)
#                 else:
#                         reset_predictions()
#         elif condition == 'buy':
#             if event > minRet and bb_cross != 0 and roc10 > 0:
#                 prime_predictionB, meta_predictionB = buy_evaluation(crypto_currency, high_frame_indicated,
#                                                                      mid_frame_indicated,
#                                                                      low_frame_indicated,
#                                                                      pmb, mmb)
#                 ret, pt, sl = ret_evaluation(crypto_currency, high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
#                 log = log_action('{} Prime prediction {}. Meta prediction {}. Ret {}. ROC10 {}.'
#                                  .format(time_stamp(), prime_predictionB, meta_predictionB, ret, roc10))
#                 if prime_predictionB == meta_predictionB and ret > market_return and roc10 > 0:
#                     set_ptsl(pair, 'S', new_timestamp, pt, sl)
#                     log = log_action('{} Limit set {}. Stop loss set {}.'.format(time_stamp(), limit, stop))
#                     trades.append(log)
#                     action(mode, crypto_currency, fiat_currency, closing_price)
#             else:
#                 reset_predictions()
#         while True:
#             if break_event.is_set():  # thread "kill" by user
#                 cancel_order()
#                 log = log_action('{} Breaking operation.'.format(time_stamp()))
#                 break
#             low_frame, mid_frame, high_frame = raw_data(crypto_currency, fiat_currency)
#             low_frame_indicated, mid_frame_indicated, high_frame_indicated = indicators(low_frame, mid_frame, high_frame)
#             chart_data(pair, high_frame_indicated, mid_frame_indicated, low_frame_indicated)
#             new_candle_time = low_frame_indicated.iloc[-1]['time']
#             closing_price = low_frame_indicated.iloc[-1]['close']
#             event = low_frame_indicated.iloc[-1]['event']
#             bb_cross = low_frame_indicated.iloc[-1]['bb_cross']
#             roc10 = low_frame_indicated.iloc[-1]['roc10']
#             new_timestamp = low_frame.iloc[-1]['time']
#             if mode != 'simulator':
#                 condition, crypto_balance, fiat_balance = get_condition(crypto_currency, fiat_currency, closing_price)
#             log = log_action('Event is: {}. BB crossing is: {}. Condition is: {}'.format(event, bb_cross, condition))
#             if new_candle_time > first_candle_time:  # wait first 30min candle to close
#                 if condition == 'sell':
#                     if closing_price < stop:
#                         log = log_action('{} Closing price < stop.'.format(time_stamp()))
#                         action(mode, crypto_currency, fiat_currency, closing_price)
#                     elif closing_price > limit or new_timestamp >= timestamp + 86400: # one day in seconds
#                         log = log_action('{} Closing price > limit or prediction outdated.'.format(time_stamp()))
#                         if event > minRet and bb_cross != 0 and roc10 > 0:
#                             prime_predictionS, meta_predictionS = sell_evaluation(crypto_currency, high_frame_indicated,
#                                                                                   mid_frame_indicated,
#                                                                                   low_frame_indicated,
#                                                                                   pms, mms)
#                             log = log_action('Prime Prediction: {} Meta Prediction {}.'
#                                              .format(prime_predictionS, meta_predictionS))
#                             if prime_predictionS != meta_predictionS:
#                                 action(mode, crypto_currency, fiat_currency, closing_price)
#                             else:
#                                 ret, pt, sl = ret_evaluation(crypto_currency, high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
#                                 if ret > market_reset and roc10 > 0:
#                                     set_ptsl(pair, 'R', new_timestamp, pt, sl)
#                                     log = log_action('{} Limit reset to {}. Stop reset to {}.'
#                                                      .format(time_stamp(), limit, stop))
#                                     trades.append(log)
#                         else:
#                             reset_predictions()
#                 elif condition == 'buy':
#                     if event > minRet and bb_cross != 0 and roc10 > 0:
#                         prime_predictionB, meta_predictionB = buy_evaluation(crypto_currency, high_frame_indicated,
#                                                                              mid_frame_indicated,
#                                                                              low_frame_indicated,
#                                                                              pmb, mmb)
#                         ret, pt, sl = ret_evaluation(crypto_currency, high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
#                         log = log_action('{} Prime prediction {}. Meta prediction {}. Ret {}. ROC10 {}.'
#                                          .format(time_stamp(), prime_predictionB, meta_predictionB, ret, roc10))
#                         if prime_predictionB == meta_predictionB and ret > market_return and roc10 > 0:
#                             set_ptsl(pair, 'S', new_timestamp, pt, sl)
#                             log = log_action('{} Limit set {}. Stop loss set {}.'.format(time_stamp(), limit, stop))
#                             trades.append(log)
#                             action(mode, crypto_currency, fiat_currency, closing_price)
#                     else:
#                         reset_predictions()
#                 log = log_action('{} Waiting next candle close.'.format(time_stamp()))
#                 time.sleep(1799)  # wait one 30min candle - 1 second
#             else:
#                 log = log_action('{} Waiting candle close.'.format(time_stamp()))
#                 time.sleep(59)  # wait one 1min - 1 second for first candle to close
#     else:
#         activation()
#         log = log_action('Your product licence is not active. Please activate or contact technical support. '
#                          'Hermes_algotrading@proton.me')
#         exit()