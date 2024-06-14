import logging
import time
import threading
import winsound
from app.back.kraken import Api, time_stamp, \
    add_order, cancel_order, get_condition, high_data, mid_data, low_data, indicators, minRet
from app.back.spring import activation  # , check TODO activate licence check
from sklearn.preprocessing import Normalizer, normalize
import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)
logging.basicConfig(level=logging.INFO)

condition = None
crypto_balance = 0
fiat_balance = 0
limit = None
stop = None
ret = 0
roc30 = 0
order_type = 'market'
closing_price = 0
pt = 1
sl = 0.5

high_chart_data = None
mid_chart_data = None
low_chart_data = None
high_ema20 = None
high_ema3 = None
high_TrD20 = None
high_TrD3 = None
mid_k = None
mid_d = None
mid_macd = None
low_ave = None
low_upper = None
low_lower = None
event = 0
bb_cross = None
prime_prediction = None
meta_prediction = None
low_Tr6 = None
low_limit = None
low_stop = None
volatility = None
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


def chart_data(high_frame, mid_frame, low_frame):
    global high_chart_data, high_ema20, high_ema3, high_TrD20, high_TrD3, mid_chart_data, mid_k, mid_d, mid_macd, \
        low_chart_data, low_ave, low_upper, low_lower, low_Tr6, low_limit, low_stop, volatility
    limit_data = []
    stop_data = []
    high_candles, ema20, ema3, TrD20, TrD3 = high_data(high_frame)
    mid_candles, k, d, mac4 = mid_data(mid_frame)
    low_candles, ave, upper, lower, Tr6, volatility = low_data(low_frame)
    for i in low_candles:
        limit_data.append({
            'x': i['x'],
            'y': limit
        })
        stop_data.append({
            'x': i['x'],
            'y': stop
        })
    high_chart_data = high_candles[-20:]
    high_ema20 = ema20[-20:]
    high_ema3 = ema3[-20:]
    high_TrD20 = TrD20
    high_TrD3 = TrD3
    mid_chart_data = mid_candles[-20:]
    mid_k = k
    mid_d = d
    mid_macd = mac4
    low_chart_data = low_candles[-20:]
    low_ave = ave[-20:]
    low_lower = lower[-20:]
    low_upper = upper[-20:]
    low_Tr6 = Tr6
    low_limit = limit_data[-20:]
    low_stop = stop_data[-20:]


def log_action(message):
    logging.info(message)
    logs.append(message + '<br>')
    return message


def sell_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, pms, mms):
    global prime_prediction, meta_prediction
    TrD20 = high_frame_indicated.iloc[-1]['TrD20']
    TrD3 = high_frame_indicated.iloc[-1]['TrD3']
    D4 = mid_frame_indicated.iloc[-1]['%D']
    mac4 = mid_frame_indicated.iloc[-1]['mac4']
    Tr6 = low_frame_indicated.iloc[-1]['Tr6']
    roc = low_frame_indicated.iloc[-1]['roc30']
    rsi = low_frame_indicated.iloc[-1]['rsi']
    bb_l = low_frame_indicated.iloc[-1]['bb_l']
    bbc = low_frame_indicated.iloc[-1]['bb_cross']
    featuresS = [[TrD20, TrD3, D4, mac4, Tr6, roc, bb_l, rsi]]
    featuresS = normalize(featuresS)
    featuresS = np.insert(featuresS, len(featuresS[0]), bbc)
    prime_predictionS = pms.predict([featuresS])
    featuresMS = featuresS
    featuresMS = np.insert(featuresMS, len(featuresS), prime_predictionS)
    meta_predictionS = mms.predict([featuresMS])
    prime_prediction, meta_prediction = prime_predictionS[0], meta_predictionS[0]
    return prime_predictionS[0], meta_predictionS[0]


def buy_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, pmb, mmb):
    global prime_prediction, meta_prediction
    TrD20 = high_frame_indicated.iloc[-1]['TrD20']
    TrD3 = high_frame_indicated.iloc[-1]['TrD3']
    mac4 = mid_frame_indicated.iloc[-1]['mac4']
    vol = low_frame_indicated.iloc[-1]['Volatility']
    vv = low_frame_indicated.iloc[-1]['Vol_Volatility']
    roc = low_frame_indicated.iloc[-1]['roc30']
    rsi = low_frame_indicated.iloc[-1]['rsi']
    bbc = low_frame_indicated.iloc[-1]['bb_cross']
    featuresB = [[TrD20, TrD3, mac4, vol, vv, roc, rsi]]
    featuresB = normalize(featuresB)
    featuresB = np.insert(featuresB, len(featuresB[0]), bbc)
    prime_predictionB = pmb.predict([featuresB])
    featuresMB = featuresB
    featuresMB = np.insert(featuresMB, len(featuresMB), prime_predictionB)
    meta_predictionB = mmb.predict([featuresMB])
    prime_prediction, meta_prediction = prime_predictionB[0], meta_predictionB[0]
    return prime_predictionB[0], meta_predictionB[0]


def ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr):
    TrD20 = high_frame_indicated.iloc[-1]['TrD20']
    TrD3 = high_frame_indicated.iloc[-1]['TrD3']
    mac4 = mid_frame_indicated.iloc[-1]['mac4']
    vol = low_frame_indicated.iloc[-1]['Volatility']
    vv = low_frame_indicated.iloc[-1]['Vol_Volatility']
    roc = low_frame_indicated.iloc[-1]['roc30']
    rsi = low_frame_indicated.iloc[-1]['rsi']
    bbc = low_frame_indicated.iloc[-1]['bb_cross']
    featuresB = [[TrD20, TrD3, mac4, vol, vv, roc, rsi]]
    featuresB = normalize(featuresB)
    featuresB = np.insert(featuresB, len(featuresB[0]), bbc)
    ret_prediction = mr.predict([featuresB])
    return ret_prediction[0]


def action(mode, crypto_currency, fiat_currency):
    global log, condition, limit, stop
    if mode == 'simulator':
        log = log_action('{} Simulating {} at {}'.format(time_stamp(), condition, closing_price))
        trades.append(log)
        if condition == 'buy':
            condition = 'sell'
        elif condition == 'sell':
            condition = 'buy'
    elif mode == 'consulting':
        log = log_action('{} Consulting {} at {}.'.format(time_stamp(), condition, closing_price))
        trades.append(log)
    elif mode == 'trading':
        log = log_action('{} {} at {}.'.format(time_stamp(),condition, closing_price))
        trades.append(log)
        asset_vol = (fiat_balance - fiat_balance * minRet) / closing_price
        tx = add_order(order_type, condition, asset_vol, closing_price, crypto_currency,
                       fiat_currency)
        log = log_action(tx)
        trades.append(time_stamp())
        trades.append(log + '<br>')


def reset_predictions():
    global prime_prediction, meta_prediction, ret
    prime_prediction = None
    meta_prediction = None
    ret = 0


def reset_ptsl():
    global limit, stop
    limit = None
    stop = None


def Prelderbot(mode, crypto_currency, fiat_currency, pmb, mmb, pms, mms, mr):
    global condition, limit, stop, ret, log, crypto_balance, fiat_balance, closing_price, event, bb_cross, \
        prime_prediction, meta_prediction, roc30
    licence = True  # check()['license_active']  # TODO activate licence check
    if licence:
        log = log_action('Your product licence is active. Thank you for using Hermes.')
        log = log_action('{} Operation start. Mode is {}.'.format(time_stamp(), mode))
        i30m = Api(crypto_currency, fiat_currency, 30, 700)
        i4H = Api(crypto_currency, fiat_currency, 240, 100)
        i24H = Api(crypto_currency, fiat_currency, 1440, 100)
        low_frame = i30m.get_frame()
        mid_frame = i4H.get_frame()
        high_frame = i24H.get_frame()
        low_frame_indicated, mid_frame_indicated, high_frame_indicated = indicators(low_frame, mid_frame, high_frame)
        chart_data(high_frame_indicated, mid_frame_indicated, low_frame_indicated)
        closing_price = low_frame_indicated.iloc[-1]['close']
        candle_time = low_frame_indicated.iloc[-1]['time']
        event = low_frame_indicated.iloc[-1]['event']
        bb_cross = low_frame_indicated.iloc[-1]['bb_cross']
        roc30 = low_frame_indicated.iloc[-1]['roc30']
        condition, crypto_balance, fiat_balance = get_condition(crypto_currency, fiat_currency, closing_price)
        if mode == 'simulator':
            condition = 'buy'
        log = log_action('Event is: {}. BB crossing is: {}. Condition is: {}'.format(event, bb_cross, condition))
        if condition == 'sell':
            if limit is None and stop is None:
                log = log_action('{} Limit and stop loss parameters are not set. This may be result of program restart.'
                                 .format(time_stamp()))
                if roc30 > 0:
                    limit = closing_price * (1 + (roc30 / 100) * pt)
                    stop = closing_price * (1 - (roc30 / 100) * sl)
                else:
                    limit = closing_price * 1.01
                    stop = closing_price * .99
                log = log_action('{} Limit set {}. Stop loss set {}.'.format(time_stamp(), limit, stop))
                trades.append(log)
            else:
                if closing_price < stop:
                    log = log_action('{} Closing price < stop.'.format(time_stamp()))
                    action(mode, crypto_currency, fiat_currency)
                    reset_ptsl()
                elif closing_price > limit:
                    log = log_action('{} Closing price > limit.'.format(time_stamp()))
                    if event != 0 and bb_cross != 0:
                        prime_predictionS, meta_predictionS = sell_evaluation(high_frame_indicated,
                                                                              mid_frame_indicated,
                                                                              low_frame_indicated,
                                                                              pms, mms)
                        log = log_action('Prime Prediction: {} Meta Prediction {}.'
                                         .format(prime_predictionS, meta_predictionS))
                        if prime_predictionS != meta_predictionS:
                            action(mode, crypto_currency, fiat_currency)
                            reset_ptsl()
                        else:
                            ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                            if ret > minRet and roc30 > 0:
                                limit = closing_price * (1 + (ret + (roc30 / 100) * pt))
                                stop = closing_price * (1 - (ret + (roc30 / 100) * sl))
                                log = log_action('{} Limit reset to {}. Stop reset to {}.'
                                                 .format(time_stamp(), limit, stop))
                                trades.append(log)
                    else:
                        reset_predictions()
        elif condition == 'buy':
            if event > minRet and bb_cross != 0:
                prime_predictionB, meta_predictionB = buy_evaluation(high_frame_indicated,
                                                                     mid_frame_indicated,
                                                                     low_frame_indicated,
                                                                     pmb, mmb)
                ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                log = log_action('{} Prime prediction {}. Meta prediction {}. Ret {}. ROC30 {}.'
                                 .format(time_stamp(), prime_predictionB, meta_predictionB, ret, roc30))
                if prime_predictionB == meta_predictionB and ret > minRet and roc30 > 0:
                    limit = closing_price * (1 + (ret + (roc30 / 100) * pt))
                    stop = closing_price * (1 - (ret + (roc30 / 100) * sl))
                    log = log_action('{} Limit set {}. Stop loss set {}.'.format(time_stamp(), limit, stop))
                    trades.append(log)
                    action(mode, crypto_currency, fiat_currency)
            else:
                reset_predictions()
        while True:
            if break_event.is_set():  # thread "kill" by user
                cancel_order()
                log = log_action('{} Breaking operation.'.format(time_stamp()))
                break
            low_frame = i30m.get_frame()
            mid_frame = i4H.get_frame()
            high_frame = i24H.get_frame()
            low_frame_indicated, mid_frame_indicated, high_frame_indicated = \
                indicators(low_frame, mid_frame, high_frame)
            chart_data(high_frame_indicated, mid_frame_indicated, low_frame_indicated)
            new_candle_time = low_frame_indicated.iloc[-1]['time']
            closing_price = low_frame_indicated.iloc[-1]['close']
            event = low_frame_indicated.iloc[-1]['event']
            bb_cross = low_frame_indicated.iloc[-1]['bb_cross']
            roc30 = low_frame_indicated.iloc[-1]['roc30']
            if mode != 'simulator':
                condition, crypto_balance, fiat_balance = get_condition(crypto_currency, fiat_currency, closing_price)
            log = log_action('Event is: {}. BB crossing is: {}. Condition is: {}'.format(event, bb_cross, condition))
            if new_candle_time > candle_time:  # wait first 30min candle to close
                if condition == 'sell':
                    if closing_price < stop:
                        log = log_action('{} Closing price < stop.'.format(time_stamp()))
                        action(mode, crypto_currency, fiat_currency)
                        reset_ptsl()
                    elif closing_price > limit:
                        log = log_action('{} Closing price > limit.'.format(time_stamp()))
                        if event != 0 and bb_cross != 0:
                            prime_predictionS, meta_predictionS = sell_evaluation(high_frame_indicated,
                                                                                  mid_frame_indicated,
                                                                                  low_frame_indicated,
                                                                                  pms, mms)
                            log = log_action('Prime Prediction: {} Meta Prediction {}.'
                                             .format(prime_predictionS, meta_predictionS))
                            if prime_predictionS != meta_predictionS:
                                action(mode, crypto_currency, fiat_currency)
                                reset_ptsl()
                            else:
                                ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                                if ret > 0 and roc30 > 0:
                                    limit = closing_price * (1 + (ret + (roc30 / 100) * pt))
                                    stop = closing_price * (1 - (ret + (roc30 / 100) * sl))
                                    log = log_action('{} Limit reset to {}. Stop reset to {}.'
                                                     .format(time_stamp(), limit, stop))
                                    trades.append(log)
                        else:
                            reset_predictions()
                elif condition == 'buy':
                    if event > minRet and bb_cross != 0:
                        prime_predictionB, meta_predictionB = buy_evaluation(high_frame_indicated,
                                                                             mid_frame_indicated,
                                                                             low_frame_indicated,
                                                                             pmb, mmb)
                        ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                        log = log_action('{} Prime prediction {}. Meta prediction {}. Ret {}. ROC30 {}.'
                                         .format(time_stamp(), prime_predictionB, meta_predictionB, ret, roc30))
                        if prime_predictionB == meta_predictionB and ret > minRet and roc30 > 0:
                            limit = closing_price * (1 + (ret + (roc30 / 100) * pt))
                            stop = closing_price * (1 - (ret + (roc30 / 100) * sl))
                            log = log_action('{} Limit set {}. Stop loss set {}.'.format(time_stamp(), limit, stop))
                            trades.append(log)
                            action(mode, crypto_currency, fiat_currency)
                    else:
                        reset_predictions()
                log = log_action('{} Waiting 30 min candle close.'.format(time_stamp()))
                time.sleep(1799)  # wait one 30min candle - 1 second
            else:
                log = log_action('{} Waiting 30 min candle close.'.format(time_stamp()))
                time.sleep(59)  # wait one 1min - 1 second for first candle to close
    else:
        activation()
        log = log_action('Your product licence is not active. Please activate or contact technical support. '
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
        'high_TrD20': high_TrD20,
        'high_TrD3': high_TrD3,
        'mid_k': mid_k,
        'mid_d': mid_d,
        'mid_macd': mid_macd,
        'low_chart_data': low_chart_data,
        'price': closing_price,
        'low_limit': low_limit,
        'low_stop': low_stop,
        'low_ave': low_ave,
        'low_upper': low_upper,
        'low_lower': low_lower,
        'event': 'True' if event > minRet else 'False',
        'bb_cross': bb_cross,
        'volatility': volatility,
        'prime_prediction': 'True' if prime_prediction == 1 else ('False' if prime_prediction == 0 else None),
        'meta_prediction': 'True' if meta_prediction == 1 else ('False' if meta_prediction == 0 else None),
        'ret': round(ret, 4) if ret != 0 else None,
        'roc30': round(roc30, 4)
    }