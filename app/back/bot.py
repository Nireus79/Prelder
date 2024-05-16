import logging
import time
import threading
import winsound
from app.back.kraken import Api, time_stamp, \
    add_order, cancel_order, get_condition, high_data, mid_data, low_data, indicators
from app.back.spring import check, activation
from sklearn.preprocessing import Normalizer, normalize
import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)
logging.basicConfig(level=logging.INFO)

condition = None
trend_24h = None
trend_4h = None
buy_flag_4h = False
buy_flag_1h = False
sell_flag_4h = None
sell_flag_1h = None
crypto_balance = 0
fiat_balance = 0
limit = None
stop = None
ret = 0
fee = 0.026
order_type = 'market'
closing_price = None

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
event = None
bb_cross = None
low_Tr6 = None
low_limit = None
low_stop = None
log = 'Please set control parameters to start.'
logs = []
trades = []

break_event = threading.Event()


def normalizer(data):
    """
    Normalization is a good technique to use when you do not know the distribution of your data or when you know the
    distribution is not Gaussian (a bell curve). Normalization is useful when your data has varying scales and the
    algorithm you are using does not make assumptions about the distribution of your data, such as k-nearest
    neighbors and artificial neural networks. :param data: :return:
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
        low_chart_data, low_ave, low_upper, low_lower, low_Tr6, low_limit, low_stop
    limit_data = []
    stop_data = []
    high_candles, ema20, ema3, TrD20, TrD3 = high_data(high_frame)
    mid_candles, k, d, mac4 = mid_data(mid_frame)
    low_candles, ave, upper, lower, Tr6 = low_data(low_frame)
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


def sell_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, pms, mms):
    TrD20 = high_frame_indicated.iloc[-1]['TrD20']
    TrD3 = high_frame_indicated.iloc[-1]['TrD3']
    D4 = mid_frame_indicated.iloc[-1]['%D']
    mac4 = mid_frame_indicated.iloc[-1]['mac4']
    Tr6 = low_frame_indicated.iloc[-1]['Tr6']
    roc30 = low_frame_indicated.iloc[-1]['roc30']
    rsi = low_frame_indicated.iloc[-1]['rsi']
    bb_l = low_frame_indicated.iloc[-1]['bb_l']
    bbc = low_frame_indicated.iloc[-1]['bb_cross']
    featuresS = [[TrD20, TrD3, D4, mac4, Tr6, roc30, bb_l, rsi]]
    featuresS = normalize(featuresS)
    featuresS = np.insert(featuresS, len(featuresS[0]), bbc)
    prime_predictionS = pms.predict([featuresS])
    featuresMS = featuresS
    featuresMS = np.insert(featuresMS, len(featuresS), prime_predictionS)
    meta_predictionS = mms.predict([featuresMS])
    return prime_predictionS, meta_predictionS


def buy_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, pmb, mmb):
    TrD20 = high_frame_indicated.iloc[-1]['TrD20']
    TrD3 = high_frame_indicated.iloc[-1]['TrD3']
    mac4 = mid_frame_indicated.iloc[-1]['mac4']
    vol = low_frame_indicated.iloc[-1]['Volatility']
    vv = low_frame_indicated.iloc[-1]['Vol_Volatility']
    roc30 = low_frame_indicated.iloc[-1]['roc30']
    mom10 = low_frame_indicated.iloc[-1]['mom10']
    rsi = low_frame_indicated.iloc[-1]['rsi']
    bbc = low_frame_indicated.iloc[-1]['bb_cross']
    featuresB = [[TrD20, TrD3, mac4, vol, vv, roc30, mom10, rsi]]
    featuresB = normalize(featuresB)
    featuresB = np.insert(featuresB, len(featuresB[0]), bbc)
    prime_predictionB = pmb.predict([featuresB])
    featuresMB = featuresB
    featuresMB = np.insert(featuresMB, len(featuresMB), prime_predictionB)
    meta_predictionB = mmb.predict([featuresMB])
    return prime_predictionB, meta_predictionB


def ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr):
    TrD20 = high_frame_indicated.iloc[-1]['TrD20']
    TrD3 = high_frame_indicated.iloc[-1]['TrD3']
    mac4 = mid_frame_indicated.iloc[-1]['mac4']
    vol = low_frame_indicated.iloc[-1]['Volatility']
    vv = low_frame_indicated.iloc[-1]['Vol_Volatility']
    roc30 = low_frame_indicated.iloc[-1]['roc30']
    mom10 = low_frame_indicated.iloc[-1]['mom10']
    rsi = low_frame_indicated.iloc[-1]['rsi']
    bbc = low_frame_indicated.iloc[-1]['bb_cross']
    featuresB = [[TrD20, TrD3, mac4, vol, vv, roc30, mom10, rsi]]
    featuresB = normalize(featuresB)
    featuresB = np.insert(featuresB, len(featuresB[0]), bbc)
    ret_prediction = mr.predict([featuresB])
    return ret_prediction


def Prelderbot(mode, crypto_currency, fiat_currency, pmb, mmb, pms, mms, mr):
    global condition, limit, stop, ret, log, crypto_balance, fiat_balance, closing_price, event, bb_cross
    licence = True  # check()['license_active'] # TODO activate licence check
    if licence:
        log = 'Your product licence is active. Thank you for using Hermes.'
        logging.info(log)
        logs.append(log + '<br>')
        log = '{} Operation start. Mode is {}.'.format(time_stamp(), mode)
        logging.info(log)
        logs.append(log + '<br>')
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
        mav = low_frame_indicated.iloc[-1]['MAV']
        roc30 = low_frame_indicated.iloc[-1]['roc30']
        condition, crypto_balance, fiat_balance = get_condition(crypto_currency, fiat_currency, closing_price)
        if mode == 'simulator':
            condition = 'Buy'
        log = 'Event is: {}. BB crossing is: {}'.format(event, bb_cross)
        logging.info(log)
        logs.append(log + '<br>')
        log = 'Condition is: {}'.format(condition)
        logging.info(log)
        logs.append(log + '<br>')
        if condition == 'Sell':
            if limit is None and stop is None:
                log = '{} Limit and stop loss parameters are not set.' \
                      ' This may be result of program restart.' \
                      ' Parameters will be set at first co event.' \
                    .format(time_stamp())
                logging.info(log)
                logs.append(log + '<br>')
                if event != 0 and bb_cross != 0:
                    prime_predictionS, meta_predictionS = sell_evaluation(high_frame_indicated,
                                                                          mid_frame_indicated,
                                                                          low_frame_indicated,
                                                                          pms, mms)
                    if prime_predictionS != meta_predictionS:
                        if mode == 'simulator':
                            log = '{} Prime Prediction: {} Meta Prediction {}. Simulating Sale at {}.' \
                                .format(time_stamp(), prime_predictionS, meta_predictionS, closing_price)
                            logging.info(log)
                            logs.append(log + '<br>')
                            trades.append(log)
                            condition = 'Buy'
                        elif mode == 'consulting':
                            log = '{} Prime Prediction: {} Meta Prediction {}. Consulting Sale at {}.' \
                                .format(time_stamp(), prime_predictionS, meta_predictionS, closing_price)
                            logging.info(log)
                            logs.append(log + '<br>')
                        elif mode == 'trading':
                            log = '{} Prime Prediction {} Meta Prediction {}.' \
                                .format(time_stamp(), prime_predictionS, meta_predictionS)
                            logging.info(log)
                            logs.append(log + '<br>')
                            asset_vol = (fiat_balance - fiat_balance * fee) / closing_price
                            tx = add_order(order_type, condition, asset_vol, closing_price, crypto_currency, fiat_currency)
                            log = tx
                            logging.info(log)
                    else:
                        ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                        limit = closing_price * (1 + (ret + (roc30 / 100)))
                        stop = closing_price * (1 - (ret + (roc30 / 100)))
                        log = '{} Prime Prediction: {} Meta Prediction {}.Limit set {}. Stop loss set {}.' \
                            .format(time_stamp(), prime_predictionS, meta_predictionS, limit, stop)
                        logging.info(log)
                        logs.append(log + '<br>')
            else:
                if closing_price < stop:
                    if mode == 'simulator':
                        log = '{} Closing price < stop.  Simulating Sale at {}.'.format(time_stamp(), closing_price)
                        logging.info(log)
                        logs.append(log + '<br>')
                        trades.append(log)
                        condition = 'Buy'
                    elif mode == 'consulting':
                        log = '{} Closing price < stop. Consulting Sale at {}.'.format(time_stamp(), closing_price)
                        logs.append(log + '<br>')
                        trades.append(log)
                    elif mode == 'trading':
                        log = '{} Closing price < stop. Sale at {}.'.format(time_stamp(), closing_price)
                        logging.info(log)
                        logs.append(log + '<br>')
                        asset_vol = (fiat_balance - fiat_balance * fee) / closing_price
                        tx = add_order(order_type, condition, asset_vol, closing_price, crypto_currency, fiat_currency)
                        log = tx
                        logging.info(log)
                        trades.append(log)
                elif closing_price > limit:
                    if event != 0 and bb_cross != 0:
                        prime_predictionS, meta_predictionS = sell_evaluation(high_frame_indicated,
                                                                              mid_frame_indicated,
                                                                              low_frame_indicated,
                                                                              pms, mms)
                        if prime_predictionS != meta_predictionS:
                            if mode == 'simulator':
                                log = '{} Closing price > limit. Prime Prediction: {} Meta Prediction {}. Simulating ' \
                                      'Sale at {}' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS, closing_price)
                                logging.info(log)
                                logs.append(log + '<br>')
                                trades.append(log)
                            elif mode == 'consulting':
                                log = '{} Prime Prediction {} Meta Prediction {}. Consulting Sale at {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS, closing_price)
                                logging.info(log)
                                logs.append(log + '<br>')
                                trades.append(log)
                            elif mode == 'trading':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS)
                                logging.info(log)
                                logs.append(log + '<br>')
                                asset_vol = (fiat_balance - fiat_balance * fee) / closing_price
                                tx = add_order(order_type, condition, asset_vol, closing_price, crypto_currency,
                                               fiat_currency)
                                log = tx
                                logging.info(log)
                                trades.append(log)
                        else:
                            ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                            if ret > 0 and roc30 > 0:
                                limit = closing_price * (1 + (ret + (roc30 / 100)))
                                stop = closing_price * (1 - (ret + (roc30 / 100)))
                                log = '{} Limit reset to {}. Stop reset to {}.' \
                                    .format(time_stamp(), limit, stop)
                                logging.info(log)
                                logs.append(log + '<br>')
                                trades.append(log)
        elif condition == 'Buy':
            if event != 0 and bb_cross != 0 and mav > fee:
                prime_predictionB, meta_predictionB = buy_evaluation(high_frame_indicated,
                                                                     mid_frame_indicated,
                                                                     low_frame_indicated,
                                                                     pmb, mmb)
                ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                if prime_predictionB == meta_predictionB and ret > fee and roc30 > 0:
                    limit = closing_price * (1 + (ret + (roc30 / 100)))
                    stop = closing_price * (1 - (ret + (roc30 / 100)))
                    if mode == 'simulator':
                        log = '{} Simulating Buy at {}. Limit set to {}. Stop set to {}.' \
                            .format(time_stamp(), closing_price, limit, stop)
                        logging.info(log)
                        logs.append(log + '<br>')
                        trades.append(log)
                        condition = 'Sell'
                    elif mode == 'consulting':
                        log = '{} Consulting Buy at {}. Limit set to {}. Stop set to {}.' \
                            .format(time_stamp(), closing_price, limit, stop)
                        logging.info(log)
                        logs.append(log + '<br>')
                        trades.append(log)
                    elif mode == 'trading':
                        log = '{} Buy at {}. Limit set to {}. Stop set to {}' \
                            .format(time_stamp(), closing_price, limit, stop)
                        logging.info(log)
                        logs.append(log + '<br>')
                        asset_vol = (fiat_balance - fiat_balance * fee) / closing_price
                        tx = add_order(order_type, condition, asset_vol, closing_price, crypto_currency, fiat_currency)
                        log = tx
                        logging.info(log)
                        trades.append(log)
        while True:
            if break_event.is_set():  # thread "kill" by user
                cancel_order()
                log = '{} Breaking operation.'.format(time_stamp())
                logging.info(log)
                logs.append(log + '<br>')
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
            mav = low_frame_indicated.iloc[-1]['MAV']
            roc30 = low_frame_indicated.iloc[-1]['roc30']
            logs.append(log + '<br>')
            if mode != 'simulator':
                condition, crypto_balance, fiat_balance = get_condition(crypto_currency, fiat_currency, closing_price)
            log = 'Event is: {}. BB crossing is: {}'.format(event, bb_cross)
            logging.info(log)
            log = 'Condition is: {}'.format(condition)
            logging.info(log)
            logs.append(log + '<br>')
            if new_candle_time > candle_time:  # wait first 30min candle to close
                if condition == 'Sell':
                    if bb_cross != 0 and event != 0:
                        prime_predictionS, meta_predictionS = sell_evaluation(high_frame_indicated,
                                                                              mid_frame_indicated,
                                                                              low_frame_indicated,
                                                                              pms, mms)
                        if prime_predictionS != meta_predictionS:
                            if mode == 'simulator':
                                log = '{} Prime Prediction: {} Meta Prediction {}. Simulating Sell at {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS, closing_price)
                                logging.info(log)
                                logs.append(log + '<br>')
                                trades.append(log)
                                condition = 'Buy'
                            elif mode == 'consulting':
                                log = '{} Prime Prediction {} Meta Prediction {}. Consulting Sell at {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS, closing_price)
                                logging.info(log)
                                logs.append(log + '<br>')
                                trades.append(log)
                            elif mode == 'trading':
                                log = '{} Prime Prediction {} Meta Prediction {}. Sale at {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS, closing_price)
                                logging.info(log)
                                logs.append(log + '<br>')
                                asset_vol = (fiat_balance - fiat_balance * fee) / closing_price
                                tx = add_order(order_type, condition, asset_vol, closing_price, crypto_currency,
                                               fiat_currency)
                                log = tx
                                logging.info(log)
                                trades.append(log)
                        else:
                            ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                            if ret > 0 and roc30 > 0:
                                limit = closing_price * (1 + (ret + (roc30 / 100)))
                                stop = closing_price * (1 - (ret + (roc30 / 100)))
                                log = '{} Limit reset to {}. Stop reset to {}.' \
                                    .format(time_stamp(), limit, stop)
                                logging.info(log)
                                logs.append(log + '<br>')
                                trades.append(log)
                elif condition == 'Buy':
                    if bb_cross != 0 and event != 0 and mav > fee:
                        prime_predictionB, meta_predictionB = buy_evaluation(high_frame_indicated,
                                                                             mid_frame_indicated,
                                                                             low_frame_indicated,
                                                                             pmb, mmb)
                        ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                        if prime_predictionB == meta_predictionB and ret > fee and roc30 > 0:
                            if mode == 'simulator':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionB, meta_predictionB)
                                logging.info(log)
                                logs.append(log + '<br>')
                                trades.append(log)
                                condition = 'Sell'
                            elif mode == 'consulting':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionB, meta_predictionB)
                                logging.info(log)
                                logs.append(log + '<br>')
                                trades.append(log)
                            elif mode == 'trading':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionB, meta_predictionB)
                                logging.info(log)
                                logs.append(log + '<br>')
                                asset_vol = (fiat_balance - fiat_balance * fee) / closing_price
                                tx = add_order(order_type, condition, asset_vol, closing_price, crypto_currency,
                                               fiat_currency)
                                log = tx
                                logging.info(log)
                                trades.append(log)
                log = '{} Waiting 30 min candle close.'.format(time_stamp())
                logging.info(log)
                logs.append(log + '<br>')
                time.sleep(1799)  # wait one 30min candle - 1 second
            else:
                log = '{} Waiting 30 min candle close.'.format(time_stamp())
                logging.info(log)
                logs.append(log + '<br>')
                time.sleep(59)  # wait one 1min - 1 second for first candle to close
    else:
        activation()
        log = 'Your product licence is not active. Please activate or contact technical support. ' \
              'Hermes_algotrading@proton.me'
        logging.info(log)
        logs.append(log + '<br>')
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
        'event': event,
        'bb_cross': bb_cross
    }
