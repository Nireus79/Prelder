import logging
import time
import threading
import winsound
from app.back.kraken import Api, time_stamp, \
    add_order, cancel_order, get_condition, high_data, mid_data, low_data, indicators
from app.back.spring import check, activation
from sklearn.preprocessing import Normalizer, normalize
import joblib
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
crypto_balance = None
fiat_balance = 0
limit = 0
stop = 0
ret = 0
minRet = 0.026
closing_price = None

runningHighFrame = None
runningMidFrame = None
runningLowFrame = None
high_chart_data = None
mid_chart_data = None
low_chart_data = None
high_ema13 = None
high_macd = None
high_d = None
high_ds = None
mid_d = None
mid_ds = None
mid_rs = None
l_atr = None
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
    # print('normalizedX -----')
    # print(normalized.head())
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
    global high_chart_data, high_ema13, high_macd, high_d, high_ds, mid_chart_data, mid_d, mid_ds, mid_rs, \
        low_chart_data, low_limit, low_stop, l_atr
    limit_data = []
    stop_data = []
    high_candles, e13, mac = high_data(high_frame)
    mid_candles, d, ds, rs = mid_data(mid_frame)
    low_candles = low_data(low_frame)
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
    high_ema13 = e13[-20:]
    high_macd = mac
    mid_chart_data = mid_candles[-20:]
    mid_d = d
    mid_ds = ds
    mid_rs = rs
    low_chart_data = low_candles[-10:]
    low_limit = limit_data[-10:]
    low_stop = stop_data[-10:]


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


def initialize_models():
    pmb = joblib.load('PrimeModelBuy.pkl')
    mmb = joblib.load('MetaModelBuy.pkl')
    pms = joblib.load('PrimeModelSell.pkl')
    mms = joblib.load('MetaModelSell.pkl')
    mr = joblib.load('ModelRisk.pkl')
    return pmb, mmb, pms, mms, mr


def Prelderbot(mode, crypto_currency, fiat_currency):
    global condition, limit, stop, ret, log, trend_24h, trend_4h, buy_flag_4h, buy_flag_1h, sell_flag_4h, sell_flag_1h, \
        crypto_balance, fiat_balance, closing_price, runningHighFrame, runningMidFrame, runningLowFrame
    licence = True  # check()['license_active'] # TODO activate licence check
    pmb, mmb, pms, mms, mr = initialize_models()
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
        closing_price = low_frame_indicated.iloc[-1]['close']
        candle_time = low_frame_indicated.iloc[-1]['time']
        event = low_frame_indicated.iloc[-1]['event']
        bb_cross = low_frame_indicated.iloc[-1]['bb_cross']
        mav = low_frame_indicated.iloc[-1]['MAV']
        roc30 = low_frame_indicated.iloc[-1]['roc30']
        condition = get_condition(crypto_currency, fiat_currency, closing_price)
        if mode == 'simulator':
            condition = 'Buy'
        log = 'event is: {}. BB crossing is: {}'.format(event, bb_cross)
        logging.info(log)
        logs.append(log + '<br>')
        log = 'Condition is: {}'.format(condition)
        logging.info(log)
        logs.append(log + '<br>')
        if condition == 'Sell':
            if limit == stop == 0:
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
                            # asset_vol = (fiat_balance - fiat_balance * 0.0026) / closing_price
                            # tx = add_order('market', condition, asset_vol, closing_price, crypto_currency, fiat_currency)
                            # log = tx
                            # logging.info(log)
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
                        condition = 'Buy'
                    elif mode == 'consulting':
                        log = '{} Closing price < stop. Consulting Sale at {}.'.format(time_stamp(), closing_price)
                        logs.append(log + '<br>')
                    elif mode == 'trading':
                        log = '{} Closing price < stop. Sale at {}.'.format(time_stamp(), closing_price)
                        logging.info(log)
                        logs.append(log + '<br>')
                        # asset_vol = (fiat_balance - fiat_balance * 0.0026) / closing_price
                        # tx = add_order('market', condition, asset_vol, closing_price, crypto_currency, fiat_currency)
                        # log = tx
                        # logging.info(log)
                elif closing_price > limit:
                    if event != 0 and bb_cross != 0:
                        prime_predictionS, meta_predictionS = sell_evaluation(high_frame_indicated,
                                                                              mid_frame_indicated,
                                                                              low_frame_indicated,
                                                                              pms, mms)
                        if prime_predictionS != meta_predictionS:
                            if mode == 'simulator':
                                log = '{} Prime Prediction: {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS)
                                logging.info(log)
                                logs.append(log + '<br>')
                            elif mode == 'consulting':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS)
                                logging.info(log)
                                logs.append(log + '<br>')
                            elif mode == 'trading':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS)
                                logging.info(log)
                                logs.append(log + '<br>')
                                # asset_vol = (fiat_balance - fiat_balance * 0.0026) / closing_price
                                # tx = add_order('market', condition, asset_vol, closing_price, crypto_currency, fiat_currency)
                                # log = tx
                                # logging.info(log)
                        else:
                            ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                            if ret > 0 and roc30 > 0:
                                limit = closing_price * (1 + (ret + (roc30 / 100)))
                                stop = closing_price * (1 - (ret + (roc30 / 100)))
                                log = '{} Limit reset to {}. Stop reset to {}.' \
                                    .format(time_stamp(), limit, stop)
                                logging.info(log)
                                logs.append(log + '<br>')
        elif condition == 'Buy':
            if event != 0 and bb_cross != 0 and mav > minRet:
                prime_predictionB, meta_predictionB = buy_evaluation(high_frame_indicated,
                                                                     mid_frame_indicated,
                                                                     low_frame_indicated,
                                                                     pmb, mmb)
                ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                if prime_predictionB == meta_predictionB and ret > minRet and roc30 > 0:
                    limit = closing_price * (1 + (ret + (roc30 / 100)))
                    stop = closing_price * (1 - (ret + (roc30 / 100)))
                    if mode == 'simulator':
                        log = '{} Simulating Buy at {}. Limit set to {}. Stop set to {}.' \
                            .format(time_stamp(), closing_price, limit, stop)
                        logging.info(log)
                        logs.append(log + '<br>')
                        condition = 'Sell'
                    elif mode == 'consulting':
                        log = '{} Consulting Buy at {}. Limit set to {}. Stop set to {}.' \
                            .format(time_stamp(), closing_price, limit, stop)
                        logging.info(log)
                        logs.append(log + '<br>')
                    elif mode == 'trading':
                        log = '{} Buy at {}. Limit set to {}. Stop set to {}' \
                            .format(time_stamp(), closing_price, limit, stop)
                        logging.info(log)
                        logs.append(log + '<br>')
                        # asset_vol = (fiat_balance - fiat_balance * 0.0026) / closing_price
                        # tx = add_order('market', condition, asset_vol, closing_price, crypto_currency, fiat_currency)
                        # log = tx
                        # logging.info(log)
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
            new_candle_time = low_frame_indicated.iloc[-1]['time']
            closing_price = low_frame_indicated.iloc[-1]['close']
            event = low_frame_indicated.iloc[-1]['event']
            bb_cross = low_frame_indicated.iloc[-1]['bb_cross']
            mav = low_frame_indicated.iloc[-1]['MAV']
            roc30 = low_frame_indicated.iloc[-1]['roc30']
            logs.append(log + '<br>')
            if mode != 'simulator':
                condition = get_condition(crypto_currency, fiat_currency, closing_price)
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
                                log = '{} Prime Prediction: {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS)
                                logging.info(log)
                                logs.append(log + '<br>')
                                condition = 'Buy'
                            elif mode == 'consulting':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS)
                                logging.info(log)
                                logs.append(log + '<br>')
                            elif mode == 'trading':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionS, meta_predictionS)
                                logging.info(log)
                                logs.append(log + '<br>')
                                # asset_vol = (fiat_balance - fiat_balance * 0.0026) / closing_price
                                # tx = add_order('market', condition, asset_vol, closing_price, crypto_currency, fiat_currency)
                                # log = tx
                                # logging.info(log)
                elif condition == 'Buy':
                    if bb_cross != 0 and event != 0 and mav > minRet:
                        prime_predictionB, meta_predictionB = buy_evaluation(high_frame_indicated,
                                                                             mid_frame_indicated,
                                                                             low_frame_indicated,
                                                                             pmb, mmb)
                        ret = ret_evaluation(high_frame_indicated, mid_frame_indicated, low_frame_indicated, mr)
                        if prime_predictionB == meta_predictionB and ret > minRet and roc30 > 0:
                            if mode == 'simulator':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionB, meta_predictionB)
                                logging.info(log)
                                logs.append(log + '<br>')
                                condition = 'Sell'
                            elif mode == 'consulting':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionB, meta_predictionB)
                                logging.info(log)
                                logs.append(log + '<br>')
                            elif mode == 'trading':
                                log = '{} Prime Prediction {} Meta Prediction {}.' \
                                    .format(time_stamp(), prime_predictionB, meta_predictionB)
                                logging.info(log)
                                logs.append(log + '<br>')
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
        'crypto_balance': crypto_balance,
        'fiat_balance': fiat_balance,
        'runningHighFrame': runningHighFrame,
        'runningMidFrame': runningMidFrame,
        'runningLowFrame': runningLowFrame,
        'trend_24h': str(trend_24h),
        'trend_4h': str(trend_4h),
        'buy_flag_4h': str(buy_flag_4h),
        'sell_flag_4h': str(sell_flag_4h),
        'buy_flag_1h': str(buy_flag_1h),
        'sell_flag_1h': str(sell_flag_1h),
        'high_chart_data': high_chart_data,
        'mid_chart_data': mid_chart_data,
        'high_ema13': high_ema13,
        'high_macd': high_macd,
        'mid_d': mid_d,
        'mid_ds': mid_ds,
        'mid_rs': mid_rs,
        'low_chart_data': low_chart_data,
        'price': closing_price,
        'l_atr': l_atr,
        'low_limit': low_limit,
        'low_stop': low_stop
    }


Prelderbot('simulator', 'ETH', 'EUR')
