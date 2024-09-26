import threading
from app.back.bot import Prelderbot, data_feed, signal_handler, multiPrelderbot
from app.back.kraken import cancel_order, time_stamp
from app import app
from flask import render_template, request, jsonify
import logging
import joblib

# Default values given as preset
asset_a = None
asset_b = None
mode = None

# Flask logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def initialize_models():
    init_pmb_eth = joblib.load('PrimeModelBuyETHEUR0.pkl')
    init_mmb_eth = joblib.load('MetaModelBuyETHEUR0.pkl')
    init_pms_eth = joblib.load('PrimeModelSellETHEUR0.pkl')
    init_mms_eth = joblib.load('MetaModelSellETHEUR0.pkl')
    init_mr_eth = joblib.load('ModelRiskETHEUR0.pkl')
    init_pmb_btc = joblib.load('PrimeModelBuyBTCEUR0.pkl')
    init_mmb_btc = joblib.load('MetaModelBuyBTCEUR0.pkl')
    init_pms_btc = joblib.load('PrimeModelSellBTCEUR0.pkl')
    init_mms_btc = joblib.load('MetaModelSellBTCEUR0.pkl')
    init_mr_btc = joblib.load('ModelRiskBTCEUR0.pkl')
    init_pmb_dot = joblib.load('PrimeModelBuyDOTEUR0.pkl')
    init_mmb_dot = joblib.load('MetaModelBuyDOTEUR0.pkl')
    init_pms_dot = joblib.load('PrimeModelSellDOTEUR0.pkl')
    init_mms_dot = joblib.load('MetaModelSellDOTEUR0.pkl')
    init_mr_dot = joblib.load('ModelRiskDOTEUR0.pkl')
    return (init_pmb_eth, init_mmb_eth, init_pms_eth, init_mms_eth, init_mr_eth,
            init_pmb_btc, init_mmb_btc, init_pms_btc, init_mms_btc, init_mr_btc,
            init_pmb_dot, init_mmb_dot, init_pms_dot, init_mms_dot, init_mr_dot)


(pmb_eth, mmb_eth, pms_eth, mms_eth, mr_eth,
 pmb_btc, mmb_btc, pms_btc, mms_btc, mr_btc,
 pmb_dot, mmb_dot, pms_dot, mms_dot, mr_dot) = initialize_models()

asset_pairs = [('DOT', 'EUR', pmb_dot, mmb_dot, pms_dot, mms_dot, mr_dot),
               ('ETH', 'EUR', pmb_btc, mmb_btc, pms_btc, mms_btc, mr_btc)]
# ('BTC', 'EUR', pmb_eth, mmb_eth, pms_eth, mms_eth, mr_eth),
@app.route('/')
@app.route('/home')
def main():
    return render_template('overview.html')


@app.route('/control', methods=['POST', 'GET'])
def control():
    global asset_a, asset_b, mode
    if request.method == 'POST':
        try:
            asset_a = request.form['assetA']
            asset_b = request.form['assetB']
            mode = request.form['mode']
        except Exception as e:
            logging.info(e)
    else:
        render_template('control.html')
    return render_template('control.html')


@app.route('/manual')
def doc():
    return render_template('doc.html')


@app.route('/history')
def history():
    return render_template('history.html')


@app.route('/update', methods=['GET'])
def update():
    dt = data_feed()
    dt['current_time'] = time_stamp()
    dt['crypto_currency'] = asset_a
    dt['fiat_currency'] = asset_b
    dt['mode'] = mode
    return jsonify(dt)


@app.route('/start')
def starter():
    if mode is None:
        logging.info('Mode not set.')
        return render_template('control.html')
    elif asset_a is None or asset_b is None:
        logging.info('Set assets.')
        return render_template('control.html')
    else:
        # trading = threading.Thread(target=Prelderbot, args=(mode, asset_a, asset_b, pmb_eth, mmb_eth, pms_eth, mms_eth, mr_eth))
        trading = threading.Thread(target=multiPrelderbot, args=(mode, asset_pairs))
        trading.daemon = True
        trading.start()
        return render_template('overview.html')


@app.route('/stop')
def stopper():
    signal_handler()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return render_template('overview.html')


@app.route('/cancel')
def canceler():
    cancel_order()
    return render_template('overview.html')
