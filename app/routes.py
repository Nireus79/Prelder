import threading
from app.back.bot import Prelderbot, data_feed, signal_handler
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
    init_pmb = joblib.load('PrimeModelBuy1.pkl')
    init_mmb = joblib.load('MetaModelBuy1.pkl')
    init_pms = joblib.load('PrimeModelSell0.pkl')
    init_mms = joblib.load('MetaModelSell0.pkl')
    init_mr = joblib.load('ModelRisk1.pkl')
    return init_pmb, init_mmb, init_pms, init_mms, init_mr


pmb, mmb, pms, mms, mr = initialize_models()


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
        trading = threading.Thread(target=Prelderbot, args=(mode, asset_a, asset_b, pmb, mmb, pms, mms, mr))
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
