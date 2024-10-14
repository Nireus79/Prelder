var intervalID = setInterval(update_values, 1000);
function update_values() {
    $.getJSON($SCRIPT_ROOT + '/update',
        function(data) {
                $('#current_time').text('Current time: '+data.current_time);
                $('#mode').text('Mode: '+data.mode);
                $('#condition').text('Evaluating to '+data.condition+':');
                $('#crypto_currency').text(data.crypto_currency);
                $('#fiat_currency').text('for  '+data.fiat_currency);
                $('#crypto_balance').text('asset A: '+data.crypto_balance);
                $('#fiat_balance').text('asset B: '+data.fiat_balance);
                $('#event').text('Cusum event: '+data.event);
                $('#bb_cross').text('Bollinger crossing: '+data.bb_cross);
                $('#prime_prediction').text('prime pred: '+data.prime_prediction);
                $('#meta_prediction').text('meta pred: '+data.meta_prediction);
                $('#ret').text('return pred: '+data.ret);
                $('#price').text('Closing price: '+data.price);
                $('#limit').text('Buy limit: '+data.limit);
                $('#stop').text('Stop loss: '+data.stop);
                $('#roc10').text('roc10: '+data.roc10);
                $('#log').text('Log: '+data.log);
        });
};