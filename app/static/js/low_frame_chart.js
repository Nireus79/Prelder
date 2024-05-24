var options = {
          series: [
          {
          name: 'Correction frame: ',
          type: 'candlestick',
          data: []
        },{
          name: 'Buy limit: ',
          type: 'line',
          data: []
        },{
          name: 'Stop loss: ',
          type: 'line',
          data: []
        }
        ],
        chart: {
          height: 300,
          type: 'line',
          stacked: false
        },
        title: {
          text: 'Correction frame',
          align: 'left'
        },
        stroke: {
          width: [3, 1]
        },
        tooltip: {
          shared: true,
          custom: [function({seriesIndex, dataPointIndex, w}) {
            return w.globals.series[seriesIndex][dataPointIndex]
          }, function({ seriesIndex, dataPointIndex, w }) {
            var o = w.globals.seriesCandleO[seriesIndex][dataPointIndex]
            var h = w.globals.seriesCandleH[seriesIndex][dataPointIndex]
            var l = w.globals.seriesCandleL[seriesIndex][dataPointIndex]
            var c = w.globals.seriesCandleC[seriesIndex][dataPointIndex]
            return (
              ''
            )
          }]
        },
        xaxis: {
          type: 'datetime'
        }
     };

var low_frame_chart = new ApexCharts(document.querySelector("#low_frame_chart"), options);

low_frame_chart.render();

var intervalID = setInterval(update_values, 1000);

function update_values() {
    $.getJSON($SCRIPT_ROOT + '/update',
        function(data_update) {
            low_frame_chart.updateSeries([{
              name: '30 minutes',
              data: data_update.low_chart_data
            },{
              name: 'Buy limit: ' + data_update.low_limit[data_update.low_limit.length - 1]['y'],
              data: data_update.low_limit
            },{
              name: 'Stop loss: ' + data_update.low_stop[data_update.low_stop.length - 1]['y'],
              data: data_update.low_stop
              },{
              name: 'ema 20: ' + data_update.low_ave[data_update.low_ave.length - 1]['y'],
              data: data_update.low_ave
            },{
              name: 'upper bb: ' + data_update.low_upper[data_update.low_upper.length - 1]['y'],
              data: data_update.low_upper
            },{
              name: 'lower bb: ' + data_update.low_lower[data_update.low_lower.length - 1]['y'],
              data: data_update.low_lower
            },{
              name: 'Volatility: ' + data_update.volatility
              },{
              name: 'ROC30: ' + data_update.roc30
              }
            ]);
        }
    );
};