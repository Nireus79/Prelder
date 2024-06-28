var options = {
          series: [
          {
          name: 'Trend frame: ',
          type: 'candlestick',
          data: []
        },{
          name: 'Closing price: ',
          type: 'line',
          data: []
        },{
          name: 'EMA20: ',
          type: 'line',
          data: []
        },{
          name: 'EMA3: ',
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
          text: 'Trend frame',
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

var high_frame_chart = new ApexCharts(document.querySelector("#high_frame_chart"), options);

high_frame_chart.render();

var intervalID = setInterval(update_values, 1000);

function update_values() {
    $.getJSON($SCRIPT_ROOT + '/update',
        function(data_update) {
            high_frame_chart.updateSeries([{
              name: '1 Day',
              data: data_update.high_chart_data
            },
            {
              name: 'Closing price: ' + data_update.price
            },
            {
              name: 'EMA20: ' + data_update.high_ema20[data_update.high_ema20.length - 1]['y'],
              data: data_update.high_ema20
            },
            {
              name: 'EMA3: ' + data_update.high_ema3[data_update.high_ema3.length - 1]['y'],
              data: data_update.high_ema3
            },
            {
              name: 'TrD20: ' + data_update.high_TrD20
              },
            {
              name: 'TrD3: ' + data_update.high_TrD3
              }
            ]);
        }
    );
};