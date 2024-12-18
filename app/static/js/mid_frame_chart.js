var options = {
          series: [
          {
          name: 'Momentum frame: ',
          type: 'candlestick',
          data: []
          }
          ],
          chart: {
          height: 220,
          type: 'line',
          stacked: false
        },
        title: {
          text: 'Momentum frame',
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

var mid_frame_chart = new ApexCharts(document.querySelector("#mid_frame_chart"), options);
mid_frame_chart.render();

var intervalID = setInterval(update_values, 1000);

function update_values() {
    $.getJSON($SCRIPT_ROOT + '/update',
        function(data_update) {
            mid_frame_chart.updateSeries([{
              name: '4 hours',
              data: data_update.mid_chart_data
              }
            ])
        }
    );
};