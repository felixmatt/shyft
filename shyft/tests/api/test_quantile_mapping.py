import numpy as np
# from numpy.testing import assert_array_almost_equal
import unittest
from shyft.api import Calendar
from shyft.api import TimeSeries
from shyft.api import TimeAxis
from shyft.api import TsVector
from shyft.api import TsVectorSet
from shyft.api import point_interpretation_policy as ts_point_fx
from shyft.api import deltahours
from shyft.api import DoubleVector as dv
from shyft.api import no_utctime
from shyft.api import quantile_map_forecast


class QuantileMapping(unittest.TestCase):
    def test_forecast(self):
        fx_avg = ts_point_fx.POINT_AVERAGE_VALUE
        utc = Calendar()
        ta = TimeAxis(utc.time(2017, 1, 1, 0, 0, 0), deltahours(24), 4)
        historical_data = TsVector()

        forecast_sets = TsVectorSet()
        weight_sets = dv()
        num_historical_data = 56

        # Let's make three sets, one of two elements, one of three, and one of
        # four.

        forecasts_1 = TsVector()
        forecasts_2 = TsVector()
        forecasts_3 = TsVector()

        forecasts_1.append(TimeSeries(ta, dv([13.4, 15.6, 17.1, 19.1]), fx_avg))
        forecasts_1.append(TimeSeries(ta, dv([34.1, 2.40, 43.9, 10.2]), fx_avg))
        forecast_sets.append(forecasts_1)
        weight_sets.append(5.0)

        forecasts_2.append(TimeSeries(ta, dv([83.1, -42.2, 0.4, 23.4]), fx_avg))
        forecasts_2.append(TimeSeries(ta, dv([15.1, 6.500, 4.2, 2.9]), fx_avg))
        forecasts_2.append(TimeSeries(ta, dv([53.1, 87.90, 23.8, 5.6]), fx_avg))
        forecast_sets.append(forecasts_2)
        weight_sets.append(9.0)

        forecasts_3.append(TimeSeries(ta, dv([1.5, -1.9, -17.2, -10.0]), fx_avg))
        forecasts_3.append(TimeSeries(ta, dv([4.7, 18.2, 15.3000, 8.9]), fx_avg))
        forecasts_3.append(TimeSeries(ta, dv([-45.2, -2.3, 80.2, 71.0]), fx_avg))
        forecasts_3.append(TimeSeries(ta, dv([45.1, -92.0, 34.4, 65.8]), fx_avg))
        forecast_sets.append(forecasts_3)
        weight_sets.append(3.0)

        for i in range(num_historical_data):
            historical_data.append(TimeSeries(ta, dv.from_numpy(np.random.random(ta.size()) * 50.0), fx_avg))

        # need one more exposed from core here: auto historical_order = qm::quantile_index<tsa_t>(historical_data, ta);

        interpolation_start = no_utctime
        interpolation_end = no_utctime
        # Act
        result = quantile_map_forecast(forecast_sets, weight_sets, historical_data, ta, interpolation_start,interpolation_end)

        self.assertIsNotNone(result)
        self.assertEqual(len(result),num_historical_data)
        for ts in result:
            self.assertEqual(ts.size(),ta.size())
        # Assert
        # for (size_t i=0; i<num_historical_data; ++i) {
        #    if (i < 4) {
        #        FAST_CHECK_EQ(result[historical_order[0][i]].value(0), -45.2);
        #    } else if (i < 7) {
        #        FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 1.5);
        #    } else if (i < 11) {
        #        FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 4.7);
        #    } else if (i < 16) {
        #        FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 13.4);
        #    } else if (i < 26) {
        #        FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 15.1);
        #    } else if (i < 32) {
        #        FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 34.1);
        #    } else if (i < 35) {
        #        FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 45.1);
        #    } else if (i < 45) {
        #        FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 53.1);
        #    } else {
        #        FAST_CHECK_EQ(result[historical_order[0][i]].value(0), 83.1);
        #    }

        #    if (i < 4) {
        #       FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -92.0);
        #    } else if (i < 14) {
        #        FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -42.2);
        #    } else if (i < 17) {
        #        FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -2.3);
        #    } else if (i < 21) {
        #        FAST_CHECK_EQ(result[historical_order[1][i]].value(1), -1.9);
        #    } else if (i < 26) {
        #        FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 2.4);
        #    } else if (i < 36) {
        #        FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 6.5);
        #    } else if (i < 42) {
        #        FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 15.6);
        #    } else if (i < 45) {
        #        FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 18.2);
        #    } else {
        #        FAST_CHECK_EQ(result[historical_order[1][i]].value(1), 87.9);
        #    }

        #    if (i < 4) {
        #        FAST_CHECK_EQ(result[historical_order[2][i]].value(2), -17.2);
        #    } else if (i < 14) {
        #        FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 0.4);
        #    } else if (i < 24) {
        #        FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 4.2);
        #    } else if (i < 27) {
        #        FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 15.3);
        #    } else if (i < 33) {
        #        FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 17.1);
        #    } else if (i < 43) {
        #        FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 23.8);
        #    } else if (i < 47) {
        #        FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 34.4);
        #    } else if (i < 52) {
        #        FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 43.9);
        #    } else {
        #        FAST_CHECK_EQ(result[historical_order[2][i]].value(2), 80.2);
        #    }

        #   if (i < 4) {
        #        FAST_CHECK_EQ(result[historical_order[3][i]].value(3), -10.0);
        #    } else if (i < 14) {
        #        FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 2.9);
        #    } else if (i < 24) {
        #        FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 5.6);
        #    } else if (i < 27) {
        #        FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 8.9);
        #    } else if (i < 33) {
        #        FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 10.2);
        #    } else if (i < 39) {
        #        FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 19.1);
        #    } else if (i < 49) {
        #        FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 23.4);
        #   } else if (i < 52) {
        #        FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 65.8);
        #    } else {
        #        FAST_CHECK_EQ(result[historical_order[3][i]].value(3), 71.0);
        #    }
        # }
