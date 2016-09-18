from ._api import *
import numpy as np
from math import sqrt
# Fix up vector types

DoubleVector.size = lambda self: len(self)

DoubleVector_FromNdArray = lambda x: DoubleVector.from_numpy(x)


def VectorString(v):
    return str(v.to_numpy())


DoubleVector.__str__ = lambda self: VectorString(self)


def ShowUtcTime(v):
    utc = Calendar()
    return "[" + ",".join([cal.to_string(t) for t in v]) + "]"


UtcTimeVector.size = lambda self: len(self)
UtcTimeVector.__str__ = lambda self: ShowUtcTime

IntVector.size = lambda self: len(self)
IntVector.__str__ = lambda self: VectorString(self)

StringVector.size = lambda self: len(self)

# fix up BW and pythonic syntax for TsVector

TsVector.size = lambda self: len(self)
TsVector.push_back = lambda self, ts: self.append(ts)
# and this is for easy syntax:
TsVector.percentiles = lambda self, ta, percentile_list: percentiles(self, ta, percentile_list)

TargetSpecificationVector.size = lambda self: len(self)

# fix bw. stl name
UtcTimeVector.push_back = lambda self, x: self.append(x)
IntVector.push_back = lambda self, x: self.append(x)
DoubleVector.push_back = lambda self, x: self.append(x)
StringVector.push_back = lambda self, x: self.append(x)

# FIx up YMDhms
YMDhms.__str__ = lambda self: "YMDhms({0},{1},{2},{3},{4},{5})".format(self.year, self.month, self.day, self.hour, self.minute, self.second)

# Fix up GeoPoint
GeoPoint.__str__ = lambda self: "GeoPoint({0},{1},{2})".format(self.x, self.y, self.z)
GeoPoint_difference = lambda a, b: GeoPoint.difference(a, b)
GeoPoint_xy_distance = lambda a, b: GeoPoint.xy_distance(a, b)

# Fix up LandTypeFractions

LandTypeFractions.__str__ = lambda self: "LandTypeFractions(glacier={0},lake={1},reservoir={2},forest={3},unspecified={4})".format(self.glacier(), self.lake(), self.reservoir(),
                                                                                                                                   self.forest(), self.unspecified())


# Fix up GeoCellData
def StrGeoCellData(gcd):
    return "GeoCellData(mid_point={0},catchment_id={1},area={2},ltf={3})".format(str(gcd.mid_point()), gcd.catchment_id(), gcd.area(), str(gcd.land_type_fractions_info()))


GeoCellData.__str__ = lambda self: StrGeoCellData(self)

# Fix up UtcPeriod
UtcPeriod.to_string = lambda self: str(self)

# Fix up Timeaxis
Timeaxis.__str__ = lambda self: "Timeaxis({0},{1},{2})".format(Calendar().to_string(self.start), self.delta_t, self.n)
Timeaxis.__len__ = lambda self: self.size()
Timeaxis.__call__ = lambda self, i: self.period(i)

Timeaxis2.__len__ = lambda self: self.size()
Timeaxis2.__str__ = lambda self: "GenericTimeaxis {0} {1} {2}".format(self.timeaxis_type, Calendar().to_string(self.total_period()), self.size())
Timeaxis2.__call__ = lambda self, i: self.period(i)

# fix up property on timeseries
Timeseries.time_axis = property(lambda self: self.get_time_axis(), doc="returns the time_axis of the timeseries")

Timeseries.__len__ = lambda self: self.size()
Timeseries.v = property(lambda self: self.values,doc="returns the point-values of timeseries, alias for .values")

Timeseries.kling_gupta = lambda self, other_ts, s_r=1.0, s_a=1.0, s_b=1.0: kling_gupta(self, other_ts, self.get_time_axis(), s_r, s_a, s_b)
Timeseries.kling_gupta.__doc__ = \
"""
computes the kling_gupta correlation using self as observation, and self.time_axis as
the comparison time-axis

Parameters
----------
other_ts : Timeseries
 the predicted/calculated time-series to correlate
s_r : float
 the kling gupta scale r factor(weight the correlation of goal function)
s_a : float
 the kling gupta scale a factor(weight the relative average of the goal function)
s_b : float
 the kling gupta scale b factor(weight the relative standard deviation of the goal function)

Return
------
KGEs : float

"""

Timeseries.nash_sutcliffe = lambda self,other_ts: nash_sutcliffe(self, other_ts, self.get_time_axis())
Timeseries.nash_sutcliffe.__doc__ = \
"""
Computes the Nash-Sutcliffe model effiency coefficient (n.s)
for the two time-series over the specified time_axis
Ref:  http://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
Parameters
----------
observed_ts : Timeseries
 the observed time-series
model_ts : Timeseries
 the time-series that is the model simulated / calculated ts
time_axis : Timeaxis2
 the time-axis that is used for the computation
Return
------
 float: The n.s performance, that have a maximum at 1.0
"""

TsFixed.values = property(lambda self:self.v,doc="returns the point values, .v of the timeseries")
TsFixed.time_axis = property(lambda self: self.get_time_axis(), doc="returns the time_axis of the timeseries")
TsPoint.values = property(lambda self:self.v,doc="returns the point values, .v of the timeseries")
TsPoint.time_axis = property(lambda self: self.get_time_axis(), doc="returns the time_axis of the timeseries")

def ta_iter(x):
    x.counter = 0
    return x


def ta_next(ta):
    if ta.counter >= len(ta):
        del ta.counter
        raise StopIteration
    ta.counter += 1
    return ta(ta.counter - 1)


Timeaxis.__iter__ = lambda self: ta_iter(self)
Timeaxis.__next__ = lambda self: ta_next(self)

Timeaxis2.__iter__ = lambda self: ta_iter(self)
Timeaxis2.__next__ = lambda self: ta_next(self)

# Fix up PointTimeaxis
PointTimeaxis.__str__ = lambda self: "PointTimeaxis(total_period={0}, n={1} )".format(str(self.total_period()), len(self))
PointTimeaxis.__len__ = lambda self: self.size()
PointTimeaxis.__call__ = lambda self, i: self.period(i)
PointTimeaxis.__iter__ = lambda self: ta_iter(self)
PointTimeaxis.__next__ = lambda self: ta_next(self)

# Fix up ARegionEnvironment
TemperatureSource.vector_t = TemperatureSourceVector
PrecipitationSource.vector_t = PrecipitationSourceVector
RadiationSource.vector_t = RadiationSourceVector
RelHumSource.vector_t = RelHumSourceVector
WindSpeedSource.vector_t = WindSpeedSourceVector

def np_array(dv):
    """
    convert flattened double-vector to numpy array
    Parameters
    ----------
    dv

    Returns
    -------
    numpy array.
    """
    f = dv.to_numpy()
    n = int(sqrt(dv.size()))
    m = f.reshape(n, n)
    return m

# fixup kalman state
KalmanState.x = property(lambda self: KalmanState.get_x(self).to_numpy(),doc="represents the current bias estimate, kalman.state.x")
KalmanState.k = property(lambda self: KalmanState.get_k(self).to_numpy(),doc="represents the current kalman gain factors, kalman.state.k")
KalmanState.P = property(lambda self: np_array(KalmanState.get_P(self)),doc="returns numpy array of kalman.state.P, the nxn covariance matrix")
KalmanState.W = property(lambda self: np_array(KalmanState.get_W(self)),doc="returns numpy array of kalman.state.W, the nxn noise matrix")

#fixup KalmanBiasPredictor
def KalmanBiasPredictor_update_with_forecast(bp,fc_set,obs,time_axis):
    """

    Parameters
    ----------
    bp
    fc_set : TemperatureSourceVector or TsVector
    obs : Timeseries
    time_axis : Timeaxis

    Returns
    -------
    nothing
    """
    if isinstance(fc_set, TemperatureSourceVector):
        KalmanBiasPredictor.update_with_geo_forecast(bp,fc_set,obs,time_axis)
    else:
        KalmanBiasPredictor.update_with_forecast_vector(bp,fc_set,obs,time_axis)

KalmanBiasPredictor.update_with_forecast = KalmanBiasPredictor_update_with_forecast