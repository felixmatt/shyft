import inspect
import traceback
import warnings
import functools

from shyft.api._api import *
import numpy as np
from math import sqrt


def deprecated(message: str = ''):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used first time and filter is set for show DeprecationWarning.
    """

    def decorator_wrapper(func):
        @functools.wraps(func)
        def function_wrapper(*args, **kwargs):
            current_call_source = '|'.join(traceback.format_stack(inspect.currentframe()))
            if current_call_source not in function_wrapper.last_call_source:
                warnings.warn("Class.method {} is now deprecated! {}".format(func, message),
                              category=DeprecationWarning, stacklevel=2)
                function_wrapper.last_call_source.add(current_call_source)

            return func(*args, **kwargs)

        function_wrapper.last_call_source = set()

        return function_wrapper

    return decorator_wrapper


# Fix up vector types

DoubleVector.size = lambda self: len(self)

DoubleVector_FromNdArray = lambda x: DoubleVector.from_numpy(x)


def VectorString(v):
    return str(v.to_numpy())


DoubleVector.__str__ = lambda self: VectorString(self)

Calendar.__str__ = lambda self: "Calendar('{0}')".format(self.tz_info.name())


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
# and this is for bw.compat
def percentiles(tsv:TsVector,time_axis:TimeAxis,percentile_list:IntVector)->TsVector:
    return tsv.percentiles(time_axis,percentile_list)

TargetSpecificationVector.size = lambda self: len(self)

# fix bw. stl name
UtcTimeVector.push_back = lambda self, x: self.append(x)
IntVector.push_back = lambda self, x: self.append(x)
DoubleVector.push_back = lambda self, x: self.append(x)
StringVector.push_back = lambda self, x: self.append(x)

# FIx up YMDhms
YMDhms.__str__ = lambda self: "YMDhms({0},{1},{2},{3},{4},{5})".format(self.year, self.month, self.day, self.hour,
                                                                       self.minute, self.second)

YMDhms.__repr__ = lambda self: "{0}({1},{2},{3},{4},{5},{6})".format(self.__class__.__name__,
                                                                    self.year, self.month, self.day, self.hour,
                                                                    self.minute, self.second)

YWdhms.__str__ = lambda self: "YWdhms({0},{1},{2},{3},{4},{5})".format(self.iso_year, self.iso_week, self.week_day, self.hour,
                                                                       self.minute, self.second)

YWdhms.__repr__ = lambda self: "{0}({1},{2},{3},{4},{5},{6})".format(self.__class__.__name__,
                                                                     self.iso_year, self.iso_week, self.week_day, self.hour,
                                                                     self.minute, self.second)

# Fix up GeoPoint
GeoPoint.__str__ = lambda self: "GeoPoint({0},{1},{2})".format(self.x, self.y, self.z)
GeoPoint_difference = lambda a, b: GeoPoint.difference(a, b)
GeoPoint_xy_distance = lambda a, b: GeoPoint.xy_distance(a, b)

# Fix up LandTypeFractions

LandTypeFractions.__str__ = lambda \
        self: "LandTypeFractions(glacier={0},lake={1},reservoir={2},forest={3},unspecified={4})".format(self.glacier(),
                                                                                                        self.lake(),
                                                                                                        self.reservoir(),
                                                                                                        self.forest(),
                                                                                                        self.unspecified())


# Fix up GeoCellData
def StrGeoCellData(gcd):
    return "GeoCellData(mid_point={0},catchment_id={1},area={2},ltf={3})".format(str(gcd.mid_point()),
                                                                                 gcd.catchment_id(), gcd.area(),
                                                                                 str(gcd.land_type_fractions_info()))


GeoCellData.__str__ = lambda self: StrGeoCellData(self)

# Fix up UtcPeriod
UtcPeriod.to_string = lambda self: str(self)

# Fix up TimeAxis

def ta_iter(x):
    x.counter = 0
    return x


def ta_next(ta):
    if ta.counter >= len(ta):
        del ta.counter
        raise StopIteration
    ta.counter += 1
    return ta(ta.counter - 1)

TimeAxisFixedDeltaT.__str__ = lambda self: "TimeAxisFixedDeltaT({0},{1},{2})".format(Calendar().to_string(self.start), self.delta_t, self.n)
TimeAxisFixedDeltaT.__len__ = lambda self: self.size()
TimeAxisFixedDeltaT.__call__ = lambda self, i: self.period(i)
TimeAxisFixedDeltaT.__iter__ = lambda self: ta_iter(self)
TimeAxisFixedDeltaT.__next__ = lambda self: ta_next(self)

TimeAxisCalendarDeltaT.__str__ = lambda self: "TimeAxisCalendarDeltaT(Calendar('{3}'),{0},{1},{2})".format(Calendar().to_string(self.start), self.delta_t, self.n,self.calendar.tz_info.name())
TimeAxisCalendarDeltaT.__len__ = lambda self: self.size()
TimeAxisCalendarDeltaT.__call__ = lambda self, i: self.period(i)
TimeAxisCalendarDeltaT.__iter__ = lambda self: ta_iter(self)
TimeAxisCalendarDeltaT.__next__ = lambda self: ta_next(self)

TimeAxisByPoints.__str__ = lambda self: "TimeAxisByPoints(total_period={0}, n={1},points={2} )".format(str(self.total_period()),len(self),repr(TimeAxis(self).time_points))
TimeAxisByPoints.__len__ = lambda self: self.size()
TimeAxisByPoints.__call__ = lambda self, i: self.period(i)
TimeAxisByPoints.__iter__ = lambda self: ta_iter(self)
TimeAxisByPoints.__next__ = lambda self: ta_next(self)

def nice_ta_string(time_axis):
    if time_axis.timeaxis_type == TimeAxisType.FIXED:
        return '{0}'.format(str(time_axis.fixed_dt))
    if time_axis.timeaxis_type == TimeAxisType.CALENDAR:
        return '{0}'.format(str(time_axis.calendar_dt))
    return '{0}'.format(str(time_axis.point_dt))


TimeAxis.__str__ = lambda self: nice_ta_string(self)
TimeAxis.__len__ = lambda self: self.size()
TimeAxis.__call__ = lambda self, i: self.period(i)
TimeAxis.__iter__ = lambda self: ta_iter(self)
TimeAxis.__next__ = lambda self: ta_next(self)


TimeAxis.time_points = property( lambda self: time_axis_extract_time_points(self).to_numpy(),doc= \
"""
 extract all time-points from a TimeAxis
 like
 [ time_axis.time(i) ].append(time_axis.total_period().end) if time_axis.size() else []

Parameters
----------
time_axis : TimeAxis

Returns
-------
time_points:numpy.array(dtype=np.int64)
   [ time_axis.time(i) ].append(time_axis.total_period().end)
""")

# fix up property on timeseries
TimeSeries.time_axis = property(lambda self: self.get_time_axis(), doc="returns the time_axis of the timeseries")
TimeSeries.__len__ = lambda self: self.size()
TimeSeries.v = property(lambda self: self.values, doc="returns the point-values of timeseries, alias for .values")

TimeSeries.kling_gupta = lambda self, other_ts, s_r=1.0, s_a=1.0, s_b=1.0: kling_gupta(self, other_ts,
                                                                                       self.get_time_axis(), s_r, s_a,
                                                                                       s_b)
TimeSeries.kling_gupta.__doc__ = \
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

TimeSeries.nash_sutcliffe = lambda self, other_ts: nash_sutcliffe(self, other_ts, self.get_time_axis())
TimeSeries.nash_sutcliffe.__doc__ = \
    """
    Computes the Nash-Sutcliffe model effiency coefficient (n.s)
    for the two time-series over the specified time_axis
    Ref:  http://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
    Parameters
    ----------
    observed_ts : TimeSeries
     the observed time-series
    model_ts : TimeSeries
     the time-series that is the model simulated / calculated ts
    time_axis : TimeAxis
     the time-axis that is used for the computation
    Return
    ------
     float: The n.s performance, that have a maximum at 1.0
    """

TsFixed.values = property(lambda self: self.v, doc="returns the point values, .v of the timeseries")
TsFixed.time_axis = property(lambda self: self.get_time_axis(), doc="returns the time_axis of the timeseries")
TsPoint.values = property(lambda self: self.v, doc="returns the point values, .v of the timeseries")
TsPoint.time_axis = property(lambda self: self.get_time_axis(), doc="returns the time_axis of the timeseries")

# some minor fixup to ease work with core-time-series vs TimeSeries
TsFixed.TimeSeries = property(lambda self: TimeSeries(self),doc="return a fully featured TimeSeries from the core TsFixed ")
TsFixed.nash_sutcliffe = lambda self, other_ts: nash_sutcliffe(self.TimeSeries, other_ts, TimeAxis(self.get_time_axis()))
TsFixed.kling_gupta = lambda self, other_ts, s_r=1.0, s_a=1.0, s_b=1.0: kling_gupta(self.TimeSeries, other_ts,
                                                                                       TimeAxis(self.get_time_axis()), s_r, s_a,
                                                                                       s_b)

TsPoint.TimeSeries = property(lambda self: TimeSeries(self.get_time_axis(),self.v,self.point_interpretation()),doc="return a fully featured TimeSeries from the core TsPoint")
TsPoint.nash_sutcliffe = lambda self, other_ts: nash_sutcliffe(self.TimeSeries, other_ts, TimeAxis(self.get_time_axis()))
TsPoint.kling_gupta = lambda self, other_ts, s_r=1.0, s_a=1.0, s_b=1.0: kling_gupta(self.TimeSeries, other_ts,
                                                                                       TimeAxis(self.get_time_axis()), s_r, s_a,
                                                                                       s_b)



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
KalmanState.x = property(lambda self: KalmanState.get_x(self).to_numpy(),
                         doc="represents the current bias estimate, kalman.state.x")
KalmanState.k = property(lambda self: KalmanState.get_k(self).to_numpy(),
                         doc="represents the current kalman gain factors, kalman.state.k")
KalmanState.P = property(lambda self: np_array(KalmanState.get_P(self)),
                         doc="returns numpy array of kalman.state.P, the nxn covariance matrix")
KalmanState.W = property(lambda self: np_array(KalmanState.get_W(self)),
                         doc="returns numpy array of kalman.state.W, the nxn noise matrix")


# fixup KalmanBiasPredictor
def KalmanBiasPredictor_update_with_forecast(bp, fc_set, obs, time_axis):
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
        KalmanBiasPredictor.update_with_geo_forecast(bp, fc_set, obs, time_axis)
    else:
        KalmanBiasPredictor.update_with_forecast_vector(bp, fc_set, obs, time_axis)


def KalmanBiasPredictor_compute_running_bias(bp, fc_ts, obs_ts, time_axis):
    """
    compute the running bias timeseries,
    using one 'merged' - forecasts and one observation time - series.

    Before each day - period, the bias - values are copied out to form
    a continuous bias prediction time-series.

    Parameters
    ----------

    bias_predictor : KalmanBiasPredictor
        The bias predictor object it self

    forecast_ts : Timeseries
        a merged forecast ts
        with period covering the observation_ts and time_axis supplied

    observation ts: Timeseries
        the observation time-series

    time_axis : Timeaxis
        covering the period/timesteps to be updated
        e.g. yesterday, 3h resolution steps, according to the points in the filter

    Returns
    -------
    bias_ts : Timeseries(time_axis,bias_vector,POINT_AVERAGE)
        computed running bias-ts
    """
    return KalmanBiasPredictor.compute_running_bias_ts(bp, fc_ts, obs_ts, time_axis)


KalmanBiasPredictor.update_with_forecast = KalmanBiasPredictor_update_with_forecast
KalmanBiasPredictor.compute_running_bias = KalmanBiasPredictor_compute_running_bias



class Timeseries(TimeSeries):
    @deprecated("please use the TimeSeries class instead")
    def __init__(self, *args, **kwargs):
        super(Timeseries, self).__init__(*args, **kwargs)

class Timeaxis2(TimeAxis):
    @deprecated("please use the TimeAxis class instead")
    def __init__(self, *args, **kwargs):
        super(Timeaxis2, self).__init__(*args, **kwargs)

class Timeaxis(TimeAxisFixedDeltaT):
    @deprecated("please start using TimeAxisFixedDeltaT")
    def __init__(self, *args, **kwargs):
        super(Timeaxis, self).__init__(*args, **kwargs)

class CalendarTimeaxis(TimeAxisCalendarDeltaT):
    @deprecated("please start using TimeAxisCalendarDeltaT")
    def __init__(self, *args, **kwargs):
        super(CalendarTimeaxis, self).__init__(*args, **kwargs)

class PointTimeaxis(TimeAxisByPoints):
    @deprecated("please start using TimeAxisByPoints")
    def __init__(self, *args, **kwargs):
        super(PointTimeaxis, self).__init__(*args, **kwargs)

TimeaxisType = TimeAxisType  #  todo: deprecate it

def ts_vector_values_at_time(tsv:TsVector, t:int):
    if not isinstance(tsv, TsVector):
        if not isinstance(tsv, list):
            raise RuntimeError('Supplied list of timeseries must be of type TsVector or list(TimeSeries)')
        list_of_ts = tsv
        tsv = TsVector()
        for ts in list_of_ts:
            tsv.append(ts)
    return tsv.values_at(t).to_numpy()

ts_vector_values_at_time.__doc__ = TsVector.values_at.__doc__.replace('DoubleVector','ndarray').replace('TsVector','TsVector or list(TimeSeries)')

TsVector.values_at_time = ts_vector_values_at_time
TsVector.values_at_time.__doc__ = TsVector.values_at.__doc__.replace('DoubleVector','ndarray')



def geo_point_source_vector_values_at_time(gtsv:GeoPointSourceVector, t:int):
    #if not isinstance(gtsv, GeoPointSourceVector):
    #    raise RuntimeError('Supplied list of timeseries must be of GeoPointSourceVector')
    return compute_geo_ts_values_at_time(gtsv, t).to_numpy()

GeoPointSourceVector.values_at_time = geo_point_source_vector_values_at_time
GeoPointSourceVector.values_at_time.__doc__ = compute_geo_ts_values_at_time.__doc__.replace('DoubleVector','ndarray')
RadiationSourceVector.values_at_time = GeoPointSourceVector.values_at_time
PrecipitationSourceVector.values_at_time = GeoPointSourceVector.values_at_time
TemperatureSourceVector.values_at_time = GeoPointSourceVector.values_at_time
RelHumSourceVector.values_at_time = GeoPointSourceVector.values_at_time
WindSpeedSourceVector.values_at_time = GeoPointSourceVector.values_at_time
