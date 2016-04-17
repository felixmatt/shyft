from ._api import *

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
Timeseries.time_shift = lambda self,delta_t: time_shift(self,delta_t)


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
