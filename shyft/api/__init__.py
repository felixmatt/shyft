from ._api import *

# Fix up vector types

DoubleVector.size= lambda self: len(self)

UtcTimeVector.size= lambda self: len(self)

IntVector.size = lambda self: len(self)

StringVector.size = lambda self: len(self)

TsVector.size = lambda self: len(self)

TsVector.push_back = lambda self, ts: self.append(ts)

TargetSpecificationVector.size =lambda self: len(self)

#fix bw. stl name
UtcTimeVector.push_back=lambda self, x: self.append(x)
IntVector.push_back=lambda self, x: self.append(x)
DoubleVector.push_back=lambda self, x: self.append(x)
StringVector.push_back=lambda self, x: self.append(x)

# FIx up YMDhms
YMDhms.__str__ = lambda self: "YMDhms({0},{1},{2},{3},{4},{5})".format(self.year,self.month,self.day,self.hour,self.minute,self.second)

# Fix up GeoPoint
GeoPoint.__str__ = lambda self: "GeoPoint({0},{1},{2})".format(self.x,self.y,self.z)
GeoPoint_difference = lambda a, b: GeoPoint.difference(a,b)
GeoPoint_xy_distance = lambda a, b: GeoPoint.xy_distance(a,b)

# Fix up UtcPeriod
UtcPeriod.to_string = lambda self: str(self)

# Fix up Timeaxis
Timeaxis.__str__= lambda self:  "Timeaxis({0},{1},{2})".format(Calendar().to_string(self.start),self.delta_t,self.n)
Timeaxis.__len__= lambda self: self.size()
Timeaxis.__call__= lambda self, i: self.period(i)

def ta_iter(x):
    x.counter = 0
    return x

def ta_next(ta):
    if ta.counter >= len(ta):
        del ta.counter
        raise StopIteration
    ta.counter += 1
    return ta(ta.counter - 1)

Timeaxis.__iter__= lambda self: ta_iter(self)
Timeaxis.__next__= lambda self: ta_next(self)

# Fix up PointTimeaxis
PointTimeaxis.__str__= lambda self:  "PointTimeaxis(total_period={0}, n={1} )".format(str(self.total_period()), len(self))
PointTimeaxis.__len__= lambda self: self.size()
PointTimeaxis.__call__= lambda self, i: self.period(i)
PointTimeaxis.__iter__= lambda self: ta_iter(self)
PointTimeaxis.__next__= lambda self: ta_next(self)


# Fix up ARegionEnvironment
TemperatureSource.vector_t = TemperatureSourceVector
PrecipitationSource.vector_t = PrecipitationSourceVector
RadiationSource.vector_t = RadiationSourceVector
RelHumSource.vector_t = RelHumSourceVector
WindSpeedSource.vector_t = WindSpeedSourceVector