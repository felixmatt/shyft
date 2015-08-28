class InputSource(object):
    """
    Input source, typically a weather measuring station.

    It can contain none, some, or all the relevant time series.
    """

    def __init__(self, geo_location, sources, tags=None, period=None):
        self.geo_location = geo_location
        self.sources = sources
        self.tags = tags if tags is not None else []

    @property
    def temperature(self):
        return self.sources.get("temperature", None)

    @property
    def precipitation(self):
        return self.sources.get("precipitation", None)

    @property
    def relative_humidity(self):
        return self.sources.get("relative_humidity", None)

    @property
    def wind_speed(self):
        return self.sources.get("wind_speed", None)

    @property
    def radiation(self):
        return self.sources.get("radiation", None)
