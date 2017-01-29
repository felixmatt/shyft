from shyft import api

# TODO: Implement me!

def create_time_series(time_axis, y0=0.0, y_min=-1.0, y_max=1.0, amplitude=1.0, t0=0, period=3600*12):
    return api.TsSinFx(time_axis,api.SinFx(y0, y_min, y_max, amplitude, t0, period))

def create_mock_time_series_data(type_, time_axis, **kwargs):

    default_kwargs = {"temperature": {"y0": 0.0,
                                    "y_min": -1.0,
                                    "y_max": 5,
                                    "amplitude": 6,
                                    "t0": 0,
                                    "period": 3600*24},
                    "precipitation": {"y0": 0.0,
                                      "y_min": 0.0,
                                      "y_max": 5,
                                      "amplitude": 10,
                                      "t0": 0,
                                      "period": 3600*30},
                    "relative_humidity":  {"y0": 0.6,
                                           "y_min": 0.5,
                                           "y_max": 0.8,
                                           "amplitude": 0.2,
                                           "t0": 0,
                                           "period": 3600*30},
                    "wind_speed": {"y0": 0.0,
                                   "y_min": 1.0,
                                   "y_max": 3.0,
                                   "amplitude": 10.0,
                                   "t0": 0,
                                   "period": 3600*2},
                    "radiation": {"y0": 0.0,
                                  "y_min": 0.0,
                                  "y_max": 500,
                                  "amplitude": 700,
                                  "t0": 0,
                                  "period": 3600*24}}
    default_kwargs[type_].update(kwargs.get(type_, {}))
    return create_time_series(time_axis, **default_kwargs[type_])


def create_mock_station_data(t0, dt, n_steps, **kwargs):
    time_axis = api.TimeAxisFixedDeltaT(t0, dt, n_steps)
    return {"temperature": create_mock_time_series_data("temperature", time_axis, **kwargs),
            "precipitation": create_mock_time_series_data("precipitation", time_axis, **kwargs),
            "relative_humidity": create_mock_time_series_data("relative_humidity", time_axis, **kwargs),
            "wind_speed": create_mock_time_series_data("wind_speed", time_axis, **kwargs),
            "radiation": create_mock_time_series_data("radiation", time_axis, **kwargs)}
