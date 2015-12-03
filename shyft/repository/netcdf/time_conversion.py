from shyft import api
from netcdftime import utime
import numpy as np

""" These are the current supported regular time-step intervals """
delta_t_dic = {'days': api.deltahours(24), 'hours': api.deltahours(1), 'minutes': api.deltaminutes(1),
               'seconds': api.Calendar.SECOND}


def convert_netcdf_time(time_spec, t):
    """
    Converts supplied numpy array to  shyft utctime given netcdf time_spec.
    Throws exception if time-unit is not supported, i.e. not part of delta_t_dic
    as specified in this file.

    Parameters
    ----------
        time_spec: string
           from netcdef  like 'hours since 1970-01-01 00:00:00'
        t: numpy array
    Returns
    -------
        numpy array type int64 with new shyft utctime units (seconds since 1970utc)
    """
    u = utime(time_spec)
    t_origin = api.Calendar(int(u.tzoffset)).time(
        api.YMDhms(u.origin.year, u.origin.month, u.origin.day, u.origin.hour, u.origin.minute, u.origin.second))

    delta_t = delta_t_dic[u.units]
    return (t_origin + delta_t * t[:]).astype(np.int64)
