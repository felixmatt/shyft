from shyft import api

from netcdftime import utime
import numpy as np


def convert_netcdf_time(time_spec, t):
    """
    :param time_spec: from netcdef  like 'hours since 1970-01-01 00:00:00'
    :param t: numpy array
    :return: numpy array with new units
    """
    u = utime(time_spec)
    t_origin = api.Calendar(int(u.tzoffset)).time(api.YMDhms(u.origin.year,u.origin.month,u.origin.day,u.origin.hour,u.origin.minute,u.origin.second))
    delta_t_dic = {'days':api.deltahours(24),'hours':api.deltahours(1),'minutes':api.deltaminutes(1),'seconds':api.Calendar.SECOND}
    delta_t = delta_t_dic[u.units]
    return (t_origin + delta_t*t[:]).astype(np.int64)