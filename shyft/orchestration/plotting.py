import numpy as np
import matplotlib.pylab as plt
from matplotlib import ticker
from shyft.api import deltahours
from matplotlib.patches import Rectangle


def blend_colors(color1, color2):
    if len(color1) != 4 or len(color2) != 4:
        raise ValueError("Both colors must be of length 4")
    r_alpha = 1 - (1 - color1[-1])*(1 - color2[-1])
    # r_alpha = color1[-1] + color2[-1]*(1 - color1[-1])  # Alternative blending strategy for alpha layer
    return list(np.asarray(color1[:-1])*color1[-1]/r_alpha +
                np.asarray(color2[:-1])*color2[-1]*(1 - color1[-1])/r_alpha) + [r_alpha]


def plot_np_percentiles(time, percentiles, base_color=1.0, alpha=0.5, plw=0.5, linewidth=1, mean_color=0.0, label=None):
    if base_color is not None:
        if not isinstance(base_color, np.ndarray):
            if isinstance(base_color, (int, float)):
                base_color = 3*[base_color]
            base_color = np.array(base_color)
    if not isinstance(mean_color, np.ndarray):
        if isinstance(mean_color, (int, float)):
            mean_color = 3*[mean_color]
        mean_color = np.array(mean_color)
    percentiles = list(percentiles)
    num_intervals = len(percentiles)//2
    f_handles = []
    proxy_handles = []
    prev_facecolor = None
    for i in range(num_intervals):
        facecolor = list(base_color) + [alpha]
        f_handles.append(plt.fill_between(time, percentiles[i], percentiles[-(i+1)],
                         edgecolor=(0, 0, 0, 0), facecolor=facecolor))
        proxy_handles.append(Rectangle((0, 0), 1, 1, fc=blend_colors(prev_facecolor, facecolor) if
                             prev_facecolor is not None else facecolor))
        prev_facecolor = facecolor
    linewidths = len(percentiles)*[plw]
    linecols = len(percentiles)*[(0.7, 0.7, 0.7, 1.0)]
    labels = len(percentiles)*[None]
    if len(percentiles) % 2:
        mid = len(percentiles)//2
        linewidths[mid] = linewidth
        linecols[mid] = mean_color
        labels[mid] = label
    handles = []
    for p, lw, lc, label in zip(percentiles, linewidths, linecols, labels):
        h, = plt.plot(time, p, linewidth=lw, color=lc, label=label)
        handles.append(h)
    if len(percentiles) % 2:
        mean_h = handles.pop(len(handles)//2)
        handles = [mean_h] + handles
    return (handles + f_handles), proxy_handles


def set_calendar_formatter(cal, str_format="{year:04d}.{month:02d}.{day:02d}", format_major=True):
    fields = {"year": None,
              "month": None,
              "day": None,
              "hour": None,
              "minute": None,
              "second": None}
    ax = plt.gca()
    fig = plt.gcf()

    def format_date(x, pos=None):
        t_utc = cal.trim(int(round(greg_to_utc(x))), deltahours(1))
        ymd = cal.calendar_units(t_utc)
        for f in fields:
            fields[f] = getattr(ymd, f)
        return str_format.format(**fields)
    if format_major:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        plt.setp(ax.get_xminorticklabels(), rotation=45, horizontalalignment='right')
    else:
        ax.xaxis.set_minor_formatter(ticker.FuncFormatter(format_date))
        plt.setp(ax.get_xmajorticklabels(), rotation=45, horizontalalignment='right')
    fig.autofmt_xdate()


def greg_to_utc(t):
    a = 3600*24.0
    b = 719164.0
    return (np.asarray(t) - b)*a


def utc_to_greg(t):
    a = 3600*24.0
    b = 719164.0
    return np.asarray(t)/a + b
