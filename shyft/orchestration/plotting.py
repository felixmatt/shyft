import numpy as np
import matplotlib.pylab as plt
from matplotlib import ticker
from shyft.api import Timeaxis
from shyft.api import deltahours
from matplotlib.patches import Rectangle


def blend_colors(color1, color2):
    if len(color1) != 4 or len(color2) != 4:
        raise ValueError("Both colors must be of length 4")
    r_alpha = 1 - (1 - color1[-1])*(1 - color2[-1])
    #r_alpha = color1[-1] + color2[-1]*(1 - color1[-1])
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
    fill_legend = []
    prev_facecolor = None
    for i in range(num_intervals):
        facecolor = list((1.0 - float(i + 1)/(num_intervals + 1))*base_color) + [alpha] if \
            base_color is not None else 4*[1.]
        f_handles.append(plt.fill_between(time, percentiles[i], percentiles[-(i+1)], 
                         edgecolor=(0, 0, 0, 0), facecolor=facecolor))
        fill_legend.append(blend_colors(prev_facecolor, facecolor) if prev_facecolor is not None else facecolor)
        prev_facecolor = facecolor
    print(fill_legend)
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
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    y_min += (y_max + y_min)/2

    dy = (y_max - y_min)/(2*num_intervals)
    dx = (x_max - x_min)/num_intervals
    for i, fill in enumerate(fill_legend):
        x = time[0]
        y = y_min + i*2*dy
        rectangle = Rectangle((x, y), dx, dy, facecolor=fill)
        ax.add_patch(rectangle)
        plt.text(x + 1.5*dx, y - dy, "Foo i")
    return (handles + f_handles), fill_legend


def set_display_time_axis(ta, cal, n_xticks=10, format="W-WY"):
    if isinstance(ta, Timeaxis):
        dt = ta.delta()
        start = ta.start()
        t = [start + i*dt for i in range(ta.size())]
    else:
        t = ta[:]
    ticks = ([t[int(round(_))] for _ in np.linspace(0, len(t) - 1, n_xticks)])
    ticks[-1] = t[-1]
    def convert(t_utc):
        ymd = cal.calendar_units(t_utc)
        return "{:04d}.{:02d}.{:02d}:{:02d}".format(ymd.year, ymd.month, ymd.day, ymd.hour)
    handles = plt.xticks(ticks, [convert(_t) for _t in ticks], rotation="vertical")
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("Time in {} coordinates.".format("UTC"))
    plt.xlim(ticks[0], ticks[-1])
    return handles


def set_calendar_formatter(cal):
    ax = plt.gca()
    fig = plt.gcf()
    def format_date(x, pos=None):
        t_utc = cal.trim(int(round(x)), deltahours(1))
        ymd = cal.calendar_units(t_utc)
        return "{:04d}.{:02d}.{:02d}:{:02d}".format(ymd.year, ymd.month, ymd.day, ymd.hour)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    fig.autofmt_xdate()
    return ax
