#!/usr/bin/env python

"""

xpm_plot.py

Python script to plot XPM matrices produced by GROMACS analysis tools.

Requires:
    * python2.7+
    * matplotlib
"""

from __future__ import print_function, division

__author__ = 'Joao Rodrigues'
__email__ = 'j.p.g.l.m.rodrigues@gmail.com'

import os
import re
import shlex
import sys

try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
except ImportError as e:
    print('[!] The required Python libraries could not be imported:', file=sys.stderr)
    print('\t{0}'.format(e))
    sys.exit(1)

##      

def parse_xpm(fname):
    """Parses XPM file colors, legends, and data"""
    
    metadata = {}
    num_data = [[], [], []] # x, y, color
    color_data = {} # color code: color hex, color value

    ff_path = os.path.abspath(fname)
    if not os.path.isfile(ff_path):
        raise IOError('File not readable: {0}'.format(ff_path))
    
    with open(ff_path, 'r') as fhandle:
        for line in fhandle:
            line = line.strip().rstrip(',')
            
            if line.startswith('/*'):
                tokens = shlex.split(line[2:].lstrip())
                t_name = tokens[0]
                
                if t_name in set(('title:', 'legend:', 'x-label:', 'y-label:')):
                    metadata[t_name.strip(':')] = tokens[1]
                
                elif t_name == 'x-axis:':
                    x_values = map(float, tokens[1:-1]) # last */
                    num_data[0].extend(x_values)

                elif t_name == 'y-axis:':
                    y_values = map(float, tokens[1:-1]) # last */
                    num_data[1].extend(y_values)

            elif line.startswith('"'):
                if line.endswith('*/'):
                    # Color
                    tokens = shlex.split(line)
                    c_code, _, _ = tokens[0].split()
                    c_value = float(tokens[2])
                    color_data[c_code] = c_value
                    
                elif line.endswith('"') and ' ' not in line:
                    num_data[2].append(line[1:-1])
    
    # Convert data to actual values
    for irow, row in enumerate(num_data[2][:]):
        num_data[2][irow] = map(color_data.get, row)

    return metadata, num_data

def plot_data(data, metadata, interactive=True, outfile=None, 
              colormap='Spectral', bg_color='white'):
    """
    Plotting function.
    """
    
    f = plt.figure()
    # ax = f.gca(projection='3d')
    ax = f.gca()
    
    _x, _y, _z = map(np.asarray, data)
    _x, _y = np.meshgrid(_x, _y)
    # ax.plot_surface(_x, _y, _z, cmap=_cmap, linewidth=0, antialiased=False)
    plt.imshow(_z, cmap=colormap, origin='upper')

    # Set xtick labels to timestamps
    xtick_list = map(int, ax.get_xticks().tolist())
    ori_xticks = [xt for xt in xtick_list if xt>=0 and xt<=len(data[0])]
    for itick, xtick in enumerate(xtick_list[:]):
        if xtick >= 0 and xtick <= len(data[0]):
            xtick_list[itick] = data[0][xtick]
    ax.set_xticklabels(xtick_list)
    
    # Hack to get the y-axis upside down in the right place
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    n_x_ticks = len(ori_xticks)
    x_spacing = ori_xticks[1] - ori_xticks[0]
    y_tick_list = [ymin - x_spacing*t for t in range(n_x_ticks)]
    y_tick_list = map(int, y_tick_list)

    y_labels = [data[1][ytick] for ytick in y_tick_list][::-1]
    offset = max(y_labels) - max(xtick_list)
    y_labels = [ytick - offset for ytick in y_labels]
    
    ax.set_yticks(y_tick_list)
    ax.set_yticklabels(y_labels)

    # Formatting Labels & Appearance
    ax.set_xlabel(metadata.get('x-label', ''))
    ax.set_ylabel(metadata.get('y-label', ''))
    ax.set_title(metadata.get('title', ''))
    ax.set_facecolor(bg_color)
    ax.grid('on')

    cbar = plt.colorbar()
    cbar.set_label(metadata.get('legend', ''))


    if outfile:
        plt.savefig(outfile)
        
    if interactive:
        plt.show()
    
    return
##

if __name__ == '__main__':

    import argparse
    from argparse import RawDescriptionHelpFormatter
    
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    ap.add_argument('xpm_f', type=str, help='XPM input file', metavar='XPM input file')

    io_group = ap.add_mutually_exclusive_group(required=True)
    io_group.add_argument('-o', '--output', type=str, help='PDF output file')
    io_group.add_argument('-i', '--interactive', action='store_true', 
                    help='Launches an interactive matplotlib session')
    
    ot_group = ap.add_argument_group('Other Options')
    ot_group.add_argument('-c', '--colormap', default='Spectral',
                          help='Range of colors used in the plot. For a list of all\
                                available colormaps refer to \
                                matplotlib.org/examples/color/colormaps_reference.html')

    ot_group.add_argument('-b', '--background-color', default='lightgray',
                          help='Background color used in the plot. For a list of all available \
                                colors refer to \
                                matplotlib.org/examples/color/named_colors.html')
    cmd = ap.parse_args()
    
    metadata, data = parse_xpm(cmd.xpm_f)
    n_x, n_y, _ = map(len, data)
    n_z = sum(map(len, data[2]))
    print('[+] Read {0}x{1} matrix ({2} elements)'.format(n_x, n_y, n_z))

    plot_data(data, metadata,
              interactive=cmd.interactive, outfile=cmd.output,
              colormap=cmd.colormap, bg_color=cmd.background_color)
    
