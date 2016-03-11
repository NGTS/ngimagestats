#!/usr/local/python/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import subprocess as sp
import fitsio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse

try:
    plt.style.use('bmh')
except (IOError, ValueError):
    # Styles are not available
    pass

def remove_overscan(image):
    overscan = image[4:, -15:].mean()
    return image[:, 20:-20] - overscan

def compute_histogram(image, *args, **kwargs):
    flat_image = image.ravel()
    ind = flat_image > 0
    log_pixel_counts = np.log10(flat_image[ind])

    values, bx = np.histogram(log_pixel_counts, *args, **kwargs)

    x = np.array(list(zip(bx[:-1], bx[1:]))).ravel()
    y = np.array(list(zip(values, values))).ravel()
    return 10 ** x, np.maximum(y, 0.1)

def pick_central_region(image, margin=8):
    x, y = image.shape
    return image[margin:y-margin, margin:x-margin]

def render_profile(filename, axis, meta, label=''):
    with fitsio.FITS(filename) as infile:
        header = infile[0].read_header()
        image = infile[0].read()

    image_electrons = image * meta.get('gain', 1.)

    x, y = compute_histogram(image_electrons, bins=200)
    l, = axis.loglog(x, y, label=label, lw=1)
    return l, x, y



def extract_meta(filename):
    header = fitsio.read_header(filename)
    return {
        'image_id': header.get('IMAGE_ID', None),
        'gain': header.get('GAINFACT', None),
        'gain_setting': header.get('GAIN', None),
        'camera_id': header.get('CAMERAID', None),
        'vi+': header.get('VI_PLUS', None),
    }

def main(args):

    locator = plt.LogLocator(subs=[1, 2, 5])
    formatter = plt.LogFormatter(labelOnlyBase=False)



    fig, axis = plt.subplots()
    for filename in args.filename:
        meta = extract_meta(filename)
        if args.override_meta:
            for entry in args.override_meta:
                key, value = entry.split('=')
                if key not in meta:
                    raise ValueError("Unsupported key: %s. Available: %s" % (
                        key, meta.keys()))

                value_type = type(meta[key])
                meta[key] = value_type(value)

        l, _, _ = render_profile(filename, axis, meta=meta)

        axis.axvline((2 ** 16 - 1) * meta.get('gain', 1.), color=l.get_color(), alpha=0.5,
                    ls='--')

    # axis.axvline(bin_max, ls='--', color='k')
    # title = 'Camera {camera_id}; VI+{vi+}'.format(**meta)
    # axis.axvline(2 ** 16 - 1)
    axis.set(xlim=(100 1E6), xlabel='Pixel value [electrons]', ylim=(1E-1, 1E7), ylabel='Number')
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output)
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='+')
    parser.add_argument('-m', '--override-meta', required=False, nargs='*')
    parser.add_argument('-o', '--output', required=False)
    main(parser.parse_args())

