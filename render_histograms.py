#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess as sp
from extract_metadata import Session, Metadata
import fitsio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sqlalchemy.sql import func
from sqlalchemy import desc
import sqlite3
import os

try:
    plt.style.use('bmh')
except (IOError, ValueError):
    # Styles are not available
    pass

bin_min = 1E3
bin_max = 5E5

def remove_overscan(image):
    # return image[:, 20:-20]

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

def render_profile(filename, axis, label, gain_value, colour):
    with fitsio.FITS(filename) as infile:
        header = infile[0].read_header()
        image = infile[0].read()


    if image.shape[1] == 2088:
        image = remove_overscan(image)
    assert image.shape == (2048, 2048)
    image = pick_central_region(image)

    image_electrons = image * gain_value
    assert image_electrons.shape[0] < 2048 and image_electrons.shape[1] < 2048

    # camera_id = header.get('CAMERAID', 'UNKNOWN')
    # gain = header.get('GAIN', 'UNKNOWN')
    # exptime = header.get('EXPTIME', None) or header.get('EXPOSURE', 'UNKNOWN')

    x, y = compute_histogram(image_electrons, bins=200,
                            range=(np.log10(bin_min), np.log10(bin_max)))

    # title = 'Camera {camera_id}; gain {gain}; exptime {exptime}'.format(
    #     camera_id=camera_id, gain=gain, exptime=exptime)

    # locator = plt.LogLocator(subs=[1, 2, 5])
    # formatter = plt.LogFormatter(labelOnlyBase=False)

    # fig, axis = plt.subplots()
    l, = axis.loglog(x, y, label=label, lw=1, color=colour)
    # axis.axvline(2 ** 16 - 1)
    # axis.set(xlim=(100, 70000), xlabel='Pixel value', ylim=(1E-1, 1E7), ylabel='Number', title=title)
    # axis.xaxis.set_major_locator(locator)
    # axis.xaxis.set_major_formatter(formatter)
    # fig.tight_layout()
    # fig.savefig(args.output, bbox_inches='tight')
    return l, x, y

def fetch_camera_gain_mapping(filename):
    with sqlite3.connect(filename) as con:
        cur = con.cursor()
        cur.execute('select camera_id, gain_setting, gain_value from camera_gains')
        rows = cur.fetchall()

    out = {}
    for (camera_id, gain_setting, gain_value) in rows:
        out[(camera_id, gain_setting)] = gain_value
    return out


if __name__ == '__main__':
    session = Session()

    # meta = session.query(Metadata, func.count(Metadata.camera_id)).filter(
    #     (Metadata.exposure_time == 30) & (Metadata.vi == 227) & (Metadata.cycle == 1)
    # ).group_by(Metadata.camera_id)

    output_dir = 'histograms'

    locator = plt.LogLocator(subs=[1, 2, 5])
    formatter = plt.LogFormatter(labelOnlyBase=False)

    camera_gain_mapping = fetch_camera_gain_mapping('metadata.db')


    camera_ids = session.query(Metadata.camera_id).filter(
        ~Metadata.camera_id.in_({802})
    ).distinct().all()
    vis = session.query(Metadata.vi).distinct().all()

    colours = {4: u'#348ABD', 2: u'#A60628', 1: u'#7A68A6'}
    for camera_id, in camera_ids:
        for vi, in vis:
            metas = session.query(Metadata).filter(
                (Metadata.camera_id == camera_id) & (Metadata.exposure_time == 30) & (Metadata.vi == vi) & (Metadata.cycle == 1) & (Metadata.gain_setting != 4)
            ).order_by(desc(Metadata.gain_setting)).all()


            output_filename = os.path.join(output_dir, 'histogram_{camera_id}_{vi}.png'.format(
                camera_id=camera_id, vi=vi))
            print(output_filename)

            fig, axis = plt.subplots()
            lines = []
            for meta in metas:
                gain_value = camera_gain_mapping[(camera_id, meta.gain_setting)]
                l, x, y = render_profile(meta.filename, axis, meta.gain_setting, gain_value,
                                  colour=colours[meta.gain_setting])
                lines.append((x, y))
                axis.axvline((2 ** 16 - 1) * gain_value, color=l.get_color(), alpha=0.5,
                            ls='--')


            # axis.axvline(bin_max, ls='--', color='k')
            title = 'Camera {camera_id}; VI+{vi}'.format(
                camera_id=camera_id, vi=vi)
            # axis.axvline(2 ** 16 - 1)
            axis.legend(loc='best')
            axis.set(xlim=(1000, 1E6), xlabel='Pixel value [electrons]', ylim=(1E-1, 1E7), ylabel='Number', title=title)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(formatter)
            fig.tight_layout()

            fig.savefig(output_filename)
            plt.close(fig)













    # output_filename = '{output_dir}/hist_{camera_id}_{vi}_{gain}_{cycle}.png'.format(
    #     output_dir=output_dir, camera_id=entry.camera_id, vi=entry.vi, gain=entry.gain_setting,
    #     cycle=entry.cycle)
    # cmd = ['python', '../attempt-1/render_image_histogram.py', entry.filename, '-o', output_filename]
    # print(cmd)
    # sp.check_call(cmd)

