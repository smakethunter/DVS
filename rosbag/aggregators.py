import csv
import numpy as np
import pandas as pd


import os
import shutil

import cv2
import matplotlib.pyplot as plt

import os.path
import threading
from glob import glob
from operator import itemgetter
from itertools import groupby
import re
import cv2

from dataset_utils.src.io.psee_loader import PSEELoader
import math
import numpy as np



def group_by(func):
    def wrapper(*args, **kwargs):
        groups = groupby(np.sort(args[0], order='track_id'), itemgetter('track_id'))
        return np.array([func(g) for k, g in groups])

    return wrapper


@group_by
def maxdate_record(agroup):
    an_array = np.array(list(agroup))
    i = np.argmax(an_array['ts'])
    return an_array[i]


@group_by
def mindate_record(agroup):
    an_array = np.array(list(agroup))
    i = np.argmin(an_array['ts'])
    return an_array[i]


@group_by
def avgdate_record(agroup):
    an_array = np.array(list(agroup))
    return an_array[len(an_array) // 2]

def frequency_aggregation(events, image_shape, delta_t=None):
    image = np.zeros(image_shape)
    for event in events:
        image[event['y'], event['x']] += event['polarity']
        image = 255 / (1 + np.exp(-image / 2))
        image = (image-127.5)
        image = image.astype(np.uint8)
        image[image > 0] += 127
        image = np.clip(image, 0, 255)

        return image


def exponantial_decay_aggregation(events, image_shape, delta_t):
    image = np.zeros(image_shape)
    for event in events:
        image[event['y'], event['x']] += np.exp(-np.abs(event['ts'] - event['polarity']) / delta_t)
    return image


def make_binary_histo(events, size, delta_t=None):
    img = np.zeros((size[0], size[1]), dtype=np.uint8)
    for event in events:
        img[event['y'], event['x']] = 255 * event['polarity']
    # image = (img - 127.5)
    # image = image.astype(np.uint8)
    # image[image > 0] += 127
    # image = np.clip(image, 0, 255)
    return img


def yield_delta_t(filename = 'events.csv', delta_t = 10000):
    event_dict = {'x': None, 'y': None, 'ts': None, 'polarity': None}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, fieldnames=[k for k in event_dict.keys()])
        header = next(reader)
        results = []
        first_timestamp = None

        for line in reader:
            if first_timestamp is None:
                first_timestamp = float(line['ts'])
                last_timestamp = first_timestamp+(delta_t*1e-6)
            if float(line['ts']) < last_timestamp:
                results.append(line)
            else:
                if len(results) > 0:
                    first_timestamp = float(line['ts'])
                    last_timestamp = first_timestamp + (delta_t * 1e-6)
                    df = pd.DataFrame(results)
                    dtypes = [int, int, float, int]
                    for c, d in zip(df.columns, dtypes):
                        df[c] = df[c].astype(d)

                    result = df.to_records(index=False)

                    yield result
                    results = []
                else:
                    last_timestamp += (delta_t * 1e-6)
def yield_event_count(filename = 'events.csv', events_count = 10000):
    event_dict = {'x': None, 'y': None, 'ts': None, 'polarity': None}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, fieldnames=[k for k in event_dict.keys()])
        header = next(reader)
        results = []
        events = 0

        for line in reader:
            if events < events_count:
                results.append(line)
                events+=1
            else:
                if len(results) > 0:
                    events=0
                    df = pd.DataFrame(results)
                    dtypes = [int, int, float, int]
                    for c, d in zip(df.columns, dtypes):
                        df[c] = df[c].astype(d)
                    result = df.to_records(index=False)
                    yield result
                    results = []


def aggregate(filename='events.csv',
              agg_method=yield_event_count,
              delta=10000,
              agg_function=make_binary_histo,
              img_shape=(720, 1280)
              ):
    cv2.namedWindow('out')
    for event_frames in agg_method(filename, delta):
        print(event_frames.dtype)
        frame = agg_function(event_frames, img_shape, delta)
        cv2.imshow('out', frame[::-1, ::-1])
        cv2.waitKey(1)
    cv2.destroyWindow('out')

aggregate(agg_function=make_binary_histo)


