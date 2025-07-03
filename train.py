import sys
import os

if len(sys.argv) != 3:
    raise NotImplementedError
gpus = sys.argv[1]
valid_period_idx = sys.argv[2]
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

from model import *


def get_name(test_time, valid_period, market_name):
    return '-'.join(filter(None, [
        f'{market_name}{test_time}',
        f'{valid_period}',
        f'{get_basic_name()}',
    ])).replace(' ', '')


season_list = ["2023q1", "2023q2", "2023q3", "2023q4", "2024q1", "2024q2"]
for market in ["ALL"]:
    for season in season_list:
        valid_period = rf"validperiod{valid_period_idx}"
        train(args, get_name(season, valid_period, market), market, season)
