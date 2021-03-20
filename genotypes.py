""""This file contains the best architecture found.
"""
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

yolo = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 1),
            ('dil_conv_3x3', 3),
            ('skip_connect', 1),
            ('sep_conv_3x3', 0)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_5x5', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 1),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6)
)
