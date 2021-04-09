import csv
import sys
from collections import namedtuple
import ast
import math

with open('test.txt') as ifile:
    with open('test_oneclass.txt', 'w') as ofile:
        lines = ifile.readlines()
        for row in lines:
            elements = row.split()
            oline = elements[0]
            if len(elements)==1:
                continue
            elif len(elements) == 2:
                ofile.write(elements[0] + ' ' + elements[1][0:(len(elements[1])-1)] + '0\n')
            else:
                ofile.write(elements[0] + ' ' + elements[1][0:(len(elements[1])-1)] + '0' + ' ' + elements[2][0:(len(elements[2])-1)] + '0\n')