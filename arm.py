# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:29:22 2022

@author: kamar
"""
#Association Rule Mining
#1. Apriori Algoritması

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('sepet.csv', header = None)

from apyori import apriori
apriori()