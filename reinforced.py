# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 17:19:53 2022

@author: kamar
"""

#######1. UCB
#random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N = 10000 #10bin satır
d = 10 #ilanlar 10 tane
toplam = 0
secilenler = []
for n in range(0,N): #n kaçınıc tur
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    toplam = toplam + odul
    
    
plt.hist(secilenler)
plt.show()

###ucb
