# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 07:22:43 2022

@author: kamar
"""

import pandas as pd
import numpy as np

yorumlar = pd.read_csv("Restaurant_Reviews.csv")

#noktalama işaretlerinden kurtulmamız lazım

import re
yorum = re.sub(yorumlar['R'])