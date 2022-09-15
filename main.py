import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pandas_profiling import ProfileReport

df= pd.read_csv("D:\\Study\\Data Science\\Python\\ineuron\\Data_Set\\advertising.csv")
pf=ProfileReport(df)
pf.to_file('advertising_profiling1.html')