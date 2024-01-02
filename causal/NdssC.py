# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:05:58 2023

@author: hanmin
"""
import numpy as np
import xlrd
from sklearn.preprocessing import StandardScaler

from econml.orf import DMLOrthoForest
from econml.sklearn_extensions.linear_model import WeightedLasso
  

# Loading data
data1 = xlrd.open_workbook(r'/GPUFS/ac_gig_bjin_1/hanmin/data//pmt.xlsx')
table1 = data1.sheets()[0]
m1=table1.nrows
n1=table1.ncols
A1=np.zeros((m1,n1)) 
y1=np.zeros((m1,1)) 
for i in range(m1):
    A1[i,:]=table1.row_values(i) 
    y1[i,:]=1
data2 = xlrd.open_workbook(r'/GPUFS/ac_gig_bjin_1/hanmin/data//nonpmt.xlsx')
table2 = data2.sheets()[0]
m2=table2.nrows
n2=table2.ncols
A2=np.zeros((m2,n2)) 
y2=np.zeros((m2,1)) 
for i in range(m2):
    A2[i,:]=table2.row_values(i) 
    y2[i,:]=0
A=np.append(A1,A2,axis=0)
y=np.append(y1,y2,axis=0)
ss = StandardScaler()
A = ss.fit_transform(A)

# Define estimator inputs
W = A
W = np.delete(W,[1679,668,679,1474,1121,196,676,650,184,857,1694,614,940,1693,598,36,709,1214,197,663],axis=1)

Y = y
T = A[:,1474]
X = np.c_[A[:,668],A[:,1679],A[:,679],A[:,1121],A[:,196],A[:,676],A[:,650],A[:,184],A[:,857],A[:,1694],A[:,614],A[:,940],A[:,1693],A[:,598],A[:,36],A[:,709],A[:,1214],A[:,197],A[:,663]]


# Initiate an EconML cate estimator
est = DMLOrthoForest(
                     model_Y=WeightedLasso(alpha=0.01),
                     model_T=WeightedLasso(alpha=0.01))
# Fit through dowhy
est_dw = est.dowhy.fit(Y, T, X=X, W=W)

identified_estimand = est_dw.identified_estimand_


# Refute
# Add Random Common Cause
res_random = est_dw.refute_estimate(method_name="random_common_cause")
print(res_random)
# Replace Treatment with a Random (Placebo) Variable
res_placebo = est_dw.refute_estimate(
    method_name="placebo_treatment_refuter", placebo_type="permute", 
    num_simulations=3
)
print(res_placebo)
# Remove a Random Subset of the Data
res_subset = est_dw.refute_estimate(
    method_name="data_subset_refuter", subset_fraction=0.8, 
    num_simulations=3)
print(res_subset)
