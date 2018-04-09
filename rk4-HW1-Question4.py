# -*- coding: utf-8 -*-
"""
Reducing the given data's dimensions using Principal Component Analysis
and finding the most similar Patient vectors to the given Patient vector  
using one of the following distance metrics given in input:
(a) Minkowski distance where h = 1 (Manhattan distance)
(b) Minkowski distance where h = 2 (Euclidean distance)
(c) Minkowski distance where h = infinite (Supremum distance)
(d) Cosine similarity

"""

import numpy as np
from sklearn.decomposition import PCA

#Reading the dataset file
file_name = 'HW1-data/Q4-analysis-input.in'
with open(file_name) as f:
    file = f.readlines()
data = [d.strip() for d in file] 

#Reading the parameters D, N, Distance Type, X, P and Matrix of data
D, N, dist_type, X = int(data[0]), int(data[1]), data[2], int(data[3])
P = [int(p) for p in data[4].split()]
matrix = []
for i in range(5, N+5):
    matrix.append([int(d) for d in data[i].split()])

#Function to compute the distance based on choice (Distance Type) and returns 
#the indices of closest 5 patients
def return_index(D, N, dist_type, P, matrix):
    if (dist_type=='1'): #Manhattan Distance 
        distance = [sum(abs(x-y) for x,y in zip(P,matrix[i])) for i in range(N)]
    elif(dist_type =='2'): #Euclidean Distance 
        distance = [np.sqrt(sum(pow(x-y,2) for x, y in zip(P,matrix[i]))) \
                    for i in range(N)]
    elif(dist_type == '3'): #Supremum Metric
        distance = [max(abs(x-y) for x,y in zip(P,matrix[i])) for i in range(N)]
    elif(dist_type == '4'): #Cosine Similarity 
        num = [sum(x*y for x,y in zip(P,matrix[i])) for i in range(N)]
        den = [np.sqrt(sum([p**2 for p in P]))*np.sqrt(sum([m**2 for m in\
               matrix[i]])) for i in range(N)]
        distance = [1 - (x/y) for x,y in zip(num, den)]
    else:
        print ('Incorrect Distance Type') 
    indexes = np.argsort(distance, kind = 'mergesort')+1 #Mergesort preserves order while breaking ties
    for idx in indexes[:5]:
        print (idx)

#Case when no PCA is implemented        
if(X == -1):
    return_index(D, N, dist_type, P, matrix)
#Case when PCA needs to be implemented 
else:
    matrix_P = np.vstack((matrix, P)) #Stacking the data for 100 patients and Patient P together
    pc = PCA(n_components=X).fit(matrix_P) #Applying PCA with X components to 100 patients and patient P
    data_PCA = pc.transform(matrix_P) #Transforming the stacked data with 101 patients with PCA to a new space
    matrix_pca = data_PCA[:-1] #Separating the data matrix for 100 patients and patient P
    P_pca = data_PCA[-1]
    return_index(D, N, dist_type, P_pca, matrix_pca) #Computing the distance in transformed space 
    print(sum(pc.explained_variance_ratio_))