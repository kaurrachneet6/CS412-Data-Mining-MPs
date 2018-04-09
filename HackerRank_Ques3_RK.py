''' Implementation of the Frag Shells Algorithm 
to create partitions '''

import numpy as np
import sys
import itertools

#Reading the dataset file
file = sys.stdin
data = [d.strip() for d in file] 

#Parameter k - No. of partitions 
k = int(data[0])
#Dataset
dataset=[]
for i in range(1, len(data)):
    dataset.append([ elem for elem in data[i].split()])
    
#Transpose of the dataset   
datasetT = np.array(dataset).T.tolist() 

#Sorting and retaining only the unique values from attribute
dataset_unique=[]
for row in datasetT:
    dataset_unique.append(sorted(list(set(row))))

#Counting the no. of elements in the partition
part_size = int(len(dataset[0])/k) #Since no. of attributes is multiple of k

#Creating the partitions 
frags = [[dataset_unique[idx] for idx in range(part_size*i, part_size*(i+1))] for i in range(k)]

#Creating the structure for one, two, etc. tuples
for i in range(len(frags)):
    part = []
    for j in range(len(frags[i])):
        for part_idx in (list(itertools.combinations(range(len(frags[i])),j+1))): part.append(list(part_idx))
    #Appending the final list
    for list_part in part:
        total_list=[]
        total_temp=[]
        for elem in list_part: total_temp.extend(frags[i][elem])
        for mid in (list(itertools.combinations(total_temp,len(list_part)))): total_list.append(list(mid))
        count_list= [sum([set(elem)<set(data_list) for data_list in dataset]) for elem in total_list]
        for count_idx in range(len(count_list)):
            #Neglecting zero counts
            if count_list[count_idx]==0:
                continue
            else:
                for element in total_list[count_idx]:
                    #Printing the list elements 
                    print (element, end = ' ')
                #Printing the count of the tuple
                print(':',count_list[count_idx])
    #A blank line after each partition
    if (i!=len(frags) -1): print () 