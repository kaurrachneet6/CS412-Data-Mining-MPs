''' Implementation of the Apriori Algorithm to mine frequent 
pattens and classify Closed patterns out of these Frequent patterns'''

import numpy as np
import sys

#Reading the input
file = sys.stdin
data = [d.strip() for d in file] 

#Min support value
min_sup = int(data[0])

#Dataset
dataset=[]
for i in range(1, len(data)):
    dataset.append([ elem for elem in data[i].split()])

#Function to compute the frequent 1 itemsets dictionary with items and their counts 
def frequent_1itemsets(min_sup): 
    freq1= {} #Dictionary for Frequent 1 items with their count 
    for trans in dataset: #For each transaction
        for item in trans: #For each item in the transaction
            try: freq1[item]+=1
            except: freq1[item] = 1
    freq1 = {item:count for (item,count) in freq1.items() if count>=min_sup} #Count must be >= Minimum Support 
    return freq1

#Function to join the itemsets to produce frequent k+1 itemsets from joining frequent k itemsets
def increment_itemset(frequent_kitemsets, new_len): 
    frequent_increment = []
    length = len(frequent_kitemsets) #Length of the frequent k itemset
    for i in range(length):
        for j in range(i + 1, length):
            if (sorted(list(frequent_kitemsets[i])[:new_len - 2]) == sorted(list(frequent_kitemsets[j])[:new_len - 2])):
                temp = list(frequent_kitemsets[i]) #Joining the itemsets
                [temp.append(x) for x in list(frequent_kitemsets[j]) if x not in temp]
                frequent_increment.append(sorted(temp))
    return frequent_increment

#Function to calculate the support of each candidate in frequent_kitemsets and return only the ones satisfying min_sup
def calc_support(frequent_kitemsets, min_sup):
    freq_k = {}
    for trans in dataset: #For each transaction
        for items in frequent_kitemsets: 
            if (set(items).issubset(set(trans))):
                try: freq_k[' '.join(items)]+=1
                except: freq_k[' '.join(items)] = 1
    freq_k = {items:count for (items,count) in freq_k.items() if count>=min_sup} #Count must be >= Minimum Support 
    return freq_k

#Function to output in the given format
def output_format(frequent_dict): #Output format  
    sorted_list = sorted(frequent_dict.items(), key = lambda x: (-x[1],x[0]))
    for tuple in sorted_list:
        print (str(tuple[1])+' ['+ tuple[0]+']')

#Function to find Closed Patterns
def closed_patterns(frequent_dict):
    freq_set = [ x for x in list(frequent_dict.keys())] #Frequent itemsets
    closed = frequent_dict
    for key1 in freq_set:
        for key2 in freq_set:
            if (set(key2) > (set(key1))) & (frequent_dict[key2] == frequent_dict[key1]): #If superset and same support
                closed[key1] = 0 #Then not closed 
    closed = {key:value for key, value in closed.items() if value}
    return closed
    
    
f1= frequent_1itemsets(min_sup)
frequent_dict = f1 #Dictionary with frequent itemsets and their counts

f1_list = [ [x] for x in list(f1.keys())] #Frequent 1 itemsets
frequent_kitemsets = increment_itemset(f1_list, 2) #Frequent 2 itemsets
fk = calc_support(frequent_kitemsets, min_sup) #Calculating support of each item and neglecting ones < min_sup
frequent_dict.update(fk)

for len_k in range(3, len(f1.keys())+1):
    frequent_kitemsets = increment_itemset(list(frequent_kitemsets), len_k)
    fk = calc_support(frequent_kitemsets, min_sup) #Dictionary of frequentk itemsets with values and count of each frequent_k itemset
    if (len(fk)==0): break
    frequent_dict.update(fk)

#Printing the result in the correct Output Format
output_format(frequent_dict)
closed_patterns = closed_patterns(frequent_dict)
print ()
output_format(closed_patterns)
