#Selection of attributes in a Decision Tree Classifier using the following metrics:
#1. Information Gain 
#2. Gain Ratio 
#3. Gini Index

import sys
import numpy as np

def InformationGain(attribute, attribute_index, label): #Attribute wise Information Gain
    value, count = np.unique(dataset[:,attribute_index], return_counts = True)
    #Probability of each attributes' value
    prob_value = {value[i]:count[i]/sum(count) for i in range(len(value))}
    col = {v:[sum((dataset[:,attribute_index] == v) & (dataset[:,-1]==l)) for l in label]/sum(dataset[:,attribute_index] == v) \
           for v in prob_value.keys()}
    #Replacing the NaN values by 0
    keys = list(col.keys())
    x = np.array(list(col.values()))
    x[x==0.] = 1
    col = {keys[i]:x[i] for i in range(len(keys))} 
    Info_D_A = {v:-sum(col[v]*np.log2(col[v])) for v in prob_value.keys()} 
    #Information of Dataset D for attribute A at it's value v
    s = sum([Info_D_A[v]*prob_value[v] for v in prob_value.keys()]) #Information for each attribute
    return s
        

def InfoGainMax(attribute, prob_label): #Max Information Gain
    #Replacing the NaN values by 0
    Info_D = -1.*np.sum(np.multiply(prob_label, np.log2(prob_label))) #Information of Dataset D
    #Information gain for each attribute
    Info_attributes = {attribute[i]: Info_D - InformationGain(attribute, i, label) for i in range(len(attribute)-1)}
    #i is the attribute index 
    #Not counting the label as the attribute
    return Info_attributes#Attribute with Max information Gain

def SplitInfo(attr): #Split Info for an attribute
    attribute_index = np.where(attribute==attr)
    value, count = np.unique(dataset[:,attribute_index], return_counts = True)
    #Probability of each attributes' value
    prob_value = {value[i]:count[i]/sum(count) for i in range(len(value))}
    x = np.array(list(prob_value.values()))
    x[x==0.] = 1.
    prob_value = {list(prob_value.keys())[i]:x[i] for i in range(len(prob_value.keys()))}
    s = -1.*sum([np.multiply(prob_value[v],np.log2(prob_value[v])) for v in prob_value.keys()]) #Split Ratio for each attribute
    return s

def GainRatioIndexMax(attribute, prob_label): #Attribute with the Maximum Gain Ratio
    Info_attributes = InfoGainMax(attribute, prob_label)
    keys = list(Info_attributes.keys())
    Info_attributes1 = {i: Info_attributes[i]/SplitInfo(i) for i in keys} #i is the attribute index 
    #Not counting the label as the attribute
    return max(Info_attributes1, key=Info_attributes1.get) #Attribute with Max Gain Ratio Index


def GiniIndex(attribute_index):
    value, count = np.unique(dataset[:,attribute_index], return_counts = True) 
    #Probability of each attributes' value
    prob_value = {value[i]:count[i]/sum(count) for i in range(len(value))}
    col = {v:[sum((dataset[:,attribute_index] == v) & (dataset[:,-1]==l)) for l in label]/sum(dataset[:,attribute_index] == v) \
           for v in prob_value.keys()}
    Gini_D_A = {v:1.-sum(col[v]**2) for v in prob_value.keys()} 
    #Gini Index of Dataset D for attribute A at it's value v
    s = sum([Gini_D_A[v]*prob_value[v] for v in prob_value.keys()]) #Gini Index for each attribute
    return s

def GiniIndexMax(attribute, prob_label):
    Gini_D = 1.- np.sum(prob_label**2) #Information of Dataset D
    Info_attributes = {attribute[i]:Gini_D - GiniIndex(i) for i in range(len(attribute)-1)} #i is the attribute index 
    #Not counting the label as the attribute
    return max(Info_attributes, key=Info_attributes.get) #Attribute with Max Gini Index

file = sys.stdin
data = [d.strip() for d in file] 

#No. of training examples value
train_count = int(data[0])

#Dataset
dataset=[]
for i in range(1, len(data)):
    dataset.append([ elem.strip(',') for elem in data[i].split(',')])
dataset = np.array(dataset)

attribute = dataset[0] #Names of the attributes
dataset = dataset[1:] #The training examples as an array

#Unique labels and the count for each in the training set
label, label_count = np.unique(dataset[:,-1], return_counts = True)

#Probability of each label
prob_label = label_count/sum(label_count)

Info_attributes = InfoGainMax(attribute, prob_label)
print (max(Info_attributes, key=Info_attributes.get)) #Attribute with Maximum Information Gain 
print (GainRatioIndexMax(attribute, prob_label)) #Attribute with Maximum Gain Ratio
print (GiniIndexMax(attribute, prob_label)) #Attribute with Maximum Gini Index
