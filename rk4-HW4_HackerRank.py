import sys
from collections import OrderedDict

file = sys.stdin
data = [d.strip() for d in file] 

#Min support value
min_sup = int(data[0])

#Dataset
dataset=[]
for i in range(1, len(data)):
    dataset.append([ elem.lower().strip('.') for elem in data[i].split()])
    #Converting the words to lower case and Removing full stops 
#List of Stop Words 
stop_words = ['a', 'an', 'are', 'as', 'at', 'by', 'be', 'for', 'from', \
              'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to' \
              'was', 'were', 'will', 'with']

for data in dataset: #Eliminating the stop words from the dataset
    for w in stop_words:
        try: data.remove(w)
        except: pass

filtered = []
#Filtering the dataset 
for data in dataset: #Combining the words with 'and' and ',' into tuples 
    filtered_dataset = []
    for i in range(len(data)):
        elem = [data[i]]
        if data[i] == 'and':
            del filtered_dataset[-1]
            comma = 0
            elem = []
            j = 2
            while (i-j >= 0 and data[i-j][-1] == ','):
                del filtered_dataset[-1]
                data[i-j] = data[i-j][:-1]
                j+=1
                comma = 1
            if (comma == 1):
                for k in range(j-1, 1, -1):
                    elem.append(data[i-k])
            elem.append(data[i-1])
            elem.append(data[i+1])
        if ((i-2 > 0) and (data[i-2] == 'and')): #Combining all the ',' and 'and' elements in the same list
            del filtered_dataset[-1]
        filtered_dataset.append(elem) 
    filtered.append(filtered_dataset) 
    #The filtered dataset with all words in lowercase and no stopwords and ',' and 'and' merged
            
#Function to compute the frequent 1 itemsets dictionary with items and their counts 
def frequent_1itemsets(min_sup): 
    freq1= {} #Dictionary for Frequent 1 items with their count 
    for trans in filtered: #For each transaction
        for item in trans: #For each item in the transaction
            try: freq1[item]=0
            except: 
                for elem in item:
                    freq1[elem]=0 #Collecting all unique 1 length items
    #Counts of each unique 1 length item
    for freq_keys in freq1.keys():
        for trans in filtered: #For each transaction
            for item in trans: #For each item in the transaction
                try:
                    if(set([freq_keys]).issubset([item])):
                        freq1[freq_keys] +=1
                        break
                except:
                    if(set([freq_keys]).issubset(item)):
                        freq1[freq_keys] +=1
                        break
    freq1 = {item:count for (item,count) in freq1.items() if count>=min_sup} #Count must be >= Minimum Support 
    return freq1

#Function to join the itemsets to produce frequent 2 itemsets from joining frequent 1 itemsets
def increment_itemset(frequent_kitemsets, new_len): 
    frequent_increment = []
    length = len(frequent_kitemsets) #Length of the frequent k itemset
    for i in range(length):
        for j in range(length):
                temp = list(frequent_kitemsets[i]) #Joining the itemsets
                [temp.append(x) for x in list(frequent_kitemsets[j])]
                #Making elements of form [ab] without any tuple for the frequent 2 dataset
                frequent_increment.append(temp)
                if(j>i):
                    temp1 = tuple(sorted((frequent_kitemsets[i][0],frequent_kitemsets[j][0]))) 
                    #Making elements of form (ab) in tuples for the frequent 2 dataset 
                    frequent_increment.append([temp1])
    return frequent_increment


#To check if a list 'l' is non empty and sorted in strictly increasing order (size_itemsets)
def check_increasing(l, size):
    if (len(l)==size):
        try: l[0] = [min(l[0])]
        except: pass
        for i in range(1, len(l)):
            try: l[i] = [max(l[i])]
            except: continue
    list = []
    for item in l:
        for elem in item:
            try:
                list.append(elem)
            except:
                continue
    if len(list) < size: returned = False
    elif len(list) == size:
        returned = bool(sum([list[k+1]-list[k]>0 for k in range(len(list)-1)]) == len(list)-1)
    else:
        returned = bool(sum([list[k+1]-list[k]>=0 for k in range(len(list)-1)]) == len(list)-1)
    return returned

#Function to calculate the support of each candidate in frequent_2itemsets and return only the ones satisfying min_sup
def calc_support(frequent_kitemsets, min_sup):
    freq_k = {}
    for items in frequent_kitemsets:
        freq_k[tuple(items)]=0
        for trans in filtered: #For each transaction
            occured = 0 #Whether this memeber (items) appeared in the transaction or not
            occur = OrderedDict()
            for item in items:
                occur[item] =  []
            #Dict to store occurance indices of each item in items 
            #print ('occur begin = ', occur)
            for i in range(len(trans)):
                for item in items:
                    if type(item)==tuple:
                        if(set(list(item)).issubset(trans[i])):
                            occur[item].append(i)
                    else:
                        if(set([item]).issubset(trans[i])):
                            occur[item].append(i)
            #print ('occur = ', occur)
            occured = check_increasing(list(occur.values()), len(items)) 
            #print (occured)
            freq_k[tuple(items)]+=occured  
    #print ('freq_k =', freq_k)
    freq_k = {items:count for (items,count) in freq_k.items() if count>=min_sup} #Count must be >= Minimum Support 
    return freq_k


#Function to join the itemsets to produce frequent 3 or more itemsets from joining frequent 2 or more itemsets
def increment_itemset_high(frequent_kitemsets, new_len): 
    #print (' input = ', frequent_kitemsets, new_len)
    frequent_increment = []
    length = len(frequent_kitemsets) #Length of the frequent k itemset
    for i in range(length):
        for j in range(length):
            #Case when none of the sets to be merged is a tuple
            if (not((type(list(frequent_kitemsets[i][0])[0])== tuple) or (type(list(frequent_kitemsets[j][0])[0]) == tuple))):
                if (list(frequent_kitemsets[i][0])[1:] == list(frequent_kitemsets[j][0])[:-1]):
                    temp = list(frequent_kitemsets[i][0]) #Joining the itemsets
                    [temp.append(list(frequent_kitemsets[j][0])[-1])]
                    frequent_increment.append(temp)
                    
            #Case when first one of the sets to be merged is a tuple
            if ((type(list(frequent_kitemsets[i][0])[0])== tuple) and not(type(list(frequent_kitemsets[j][0])[0]) == tuple)):
                if (list(list(frequent_kitemsets[i][0])[0])[1:] == list(frequent_kitemsets[j][0])[:-1]):
                    temp = list(frequent_kitemsets[i][0]) #Joining the itemsets
                    [temp.append(x) for x in list(frequent_kitemsets[j][0])[-1] ]
                    frequent_increment.append(temp)
                    
            #Case when second one of the sets to be merged is a tuple
            if (not((type(list(frequent_kitemsets[i][0])[0])== tuple)) and (type(list(frequent_kitemsets[j][0])[0]) == tuple)):
                if (list(frequent_kitemsets[i][0])[1:] == list(list(frequent_kitemsets[j][0])[0])[:-1]):
                    temp = list(frequent_kitemsets[i][0])[1:] + [(list(frequent_kitemsets[j][0]))[0]]#Joining the itemsets
                    frequent_increment.append(temp)
                
            #Case when both of the sets to be merged are tuples
            if ((type(list(frequent_kitemsets[i][0])[0])== tuple) and (type(list(frequent_kitemsets[j][0])[0]) == tuple)):
                if (list(list(frequent_kitemsets[i][0])[0])[1:] == list(list(frequent_kitemsets[j][0])[0])[:-1]):
                    temp = [(list(frequent_kitemsets[i][0]))[0]]+[(list(frequent_kitemsets[j][0]))[0]]#Joining the itemsets
                    frequent_increment.append(temp)
    return frequent_increment #Merged frequent itemsets 


#Function to output in the given format
def output_format(fk): #Output format  
    #First sort wrt Support, then according to len i.e. tuples are given preference 
    #Eg: (ab) is given preference over ab
    #Then alphabetically sort the rest
    sorted_list = sorted(fk.items(), key = lambda x: (-x[1], len(x[0]), x[0]))
    #print (sorted_list)
    for tup in sorted_list:
        x = str()
        #Style of a tuple [(ab)]
        for i in range(len(tup[0])):
            if (type(tup[0][i])==tuple):
                x+='('
                for j in tup[0][i]:
                    x+=j
                x+=')'
            else:   #If not a tuple [ab]
                x+=tup[0][i]
                if (i!=len(tup[0])-1): x+=' '
        print (str(tup[1])+ ' ['+x+ ']')        
        
f1 = frequent_1itemsets(min_sup) #Frequent 1 itemsets with their counts
f1_list = [ [x] for x in list(f1.keys())] #Frequent 1 itemsets
frequent_kitemsets = increment_itemset(f1_list, 2) #Frequent 2 itemsets
#print (' 2 itemsets=', frequent_kitemsets)
fk = calc_support(frequent_kitemsets, min_sup) #Calculating support of each item and neglecting ones < min_sup
if (fk=={}):
    final = f1
    done = 1
else:
    new_len = 3
    done = 0

while (done != 1): #'done' decides when to stop merging further frequent itemsets 
    fk_list = [ [x] for x in list(fk.keys())] #Frequent 2 itemsets
    frequent_kitemsets = increment_itemset_high(fk_list, new_len )
    fk_new = calc_support(frequent_kitemsets, min_sup) #Calculating support of each item and neglecting ones < min_sup
    if (fk_new=={}): #If the new merged set if empty, hence the last one was longest 
        final = fk
        done = 1
    else:
        fk=  fk_new #Else, we keep on continue to merge 
        new_len+=1
        done = 0       
output_format(final) #Output in the appropriate format
