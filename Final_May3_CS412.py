# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 00:47:27 2018

@author: Chaitra Niddodi, Rachneet Kaur
"""

#IMPORTING PACKAGES 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import random
import time
import seaborn as sns
import ml_metrics
#from sklearn.decomposition import PCA as sklearnPCA #We did not use PCA for final submission
sns.set()  
   

''' FUNCTION FOR VISUALIZATION OF THE MISSING VALUES'''
#Function for plotting missing values in the training dataset features 
def plot_missing_data(train_X):
  plt.figure(figsize = (8,6))
  plt.bar(range(train_X.isnull().sum().shape[0]), train_X.isnull().sum().values)
  plt.xlabel('Features')
  plt.title ('No. of missing values plot for features in Training set')
  plt.savefig('Insurance_MissingValues.jpg')
  plt.show()

''' FUNCTION TO NORMALIZE THE DATASET'''
#Normalization 
def normalize(Train1,b):
  col_names = list(Train1.columns)
  if (b == 'z'):#z-norm
    for i in range(Train1.shape[1]):
      Train1[col_names[i]] = (Train1[col_names[i]] - Train1[col_names[i]].mean(skipna = True)) / Train1[col_names[i]].std(skipna = True)
  else:# min-max norm
    for i in range(Train1.shape[1]):
      Train1[col_names[i]] = (Train1[col_names[i]] - min(Train1[col_names[i]]) )/ ( max(Train1[col_names[i]] ) - min(Train1[col_names[i]]) )
  return Train1

''' FUNCTIONS TO PLOT THE PCA VARIANCE EXPLAINED AND APPLY PCA ON THE DATASET
#Plotting the no. of components in PCA with respect to the cumulative explained variance ratio
def pca_variance_plot():
  cols = train_X.columns.shape[0] #train_X.shape[1]
  cvar = []
  #Applying PCA to reduce the dimensions of the model
  for i in range(1,cols+1):
    model_pca = sklearnPCA(n_components=i)
    model_pca.fit_transform(train_X)
    cvar.append(sum(model_pca.explained_variance_ratio_))
    
  #Plot the no. of components in PCA with respect to the cumulative explained variance ratio
  plt.figure(figsize=(7, 7))
  plt.plot(range(1,cols+1),cvar, '*-b')
  plt.title('Plotting the no. of components used PCA and cumulative explained variance ratio')
  plt.xlabel('No. of components')
  plt.ylabel('Cumulative Explained Variance')
  plt.savefig('PCA_variance_ratio.jpg')
  plt.show()

#Plotting the no. of components in PCA with respect to the cumulative explained variance ratio
#pca_variance_plot()

#Function to apply PCA with 90 components (inferred from the graph of explained variance ratio)
def apply_pca(train_X, test_set):
  model_pca = sklearnPCA(n_components= 90)
  train_pca = model_pca.fit_transform(train_X)
  test_pca = model_pca.transform(test_set)
  train_X = pd.DataFrame(train_pca, index = train_X.index)
  test_set = pd.DataFrame(test_pca, index = test_set.index)
  return train_X, test_set
'''
  
''' FUNCTIONS FOR VISUALIZATIONS OF THE FEATURES IN THE DATASET '''
#Plotting the bar plot of categorical features of the training set
def plot_categorical(feature_name, train_X):
  plt.figure()
  sns.countplot(x=feature_name, data=train_X, palette="Greens_d");
  plt.xlabel(str(feature_name))
  plt.title ('Bar plot for '+str(feature_name)+ ' in Training set')
  plt.savefig('Insurance_'+str(feature_name)+'.jpg')
  plt.show()

#Plotting the density and histogram numerical variable of the training set
def plot_numerical(feature_name, train_X):
  plt.figure()
  sns.distplot(train_X[feature_name]);
  plt.xlabel(str(feature_name))
  plt.savefig('Insurance_'+str(feature_name)+'.jpg')
  plt.title ('Density and Histogram plot for '+str(feature_name)+ ' in Training set')
  plt.show()

#Plotting the bar plot for the response variable 
def plot_response_variable(train_y):
  plt.figure()
  sns.countplot(x= 'Response', data=pd.DataFrame(train_y), palette="Greens_d");
  plt.xlabel('Response Risk Level')
  plt.title ('Bar plot for Response risk level in Training set')
  plt.savefig('Insurance_Response.jpg')
  plt.show()

''' FUNCTION TO ASSIGN CLASSES (DISCRETE) BASED ON THE RESPONSES (CONTINUOUS) 
OBTAINED BY LINEAR  REGRESSION'''
def assign_classes(x, preds): 
    #Function to assign classes to the continuous responses obtained by regression
    classes = []
    for response in list(preds):
        if response >= x[6]: class_assign = 8
        elif response >= x[5]: class_assign = 7
        elif response >=x[4]: class_assign = 6
        elif response >= x[3]: class_assign = 5
        elif response >= x[2]: class_assign = 4
        elif response >= x[1]: class_assign = 3
        elif response >= x[0]: class_assign  = 2
        else: class_assign  = 1
        classes.append(class_assign)
    return classes

''' FUNCTIONS TO FIND THE BEST CUTOFF TO ASSIGN CLASSES BASED ON MINIMIZING 
THE QUADRATIC KAPPA FUNCTION VALUE'''
#Function to calculate the Quadratic Kappa value 
def get_kappa(x): 
    global train_preds, train_y
    kappa_value = -1*ml_metrics.quadratic_weighted_kappa(train_y, assign_classes(x, train_preds))  
    return kappa_value

#Function to minimize quadratic weighted kappa using Grid Search for best cutoffs
def grid_search(x0): 
    #Input: Initial point to start the iterations with
	last_index = 0
	for iter in range(len(x0)): 
	    list_kappa = []
	    for i in range(last_index, len(grid)): 
	        x0[iter] = grid[i]
	        list_kappa.append(get_kappa(x0))
	    last_index =  last_index + np.argmin(list_kappa) +1
	    x0[iter] = grid[last_index-1]
	return x0 #Returns the best grid searched cutoffs 

''' Reading the training and testing sets'''
#training_set = pd.read_csv(sys.argv[1], index_col = 'Id') 
training_set = pd.read_csv('training.csv', index_col = 'Id') 
#20000 rows and 128 columns including the Id column and response columns

#test_set = pd.read_csv(sys.argv[2], index_col = 'Id') 
test_set = pd.read_csv('testing.csv', index_col = 'Id') 
#10000 rows and 127 columns including the Id column (No response column)

train_X =   training_set.drop(['Response'], axis = 1)    
#20000 rows and 127 columns exclusing the Response column 

train_y = training_set['Response']
#20000 rows with the response variable

''' FEATURE SELECTION AND MODIFICATIONS'''
#Plotting missing values in the training dataset features    
#plot_missing_data(train_X)

#Total 20000 rows with 0 rows with more than 50% column entries as Null
train_X = train_X[ (train_X.isnull().sum(axis = 1) < 0.5*train_X.shape[1]) ]
#Keeping all rows 

#Collecting the continuous attributes - 13 total
continuous = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4',\
             'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4','Family_Hist_5'] 

#Collecting discrete attributes 
discrete = ["Medical_History_1", "Medical_History_10", "Medical_History_15", \
            "Medical_History_24", "Medical_History_32"]

numerical = list(continuous) + list(discrete)

#Filling the NaNs in Numerical attributes by mean of the column 
for col in numerical:
  train_X[col].fillna(train_X[col].mean(skipna = True), inplace = True)
  test_set[col].fillna(test_set[col].mean(skipna = True), inplace = True)


#Introducing a new column which is the product of user's BMI index and age since these are very dependent columns
#so we wish to combine them in a ussually used metric
train_X['Newcol_BMI_times_age'] = train_X['BMI']*train_X['Ins_Age']
test_set['Newcol_BMI_times_age'] = test_set['BMI']*test_set['Ins_Age']


#Looking at the Product_Info_4 column, we can see that it has the most variation of values among all, hence,
#let's try putting a cutoff, to treat it as a categorical 0/1 since all the other Product_Info are categorical
train_X['Product_Info_4_cutoff'] = train_X['Product_Info_4'] < 0.05 #0.05 selected based on cross validation results
train_X['Product_Info_4_cutoff']  = train_X['Product_Info_4_cutoff'] *1
test_set['Product_Info_4_cutoff'] = test_set['Product_Info_4'] < 0.05
test_set['Product_Info_4_cutoff']  = test_set['Product_Info_4_cutoff'] *1  
      
#Filling NaNs in the Categorical attributes by the mode of the column
categorical = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', \
               'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', \
               'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5',\
               'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', \
               'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8',\
               'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', \
               'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', \
               'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12',\
               'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17',\
               'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21',\
               'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26',\
               'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30',\
               'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', \
               'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39',\
               'Medical_History_40', 'Medical_History_41']

#Dummy coding for the string classes in Product_Info_2 variable 
train_X['Product_Info_2'] = pd.factorize(train_X['Product_Info_2'])[0]
test_set['Product_Info_2'] = pd.factorize(test_set['Product_Info_2'])[0]

#Filling the NaN in the Categorical variable using the mode
for col in categorical:
  train_X[col].fillna(mode(train_X[col].dropna())[0][0], inplace = True)
  test_set[col].fillna(mode(train_X[col].dropna())[0][0], inplace = True)
  
#Dummy variables 
dummy = ["Medical_Keyword_{}".format(i) for i in range(1, 49)]
#Filling NaN in dummy variable using mode
for col in dummy:
  train_X[col].fillna(mode(train_X[col].dropna())[0][0], inplace = True)
  test_set[col].fillna(mode(train_X[col].dropna())[0][0], inplace = True)

'''Plotting the categorical, numerical and response variable to visualize the dataset'''
#plot_categorical('Insurance_History_1', train_X) #Plotting the bar plot categorical variable of the training set
#plot_response_variable(train_y) #Plotting the bar plot for the response variable 
#plot_numerical('Employment_Info_1', train_X) #Plotting the density and histogram numerical variable of the training set
               
#Looking at the Medical_History_15, Medical_History_24, Medical_History_2, Medical_History_32 columns, 
#we can see that it has the most variation of values among all, hence,
#let's try putting a cutoff, to treat it as a categorical 0/1
train_X['Medical_History_2_cutoff'] = train_X['Medical_History_2'] < 50 #Selected on basis of Cross Validation results
train_X['Medical_History_2_cutoff']  = train_X['Medical_History_2_cutoff'] *1
test_set['Medical_History_2_cutoff'] = test_set['Medical_History_2'] < 50
test_set['Medical_History_2_cutoff']  = test_set['Medical_History_2_cutoff'] *1

train_X['Medical_History_15_cutoff'] = train_X['Medical_History_15'] < 24 #Selected on basis of Cross Validation results
train_X['Medical_History_15_cutoff']  = train_X['Medical_History_15_cutoff'] *1
test_set['Medical_History_15_cutoff'] = test_set['Medical_History_15'] < 24
test_set['Medical_History_15_cutoff']  = test_set['Medical_History_15_cutoff'] *1
        
train_X['Medical_History_24_cutoff'] = train_X['Medical_History_24'] < 25 #Selected on basis of Cross Validation results
train_X['Medical_History_24_cutoff']  = train_X['Medical_History_24_cutoff'] *1
test_set['Medical_History_24_cutoff'] = test_set['Medical_History_24'] < 25
test_set['Medical_History_24_cutoff']  = test_set['Medical_History_24_cutoff'] *1
        
train_X['Medical_History_32_cutoff'] = train_X['Medical_History_32'] < 25 #Selected on basis of Cross Validation results
train_X['Medical_History_32_cutoff']  = train_X['Medical_History_32_cutoff'] *1
test_set['Medical_History_32_cutoff'] = test_set['Medical_History_32'] < 25
test_set['Medical_History_32_cutoff']  = test_set['Medical_History_32_cutoff'] *1

        
#Adding a new column which is the square root of the BMI-index
train_X['square_rootBMI'] = np.sqrt(train_X['BMI'])
test_set['square_rootBMI'] = np.sqrt(test_set['BMI'])

'''FINAL MODEL IMPLEMETATION'''
start_time = time.time() #Start time
X = train_X.values
test_X = test_set.values
ones_col = np.ones((len(X),1)) #Stacking a column of ones for the intercept
X = np.hstack((ones_col,X))
ones_col = np.ones((len(test_X),1))
test_X = np.hstack((ones_col,test_X))
Y = train_y.values

#We used the linear regression model for our final submission 
coeff = (np.linalg.pinv(X.T.dot(X)).dot(X.T)).dot(Y)
train_preds = X.dot(coeff) 
#The predictions (continous values are a product of X and coefficients from linear regression )

#CROSS VALIDATION
#scores = cross_val_score(linear_model, train_X, train_y, cv=5)
#model accuracy for validation sets  
#print ('Mean of 5 fold CV scores: ', np.mean(scores),' and Standard Deviation = ', np.std(scores))

test_preds = test_X.dot(coeff)

grid = sorted(np.linspace(1,7, num = 7*20)) #To make a grid between 1 and 7 to find the best cutoffs
#Initialize random values as class boundaries to start the grid search with
x0 = [1, 2, 3, 4, 5, 6, 7] 
#Initialization with random values or most intuitive cuttoffs, simply 1 to 7 

#x0 = random.sample(range(1,8), 7) 
#Dear TA: - if you wish to use random, please use them, we used 1-7 to make submissions so that our 
#code when run again by you matches the old answers on submissions made by us

x0 = grid_search(x0)  #Call grid_search() with above random values
x_optimized = grid_search(x0)  #Call grid_search() again after getting fairly optimal values from last call
print ('Accuracy in terms of quadratic weighted kappa =',-get_kappa(x_optimized))
predictions = assign_classes(x_optimized, test_preds)
end_time = time.time()
print ('Execution Time =', end_time - start_time, 'seconds')
                           
#Creating the final submission File
submission = pd.DataFrame({
        "Id": test_set.index,
        "Response": predictions
    })
submission.to_csv('submission.csv', index=False)

# For our first submission, we used linear regression with grid search and only 2 extra columns 
# Newcol_BMI_times_age and square_rootBMI were added in feature engineering step 

# For our second submission, we used linear regression with grid search and only 6 extra columns 
# Newcol_BMI_times_age, square_rootBMI, Product_Info_4_cutoff, Medical_History_2_cutoff, Medical_History_15_cutoff, Medical_History_24_cutoff, Medical_History_32_cutoff  were added in feature engineering step
