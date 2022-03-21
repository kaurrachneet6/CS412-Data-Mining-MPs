# CS412 - UIUC - Data-Mining-MPs
Data Mining Machine Problems 

**Problem 1:** Reducing the given data's dimensions using Principal Component Analysis and finding the most similar Patient vectors to the given Patient vector

**Problem 2:** Implementation of the Frag Shells algorithm for fragmentation 

**Problem 3:** Implementation of the Apriori Algorithm to find frequent patterns 

**Problem 4:**  Implement GSP sequential pattern mining algorithm to find frequent phrases from the text

**Problem 5:** Implement selection of attributes in a Decision Tree Classifier using the following metrics:
* Information Gain 
* Gain Ratio 
* Gini Index

**Final project:** Insurance classification problem

The Insurance classification problem is a supervised learning multi class classification problem. The aim of the given problem is to classify the risk in providing insurance to new clients based on data from past clients. Risk is classified into 8 levels or classes namely 1, 2, ..., 8. The key idea in such classification is to find similarities in the attribute values of the new client with one or more of the past clients. For this purpose, we are given a training dataset containing 20000 client records with 126 (excluding the Client Id and response attributes) continuous, categorical, discrete and dummy attributes like insurance history, medical history, employment information, age, BMI etc. describing almost every aspect of the past clients. We first do some pre-processing on the given training dataset (i.e, data of past clients). Based on the characteristics of the dataset in this problem we try certain classifiers (Naive Bayes, KNN, SVM etc.) and evaluate their accuracy to choose the one that provides the most accurate results.
