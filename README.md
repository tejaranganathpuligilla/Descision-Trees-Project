# Descision-Trees-Project
Explaination of Descision Trees with a dataset 
![image](https://user-images.githubusercontent.com/82372055/118543481-ecfb1f00-b771-11eb-90b7-fd0040ff6c30.png)


Def : A decision tree is a flowchart-like structure in which each internal node represents a test on a feature, each leaf node represents a class label and branches represent conjunctions of features 
## Import the libraries for pandas and plotting.

Pandas is used for data analysis. The library allows various data manipulation operations such as merging, reshaping, selecting, as well as data cleaning, and data wrangling features

Numpy is used in the industry for array computing

Seaborn is a Python library used for enhanced data visualization
```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
```
## Reading the Data
We use pandas to read loan_data.csv as a data frame called loans

```python
loans = pd.read_csv("/content/loan_data.csv")
```

## Getting More Information
```python
loans.info()
loans.head()
loans.describe()
```
![image](https://user-images.githubusercontent.com/82372055/118543799-61ce5900-b772-11eb-916e-5c9eebd9072a.png)
## Exploratory Data Analysis
We'll use seaborn and pandas built-in plotting capabilities to do some data visualizations!

```python
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')

```
from this plot, we get the histplot with the condition in credit policy
![image](https://user-images.githubusercontent.com/82372055/118543860-7ca0cd80-b772-11eb-8184-7108d48763c1.png)

```python
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')

```
from this plot, we get the histplot with the condition in not.fully.paid
![image](https://user-images.githubusercontent.com/82372055/118543904-8aeee980-b772-11eb-9bb2-6384556a7ad6.png)

Creating a count plot using seaborn showing the counts of loans by purpose
```python
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
```
![image](https://user-images.githubusercontent.com/82372055/118543970-9e01b980-b772-11eb-8d7a-6802d2482ebe.png)
creating a joint plot to see the trend between FICO and interest rate
```python
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
```
![image](https://user-images.githubusercontent.com/82372055/118544024-aeb22f80-b772-11eb-90ec-8cb814b7761b.png)
## checking data any null-values

```python
loans.info()
```

#get dummies for the dataset and a categorical feature list 

categorical features

```python
cat_feats = ['purpose']
```
assigning dummies
```python
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

```
checking data with dummies

```python
final_data.info()

```
![image](https://user-images.githubusercontent.com/82372055/118544093-c984a400-b772-11eb-8686-a4cf7998cf4a.png)
## now our data is ready for Train Test Split

importing train_test_split from sklearn.model_selection

```python
from sklearn.model_selection import train_test_split
```
TTS
```python
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
```

## now lets train our Decision Tree Model

```python
from sklearn.tree import DecisionTreeClassifier
```
Creating an instance of DecisionTreeClassifier() called dtree and fit it to the training data
```python
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)
```
# now lets predict and evaluate our model 
```python
predictions = dtree.predict(x_test)
```
to check f1-score , precision and recall we need to import classification_report,confusion_matrix
```python
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
```
![image](https://user-images.githubusercontent.com/82372055/118544149-de613780-b772-11eb-8e1e-18a6f94a87f8.png)
now let's check the confusion matrics which gives us the datils of misclassified and correctly classified points
```python
print(confusion_matrix(y_test,predictions))
```
![image](https://user-images.githubusercontent.com/82372055/118544336-ede08080-b772-11eb-828a-f56e3509638b.png)
