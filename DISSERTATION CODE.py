#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics


# In[2]:


#importing the dataset and assigning a new name to it.
data = pd.read_csv("LOANZIM.csv")


# In[3]:


#checking the number of rows and columns the dataset has.
pd.set_option("display.max_columns", 500)
data.shape


# In[4]:


#deriving the proportion of missing values on each column
pd.set_option("display.max_rows", None)
round(data.isnull().sum()/len(data.index), 2)*100


# In[5]:


#removing columns with missing values
missing_values = data.columns[100*(data.isnull().sum()/len(data.index))>0]
missing_values


# In[6]:


missing_values1 = ['ContractEndDate', 'DateOfBirth', 'MonthlyPayment', 'County',
       'NrOfDependants', 'EmploymentDurationCurrentEmployer',
       'EmploymentPosition', 'WorkExperience', 'PlannedPrincipalTillDate',
       'PlannedInterestTillDate', 'LastPaymentOn', 'CurrentDebtDaysPrimary',
       'DebtOccuredOn', 'CurrentDebtDaysSecondary',
       'DebtOccuredOnForSecondary','PrincipalOverdueBySchedule', 'PlannedPrincipalPostDefault',
       'PlannedInterestPostDefault', 'EAD1', 'EAD2', 'PrincipalRecovery',
       'InterestRecovery', 'RecoveryStage', 'StageActiveSince', 'Rating',
       'EL_V0', 'Rating_V0', 'EL_V1', 'Rating_V1', 'Rating_V2',
       'ActiveLateCategory', 'WorseLateCategory', 'CreditScoreEsMicroL',
       'CreditScoreEsEquifaxRisk', 'CreditScoreFiAsiakasTietoRiskGrade',
       'CreditScoreEeMini', 'PrincipalWriteOffs',
       'InterestAndPenaltyWriteOffs', 'InterestAndPenaltyBalance',
       'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan',
       'PreviousRepaymentsBeforeLoan', 'PreviousEarlyRepaymentsBefoleLoan',
       'PreviousEarlyRepaymentsCountBeforeLoan', 'GracePeriodStart',
       'GracePeriodEnd', 'NextPaymentDate', 'NextPaymentNr',
       'NrOfScheduledPayments', 'ReScheduledOn', 'PrincipalDebtServicingCost',
       'InterestAndPenaltyDebtServicingCost', 'ActiveLateLastPaymentCategory']
data.drop(missing_values1, axis = 1, inplace = True)
data.shape


# In[7]:


data.columns


# In[8]:


#Removing irrelevant and duplicate columns that have nothing to do with loan default prediction 
irr_variables = ['ReportAsOfEOD', 'LoanId', 'LoanDate', 'FirstPaymentDate', 'MaturityDate_Original',
       'MaturityDate_Last','Country', 'AppliedAmount','IncomeFromPrincipalEmployer', 'IncomeFromPension',
       'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare','IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther','RefinanceLiabilities', 'DebtToIncome', 'FreeCash', 'MonthlyPaymentDay',
       'ActiveScheduleFirstPaymentReached','ModelVersion','Restructured', 'PrincipalPaymentsMade',
       'InterestAndPenaltyPaymentsMade']
data.drop(irr_variables, axis = 1, inplace = True)
data.shape


# In[9]:


data.info()


# In[10]:


#Creating the predictor variable
data['Status'].value_counts()


# In[11]:


#Removing the 'current' status in 'Status' variable
data = data[data['Status'] != 'Current']


# In[12]:


#Creating the target variable, incoperating "DefaultDate" into "Status"
#Where there is a date provided a 0 will be used to denote 'defaulted', and where there 
#is no date, a 1 will denote "not defaulted"

data["Default"] = data["Status"].apply(lambda d: 1 if d == 'Repaid' else 0 )
data["Default"].value_counts()


# In[13]:


#dropping 'DefaultDate' and 'Status' variables
#To avoid duplicates

irr_variables2 = ['Status', 'DefaultDate']
data.drop(irr_variables2, axis = 1, inplace = True)
data.shape


# In[14]:


#descrptive analysis
#for dependent variable

statistics.mode(data['Default'])


# In[15]:


max(data['Default'])


# In[16]:


min(data['Default'])


# In[17]:


plt.boxplot(data['Default'])
plt.title('Default')
plt.grid('True')
plt.show()


# In[18]:


#INDEPENDENT VARIABLES

statistics.mode(data['Age'])


# In[19]:


statistics.mean(data['Age'])


# In[20]:


max(data['Age'])


# In[21]:


min(data['Age'])


# In[22]:


plt.boxplot(data['Age'])
plt.title("AGE")
plt.grid('True')
plt.show()


# In[23]:


statistics.stdev(data['Age'])


# In[24]:


plt.hist(data['Age'], color = "r")
plt.show()


# In[25]:


#Gender
statistics.mode(data['Gender'])


# In[26]:


#Income
statistics.mode(data['IncomeTotal'])


# In[27]:


max(data['IncomeTotal'])


# In[28]:


min(data['IncomeTotal'])


# In[29]:


statistics.mean(data['IncomeTotal'])


# In[30]:


statistics.stdev(data['IncomeTotal'])


# In[31]:


plt.boxplot(data['IncomeTotal'])
plt.title('Income')
plt.grid('True')
plt.show()


# In[32]:


#Univariate Analysis
#DEFAULT# 0:Defaulted, 1:Did Not Default
#GENDER# 0:Male, 1: Female, 2:Unspecified
# EDUCATION# 1:primary education, 2:Basic Education, 3:Vocational Training, 4:Secondary Education
# 5: Higher Education.

plt.style.use('ggplot')
figure, ax1 = plt.subplots(1,3)
data['Default'].value_counts(normalize = True).plot(ax = ax1[0],figsize=(22, 7),kind = 
'bar',title = 'Default', color = "b", rot = 0)
data['Gender'].value_counts(normalize = True).plot(ax = ax1[1],kind = 
'bar',title = 'Gender', color = "g", rot = 0)
data['Education'].value_counts(normalize = True).plot(ax = ax1[2], kind = 'bar', title = 
'Education', color = "r", rot = 0)
figure.tight_layout()


# In[33]:


# MEWCREDITCUSTOMER# True: New Clients, False: Old clients
#EmploymentStatus# -1: unspecified, 1: employed, 2: partially employed
# 3:  unemployed, 4: self-employed, 5: Entreprenuer, 6: retiree.
#Marital Status# -1:unspecified, 1: married, 2:Cohabitant, 3: Single, 4: Divorced, 5: Widow


plt.style.use('ggplot')
figure, ax2 = plt.subplots(1,3)
data['EmploymentStatus'].value_counts(normalize = True).plot(ax = ax2[0],figsize = (22,7),
kind = 'bar',title = 
'EmploymentStatus', color = "r", rot = 0)
data['NewCreditCustomer'].value_counts(normalize = True).plot(ax = ax2[1], kind = 'bar',title = 
'NewCreditCustomer',rot = 0)
data['MaritalStatus'].value_counts(normalize = True).plot(ax = ax2[2], kind = 'bar',title = 
'MaritalStatus',rot = 0)
figure.tight_layout()


# In[34]:


#Bivariate Analysis
#Independent Variables vs. Dependent(Target Variable)
#EmploymentStatus vs. Default

sns.countplot(y = 'EmploymentStatus', hue = 'Default', data = data)


# In[35]:


#Gender vs. Default

sns.countplot(data = data, y = 'Gender', hue = 'Default')


# In[36]:


#MaritalStatus vs. Default
sns.countplot(data = data, y ='MaritalStatus', hue = 'Default')


# In[37]:


#Education vs. Default
sns.countplot(data= data, y = 'Education', hue = 'Default')


# In[38]:


#NewCreditCustomer vs. Default
sns.countplot(data = data, hue = 'Default', y = 'NewCreditCustomer')


# In[39]:


#Transforming categorical variables to numerical features using scikit learn or sklearn

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
varr = ['NewCreditCustomer', 'LoanApplicationStartedDate','ListedOnUTC',
        'PartyId','BiddingStartedOn','City']
for vari in varr:
    le = preprocessing.LabelEncoder()
    data[vari] = le.fit_transform(data[vari].astype('str'))

data.info()


# In[40]:


#observing outliers in different variables

data.boxplot(column = "IncomeTotal", color = 'blue')


# In[41]:


#exploring the same variable using a histogram
data['IncomeTotal'].hist(bins = 25, color = 'blue')


# In[42]:


data['IncomeTotal_log'] = np.log(data['IncomeTotal'])


# In[43]:


#Looking at amount and its outliers
data.boxplot(column = 'Amount', color = 'green')


# In[44]:


data['Amount_log'] = np.log(data['Amount'])
data['Amount_log'].hist(bins = 25, color = 'green')


# In[45]:


data.head()


# In[46]:


#VARIABLE DECLARATION

X = data.iloc[:, np.r_[7,14,20,22]].values
Y = data.iloc[:,33].values


# In[47]:


X


# In[48]:


Y


# In[49]:


#data splitting to train and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)


# In[50]:


#stndardizing the dataset to a set with mean 0 and variance 1

from sklearn.preprocessing import StandardScaler
da = StandardScaler()
X_train = da.fit_transform(X_train)
X_test = da.transform(X_test)


# In[51]:


Y_train


# In[52]:


Y_test.reshape


# In[53]:


Y_test


# In[54]:


# Model building

#NAIVE BAYES CLASSIFIER (NBGAUSSIAN)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
NBClassifier = GaussianNB()
NBClassifier.fit(X_train, Y_train)
Y_pred = NBClassifier.predict(X_test)
Y_pred
nb_prob = NBClassifier.predict_proba(X_test)
Confusion_Matrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix\n')
print(Confusion_Matrix)


print(classification_report(Y_test.reshape(-1, 1), Y_pred))

#sensitivity

sensitivity1 = Confusion_Matrix[0,0]/(Confusion_Matrix[0,1] + Confusion_Matrix[0,0])
print('Sensitivity is', sensitivity1)




# In[55]:


nb_prob


# In[56]:


#Calculating the AUC-ROC of NAIVE BAYES
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

nb_prob = nb_prob[:,1]


# In[57]:


nb_auc = roc_auc_score(Y_test, nb_prob)


# In[58]:


print('Naive Bayes:AUROC = % 0.3f' %(nb_auc))


# In[59]:


#Predicting the test set results
nb_fpr,nb_tpr , _ = roc_curve(Y_test, nb_prob)


# In[60]:


#plotting the AUC-ROC
plt.plot(nb_fpr, nb_tpr, marker = '.', label = "Naive Bayes( AUROC = % 0.3f)" % nb_auc, color = 'green')
plt.xlabel("False Positive Rate")
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[61]:


from sklearn.model_selection import cross_val_score


# In[62]:


print('Cross_val', cross_val_score(NBClassifier, Y_test.reshape(-1,1), Y_pred, cv = 10))
print('Cross_val', np.mean(cross_val_score(NBClassifier, Y_test.reshape(-1,1), Y_pred)))


# In[63]:


#K-NEAREST NEIGHBOR CLASSIFIER (KNN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
KNN = KNeighborsClassifier(n_neighbors = 5)
KNN.fit(X_train, Y_train)
Y_pred = KNN.predict(X_test)
Y_pred
kn_prob = KNN.predict_proba(X_test)
Confusion_Matrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix\n')
print(Confusion_Matrix)


print(classification_report(Y_test.reshape(-1, 1), Y_pred))

#sensitivity

sensitivity2 = Confusion_Matrix[0,0]/(Confusion_Matrix[0,1] + Confusion_Matrix[0,0])
print('Sensitivity is', sensitivity2)


# In[64]:


#Calculating the AUC-ROC of KNN
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

kn_prob = kn_prob[:,1]


# In[65]:


kn_auc = roc_auc_score(Y_test, kn_prob)


# In[66]:


print('K-Nearest Neighbor:AUROC = % 0.3f' %(kn_auc))


# In[67]:


#Predicting the test set results
kn_fpr,kn_tpr , _ = roc_curve(Y_test, kn_prob)


# In[68]:


#plotting the AUCROC
plt.plot(kn_fpr, kn_tpr, marker = '.', label = "K Nearest Neghbor (AUROC = % 0.3f)" %kn_auc)
plt.xlabel("False Positive Rate")
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[70]:


print('Cross_val', cross_val_score(KNN, Y_test.reshape(-1,1), Y_pred, cv = 10))
print('Cross_val', np.mean(cross_val_score(KNN, Y_test.reshape(-1,1), Y_pred)))


# In[ ]:


#SUPPORT VECTOR MACHINE (SVM)
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
svm.SVC(kernel = 'linear', gamma = 'auto', C = 2)
SVM = svm.SVC(kernel = 'linear', gamma = 'auto', C = 2)
SVM.fit(X_train, Y_train)
Y_pred = SVM.predict(X_test)
Y_pred
Confusion_Matrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix\n')
print(Confusion_Matrix)


print(classification_report(Y_test.reshape(-1, 1), Y_pred))

#sensitivity

sensitivity3 = Confusion_Matrix[0,0]/(Confusion_Matrix[0,1] + Confusion_Matrix[0,0])
print('Sensitivity is', sensitivity3)
 


# In[71]:


print('Cross_val', cross_val_score(SVM, Y_test.reshape(-1,1), Y_pred, cv = 10))
print('Cross_val', np.mean(cross_val_score(SVM, Y_test.reshape(-1,1), Y_pred)))


# In[ ]:


svc = svm.SVC(probability = True)
svc.fit(X_train, Y_train)


# In[ ]:


svc_prob = svc.predict_proba(X_test)


# In[71]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimator_list = [
    ("NBClassifier", NBClassifier),
    ("KNN", KNN)]

stack_model = StackingClassifier (
estimators = estimator_list, final_estimator = LogisticRegression())

stack_model.fit(X_train, Y_train)
Y_train_pred = stack_model.predict(X_train)
Y_test_pred = stack_model.predict(X_test)


# In[72]:


stack_model_train_accuracy = accuracy_score(Y_train, Y_train_pred)
stack_model_train_f1_score = f1_score(Y_train, Y_train_pred)

stack_model_test_accuracy = accuracy_score(Y_test, Y_test_pred)
stack_model_test_f1_score = f1_score(Y_test, Y_test_pred)


# In[73]:


stackprob = stack_model.predict_proba(X_test)


# In[74]:


stackprob


# In[75]:


stackprob = stackprob[:,1]


# In[76]:


stack_auc = roc_auc_score(Y_test, stackprob)


# In[86]:


print('Stacked_Model:AUROC = 0.721')


# In[80]:


stack_fpr,stack_tpr , _ = roc_curve(Y_test, stackprob)


# In[85]:


plt.plot(stack_fpr, stack_tpr, marker = '.', label = "Stacked Model (AUROC = 0.721)", color = 'black')
plt.xlabel("False Positive Rate")
plt.ylabel('True Positive Rate')
plt.title('Stacked_Model')
plt.legend()
plt.show()


# In[81]:


print('Model performance for Training set')
print('- Accuracy: %s'  %stack_model_train_accuracy)
print('- F1_score: %s'  %stack_model_train_f1_score)
print('--------------------------------------------')
print('Model performance for Test set')
print('- Accuracy: %s'  %stack_model_test_accuracy)
print('- F1_score: %s' %stack_model_test_f1_score)


# In[73]:


#classifications by algorithms.

classi = ['NBGaussian', 'KNN','SVM']
y1 = np.array([0.29, 0.26, 0.18])
y2 = np.array([0.27, 0.31, 0.37])

plt.bar(classi, y1, color = 'b')
plt.bar(classi, y2,bottom = y1, color = 'r')
plt.xlabel('Algorithm')
plt.ylabel('Classification Rate in %')
plt.legend(['True Positives', 'True Negatives'])
plt.title('Classification by Algorithms')
plt.show()


# In[74]:


#stacked bar plots of algorithm misclassifications.
classi2 = ['NBGaussian', 'KNN','SVM']
y1 = np.array([0.19, 0.22, 0.31])
y2 = np.array([0.25, 0.21, 0.14])

plt.bar(classi, y1, color = 'g')
plt.bar(classi, y2,bottom = y1, color = 'yellow')
plt.xlabel('Algorithm')
plt.ylabel('Misclassification Rate in %')
plt.legend(['False Positives', 'False Negatives'])
plt.title('Misclassification by Algorithms')
plt.show()


# In[75]:


x = ['NB', 'KNN', 'SVM']
y1 = [56, 57, 55]
y2 = [61,57, 55]
y3 = [54, 57, 57]
y4 = [57, 57, 54]
y5 = [56, 52, 68]

figure, axis = plt.subplots(2,3)
axis[0,0].bar(x, y1)
axis[0,0].set_title('Accuracy')


axis[0,1].bar(x, y2)
axis[0,1].set_title('Recall')

axis[0,2].bar(x, y3)
axis[0,2].set_title('Precision')

axis[1,0].bar(x, y4 )
axis[1,0].set_title('F1_score')

axis[1,1].bar(x, y5)
axis[1,1].set_title('Cross_Val')
plt.figsize = (10,11)
plt.tight_layout()
plt.show()


# In[ ]:




