#Import packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#Load the training data
df_train=pd.read_csv('train.csv')

# reading test data
df_test = pd.read_csv('test.csv')

# extracting and then removing the targets from the training data 
targets = df_train['Survived']
df_train.drop(['Survived'], 1, inplace=True)
    
# merging train data and test data for future feature engineering
# we'll also remove the PassengerID since this is not an informative feature
combined = df_train.append(df_test)
combined.reset_index(inplace=True)
combined.drop(['index', 'PassengerId'], inplace=True, axis=1)

#Now let's map the title can bin them
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

#Generate a new Title column
combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
combined['Title'] = combined['Title'].map(Title_Dictionary)

#let's get the median age based on people's gender, Pclass and Title
grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

#Now we just need to map these medium ages to the missing parts. [0] is to convert list to number.
def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) & 
        (grouped_median_train['Title'] == row['Title']) & 
        (grouped_median_train['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_train[condition]['Age'].values[0]

combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

# Name can be dropped now
combined.drop('Name', axis=1, inplace=True)

# encoding in dummy variable
titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
combined = pd.concat([combined, titles_dummies], axis=1)

# removing the title variable
combined.drop('Title', axis=1, inplace=True)

#Fill out the missing fare data
combined['Fare'].fillna(combined['Fare'].mean(), inplace=True)

# two missing embarked values - filling them with the most frequent one in the train set
combined['Embarked'].fillna('S', inplace=True)

# encoding in dummy variable
embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
combined = pd.concat([combined, embarked_dummies], axis=1)
combined.drop('Embarked', axis=1, inplace=True)

#Now let's fill out the missing values for Cabin
combined['Cabin'].fillna('M', inplace=True)
combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

# dummy encoding ...
cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    
combined = pd.concat([combined, cabin_dummies], axis=1)
combined.drop('Cabin', axis=1, inplace=True)

# encoding into 3 categories:
pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    
# adding dummy variable
combined = pd.concat([combined, pclass_dummies],axis=1)
    
# removing "Pclass"
combined.drop('Pclass',axis=1,inplace=True)

# mapping gender to numerical one 
combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})

#Previously we have explored the SibSp and Parch, now we will merge these two together
# introducing a new feature : the size of families (including the passenger)
combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
# introducing other features based on the family size
combined['Single'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

#a function that extracts each prefix of the ticket, returns 'NONE' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = [x for x in ticket if not x.isdigit()]
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'NONE'

#Get Ticket info
combined['Ticket'] = combined['Ticket'].map(cleanTicket)

# Extracting dummy variables from tickets:
tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
combined = pd.concat([combined, tickets_dummies], axis=1)
combined.drop('Ticket', inplace=True, axis=1)

#Prepare the training dataset
df_im_input=combined.iloc[:891]
df_im_output=targets

#Now let's get the importance of each feature
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(df_im_input, df_im_output)

features = pd.DataFrame()
features['feature'] = df_im_input.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)

#15 features
#select top 15 important features
top_15_feature=features.nlargest(15, 'importance')
df_input_final=df_im_input[top_15_feature['feature']]

#build Logistic Regression Model
rfc = RandomForestClassifier(criterion='gini', n_estimators=700,min_samples_split=10, min_samples_leaf=1,max_features='auto',oob_score=True,random_state=1,n_jobs=-1)
rfc.fit(df_input_final,targets) 

#get the test input and predictions
df_test_input_final=combined.iloc[891:][top_15_feature['feature']]
df_test_preds=rfc.predict(df_test_input_final)

#output the results to a csv file
submit = pd.DataFrame()
test = pd.read_csv('test.csv')
submit['PassengerId'] = test['PassengerId']
submit['Survived'] = df_test_preds
submit.to_csv('Titanic_rfc.csv', index=False)