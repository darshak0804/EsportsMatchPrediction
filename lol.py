import pandas as pd
import numpy as np

# path_to_dataset = input("Enter path to dataset: ")
path_to_dataset = "./lol.csv"
# ./lol.csv
dataset = pd.read_csv(path_to_dataset)

cols = ['gameId', 'redFirstBlood', 'redKills', 'redEliteMonsters', 
        'redDragons','redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 
        'redGoldDiff', 'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin', 
        'redHeralds', 'blueGoldDiff', 'blueExperienceDiff', 'blueCSPerMin', 
        'blueGoldPerMin', 'blueTotalMinionsKilled']

clean_dataset = dataset.drop(cols, axis=1)

corr_list = clean_dataset[clean_dataset.columns[1:]].apply(
    lambda x: x.corr(clean_dataset['blueWins']))
cols = []
for col in corr_list.index:
    if (corr_list[col]>0.2 or corr_list[col]<-0.2):
        cols.append(col)
        
df = clean_dataset[cols]

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = df
y = dataset['blueWins']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

choice = input("Choose the algorithm from the following:\n1)Logistic Regression\n2)Knn\n3)SVM\n4)Naive Bayes\n5)Decision Tree\n6)Random Forest\nYour Choice:\n")

if choice == '1':
    print("\nAlgorithm: Logistic Regression")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    iterations = int(input("Enter number of iterations: "))
    score=[]
    for i in range(iterations): #10000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        


        # grid = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}
        # clf_rf = GridSearchCV(rf, grid, cv=5)
        # clf_rf.fit(X_train, y_train)

        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        score.append(accuracy_score(y_test,y_pred))
        # print(accuracy_score(y_test, y_pred))
    print("Prediction Accuracy: ",sum(score)/len(score))

elif choice == '2':
    print("\nAlgorithm: K Nearset Neighbors")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    iterations = int(input("Enter number of iterations: "))
    neighbors = int(input("Enter number of neighbors: "))
    score=[]
    for i in range(iterations): #10000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        


        # grid = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}
        # clf_rf = GridSearchCV(rf, grid, cv=5)
        # clf_rf.fit(X_train, y_train)

        classifier = KNeighborsClassifier(n_neighbors = neighbors )
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        score.append(accuracy_score(y_test,y_pred))
        # print(accuracy_score(y_test, y_pred))
    print("Prediction Accuracy: ", sum(score)/len(score))

elif choice == '3':
    print("\nAlgorithm: Support Vector Machine")
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    iterations = int(input("Enter number of iterations: "))
    kern = input("Choose the kernel\n1)linear\n2)rbf\n3)poly\nYour Choice:")
    score=[]
    for i in range(iterations): #10000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        


        # grid = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}
        # clf_rf = GridSearchCV(rf, grid, cv=5)
        # clf_rf.fit(X_train, y_train)

        classifier = SVC( kernel = kern )
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        score.append(accuracy_score(y_test,y_pred))
        # print(accuracy_score(y_test, y_pred))
    print("Prediction Accuracy: ", sum(score)/len(score))

elif choice == '4':
    print("\nAlgorithm: Naive Bayes")
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    iterations = int(input("Enter number of iterations: "))
    score=[]
    for i in range(iterations): #10000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        


        # grid = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}
        # clf_rf = GridSearchCV(rf, grid, cv=5)
        # clf_rf.fit(X_train, y_train)

        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        score.append(accuracy_score(y_test,y_pred))
        # print(accuracy_score(y_test, y_pred))
    print("Prediction Accuracy: ", sum(score)/len(score))

elif choice == '5':
    print("\nAlgorithm: Decision Tree")
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    iterations = int(input("Enter number of iterations: "))
    score=[]
    for i in range(iterations): #10000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        


        # grid = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}
        # clf_rf = GridSearchCV(rf, grid, cv=5)
        # clf_rf.fit(X_train, y_train)

        classifier = DecisionTreeClassifier(criterion='entropy')
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        score.append(accuracy_score(y_test,y_pred))
        # print(accuracy_score(y_test, y_pred))
    print("Prediction Accuracy: ", sum(score)/len(score))

elif choice == '6':
    print("\nAlgorithm: Random Forest")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    iterations = int(input("Enter number of iterations: "))
    no_of_tress =  int(input("Enter number of trees: "))
    score=[]
    for i in range(iterations): #10000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        


        # grid = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}
        # clf_rf = GridSearchCV(rf, grid, cv=5)
        # clf_rf.fit(X_train, y_train)

        classifier = RandomForestClassifier(n_estimators=no_of_tress, criterion='entropy')
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        score.append(accuracy_score(y_test,y_pred))
        # print(accuracy_score(y_test, y_pred))
    print("Prediction Accuracy: ", sum(score)/len(score))