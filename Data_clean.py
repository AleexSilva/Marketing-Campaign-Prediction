import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Functions

def nullValues(dataSet):
    ''' Analize null values '''
    x = dataSet.isna().sum()
    if len(x[x>0])>0:
        d = {'NullRecord': x[x>0], 'TotalRecord': dataSet.shape[0]}
        y = pd.DataFrame(d)
        #y["CompleteRecord"] = y["TotalRecord"] - y["NullRecord"]
        y["Empty %"] = round(y['NullRecord']/y['TotalRecord'],2) 
    return y.sort_values("NullRecord",ascending=True)


def Zerovalues(df):
    d = {'TotalRecord': df.shape[0], '#Zeros':0, '%Zeros':0}
    y = pd.DataFrame(d, index= df.columns)
     
    for c  in df.columns:
         y.loc[c,"#Zeros"] = len(df[df[c]==0])
         y.loc[c,"%Zeros"] = round((y.loc[c,"#Zeros"] / y.loc[c,"TotalRecord"] ),2) * 100
    return y[y["#Zeros"]>0]


#Print indice de unicidad

def uniqueRate(data,initialColumn):
    
  d = pd.DataFrame(columns={"UniqueValue","UniqueRate"},index=data.columns)
  x = data.columns

  for i in data.columns:
    d.loc[i,"UniqueValue"]=data[i].nunique()
    #print(x[i])

  d["UniqueRate"]= d["UniqueValue"] / data.shape[0]
  d["UniqueRate"] = d["UniqueRate"].astype(float)
  d["UniqueRate"] = round(d["UniqueRate"],3)
  return d.sort_values(by="UniqueValue",ascending=False)

# Pivot table with categorical data

def pivot_table(data,columns,count):

    for i in columns:
        d = data.pivot_table(index= i, values=count, margins=True,margin_name='Total',aaggfunc='count')
        d['%']=d/data.shape[0]
        d['%']=round(d['%'].astype(float),2)
        print(d.sort_values(by=count,ascending=True).reset_index())
        print('\n')

def neg(data,variables):
    for i in variables:
        n=data[i]<0
        if n.any() == True:
            print(f'The column {i} has {n.shape[0]} negative values.')


# Function created to know measures of central tendency

def MedMedMaxMin(df,column,bins):
    
  print(pd.DataFrame({"Mean":df[column].mean(),
                "Median":df[column].median(),
                "Min":df[column].min(),
                "Max":df[column].max()
                },index={column}
              ),'\n')

  print(df[column].describe().transpose(),'\n')

  print (pd.cut(df[column], bins=bins))

  f, axes = plt.subplots(2, 2, figsize=(11, 11), sharex=False)

  sns.distplot(df[column],kde=False,bins=bins,ax=axes[0, 0])
  sns.boxplot(x=df[column],ax=axes[0, 1])


  return (f, axes)

# Functions to create Plots

def Boxplot(df,lists):
    for i in lists:
        if(df[i].dtype !="object"):
            plt.figure(figsize=(5,5))
            plt.title(i)
            sns.boxplot(x=df[i])



def Analisis_crosstab_target_vs(data,target,value,param):
      
  if (param==1):
    plt.figure(figsize=(20,5))
    table=pd.crosstab(data[value],data[target])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
  else:
    table=pd.crosstab(data[value],data[target])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

    sns.catplot(x=value, 
            #hue="who", 
            col=target,
            data=data, 
            kind="count",
            height=6, 
            aspect=.7);

    plt.figure(figsize=(10,5))
    sns.countplot(x = value,
              data = data)

    plt.figure(figsize=(10,5))
    sns.countplot(x = value,
              data = data[data[target]==1])

    plt.figure(figsize=(10,5))
    sns.countplot(x = value,
              data = data[data[target]==0])

    print('Group sum every Terms in Data')
    print(data.groupby(value)['target'].value_counts())

def catPlotCount(df,feature,target,sg):
    if(sg==False):
        print(df.groupby(feature)[target].value_counts())
        print(df.groupby(feature)[target].value_counts(normalize=True))

        sns.catplot(x=feature, 
                    #hue="who", 
                    col=target,
                    data=df, 
                    kind="count",
                    height=4, 
                    aspect=.7);

        #sns.set(color_codes=True)

        x,y = feature, target

        print("")

        (df
        .groupby(x)[y]
        .value_counts(normalize=True)
        .mul(100)
        .rename('percent')
        .reset_index()
        .pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))     
    else:
        sns.catplot(x=feature, 
                #hue="who", 
                col=target,
                data=df, 
                kind="count",
                height=4, 
                aspect=.7);

    #sns.set(color_codes=True)

        x,y = feature, target

        print("")

        (df
        .groupby(x)[y]
        .value_counts(normalize=True)
        .mul(100)
        .rename('percent')
        .reset_index()
        .pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))  


#Function to set and delimite our model

def Arbol (lablesColumns,targetColumn ,dataFrame, maxDep):
    from sklearn.metrics import confusion_matrix
    X = dataFrame[lablesColumns] # Features

    y = dataFrame[targetColumn] # Target variable

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y, test_size=0.30) # 70% training and 30% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=maxDep)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train_t,y_train_t)

    #Predict the response for test dataset
    y_pred_t = clf.predict(X_test_t)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test_t, y_pred_t))

    print("Recall:",metrics.recall_score(y_test_t, y_pred_t))
    print("precision_score:",metrics.precision_score(y_test_t, y_pred_t))
    print("jaccard_score:",metrics.jaccard_score(y_test_t, y_pred_t))
  
    print("/n")
    from sklearn.metrics import classification_report
    print(classification_report(y_test_t, y_pred_t))

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                  filled=True, rounded=True,
                  special_characters=True, feature_names = feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('churn.png')
  
  
    cm_t = confusion_matrix(y_test_t, y_pred_t)

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(2,2))
        ax = sns.heatmap(cm_t, 
                        square=True,
                        cmap="Blues",
                        annot=True,
                        fmt="d",
                        cbar=False)
        #ax = sns.heatmap(c, mask=mask, vmax=.3, square=True,cmap="Blues",annot=True,cbar=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
        #ax.set_xticklabels({"negative","positive"}, rotation=0, horizontalalignment='right')
        #ax.set_yticklabels({"negative","positive"}, rotation=0, horizontalalignment='right')

    plt.show()
  
    print("\n")
  
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_test_t, clf.predict(X_test_t))
    fpr, tpr, thresholds = roc_curve(y_test_t, clf.predict_proba(X_test_t)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

    print("\n")

    return graph.create_png()
    #Image(graph.create_png())


def myRandomForest(data,features,target,maxFeat,numEst):
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    X=data[features]  # Features
    y=data[target]  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

    #Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=numEst, criterion ='entropy',max_features = maxFeat )

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    print("/n")
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data,  
    #                 filled=True, rounded=True,
    #                 special_characters=True, feature_names = features,class_names=['0','1'])
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # graph.write_png('churn.png')
  
  
    cm_t = confusion_matrix(y_test, y_pred)

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(2,2))
        ax = sns.heatmap(cm_t, 
                      square=True,
                      cmap="Blues",
                      annot=True,
                      fmt="d",
                      cbar=False)
        #ax = sns.heatmap(c, mask=mask, vmax=.3, square=True,cmap="Blues",annot=True,cbar=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
        #ax.set_xticklabels({"negative","positive"}, rotation=0, horizontalalignment='right')
        #ax.set_yticklabels({"negative","positive"}, rotation=0, horizontalalignment='right')

        # plt.show()
  
    print("\n")
  
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

    print("\n")

    # return graph.create_png()
    #Image(graph.create_png())


def FindingImportantFeatures(data,features,target):
    from sklearn.model_selection import train_test_split

    X=data[features]  # Features
    X.drop(columns=target,axis=1,inplace=True)
    y=data[target]  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

    from sklearn.ensemble import RandomForestClassifier

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)

    from imblearn.combine import SMOTETomek

    # smt = SMOTETomek(ratio='auto')
    # X_res, y_res = smt.fit_sample(X_train, y_train)

    # X_train = X_res
    # y_train = y_res

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    import pandas as pd
    feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
    print(feature_imp)

    import matplotlib.pyplot as plt
    import seaborn as sns
    #%matplotlib inline
    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()

def myLogReg (regLog_columns,targetColName,data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split


    X = data[regLog_columns]
    y = data[targetColName]

    from imblearn.over_sampling import SMOTE

    #os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.fit_transform(X_test)

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)

    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


def myLogReg_SmoteTomek (regLog_columns,targetColName,data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split


    X = data[regLog_columns]
    y = data[targetColName]

    from imblearn.over_sampling import SMOTE

    #os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.fit_transform(X_test)

        # Split dataset into training set and test set
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts) # 70% training and 30% test

    #print("Data X sin balancear: ",len(X))
    #print("Data y[0] sin balancear: ",len(y[y==0]))
    #print("Data y[1] sin balancear: ",len(y[y==1]))

    #print("Data train sin balancear: ",len(X_train))
    #print("Data test sin balancear: ",len(X_test))

    from imblearn.combine import SMOTETomek

    smt = SMOTETomek(ratio='auto')
    X_res, y_res = smt.fit_sample(X_train, y_train)

    X_train = X_res
    y_train = y_res

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)

    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
        
    # cm_t = confusion_matrix(y_test, y_pred)

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(2,2))
        ax = sns.heatmap(confusion_matrix, 
                        square=True,
                        cmap="Blues",
                        annot=True,
                        fmt="d",
                        cbar=False)
        #ax = sns.heatmap(c, mask=mask, vmax=.3, square=True,cmap="Blues",annot=True,cbar=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
        #ax.set_xticklabels({"negative","positive"}, rotation=0, horizontalalignment='right')
        #ax.set_yticklabels({"negative","positive"}, rotation=0, horizontalalignment='right')

        # plt.show()

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


# Import train_test_split function

def myRandomForest_SMOTETomek(data,features,target,maxFeat,numEst, ts):
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import confusion_matrix
  
  from imblearn.under_sampling import NearMiss
  from imblearn.over_sampling import RandomOverSampler
  from imblearn.combine import SMOTETomek
  from imblearn.ensemble import BalancedBaggingClassifier

  X=data[features]  # Features
  y=data[target]  # Labels

  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts)

  #print("Data X sin balancear: ",len(X))
  #print("Data y[0] sin balancear: ",len(y[y==0]))
  #print("Data y[1] sin balancear: ",len(y[y==1]))

  #print("Data train sin balancear: ",len(X_train))
  #print("Data test sin balancear: ",len(X_test))



  #print("Data X balancear: ",len(X_res))
  #print("Data y[0] balancear: ",len(y_res[y_res==0]))
  #print("Data y[1] balancear: ",len(y_res[y_res==1]))

  # print ("Distribution before resampling {}".format(Counter(y_train)))
  # print ("Distribution labels after resampling {}".format(Counter(y_train_res)))
  

  # Split dataset into training set and test set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts) # 70% training and 30% test

  #print("Data X sin balancear: ",len(X))
  #print("Data y[0] sin balancear: ",len(y[y==0]))
  #print("Data y[1] sin balancear: ",len(y[y==1]))

  #print("Data train sin balancear: ",len(X_train))
  #print("Data test sin balancear: ",len(X_test))

  from imblearn.combine import SMOTETomek

  smt = SMOTETomek(ratio='auto')
  X_res, y_res = smt.fit_sample(X_train, y_train)

  #print("Data train  balancear: ",len(X_res))
  #print("Data test  balancear: ",len(y_res))
  
  # from imblearn.over_sampling import SMOTE

  # smote = SMOTE(ratio='minority')
  # X_train_res, y_train_res = smote.fit_sample(X, y)



  # os =  RandomOverSampler(ratio=0.5)
  # X_train_res, y_train_res = os.fit_sample(X_train, y_train)

  #Import Random Forest Model
  from sklearn.ensemble import RandomForestClassifier

  #Create a Gaussian Classifier
  clf=RandomForestClassifier(n_estimators=numEst, criterion ='entropy',max_features = maxFeat,class_weight="balanced" )

  #Train the model using the training sets y_pred=clf.predict(X_test)
  clf.fit(X_res,y_res)

  y_pred=clf.predict(X_test)

  from sklearn import metrics
  # Model Accuracy, how often is the classifier correct?
  print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

  print("/n")
  from sklearn.metrics import classification_report
  print(classification_report(y_test, y_pred))

  # dot_data = StringIO()
  # export_graphviz(clf, out_file=dot_data,  
  #                 filled=True, rounded=True,
  #                 special_characters=True, feature_names = features,class_names=['0','1'])
  # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
  # graph.write_png('churn.png')
  
  
  cm_t = confusion_matrix(y_test, y_pred)

  with sns.axes_style("white"):
      f, ax = plt.subplots(figsize=(2,2))
      ax = sns.heatmap(cm_t, 
                      square=True,
                      cmap="Blues",
                      annot=True,
                      fmt="d",
                      cbar=False)
      #ax = sns.heatmap(c, mask=mask, vmax=.3, square=True,cmap="Blues",annot=True,cbar=False)
      ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
      ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
      #ax.set_xticklabels({"negative","positive"}, rotation=0, horizontalalignment='right')
      #ax.set_yticklabels({"negative","positive"}, rotation=0, horizontalalignment='right')

    # plt.show()
  
  print("\n")
  
  from sklearn.metrics import roc_auc_score
  from sklearn.metrics import roc_curve
  logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
  fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
  plt.figure()
  plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.savefig('Log_ROC')
  plt.show()

  print("\n")

  # return graph.create_png()
  #Image(graph.create_png())



