from operator import truediv
import scipy
import codecs
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as stc
from sklearn import preprocessing
from sklearn import model_selection,neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def st_html(index_html):
    calc_file = codecs.open(index_html,'r')
    page = calc_file.read()
    stc.html(page,scrolling=False)
    

def model(place,river):
#     st.markdown("""
# <html>
# <body>

# <h1 style="color:blue;text-align:center;">This is a heading</h1>
# <p style="color:red;">This is a paragraph.</p>

# </body>
# </html>

# """,unsafe_allow_html=True )
    # newpath = "C:\\Users\\ajayp\\Documents\\Flood-Prediction-Model-master\\States\\"+place+".csv"
    # os.chdir(r'C:\Users\ajayp\Documents\Flood-Prediction-Model-master') # Path of our Project Folder
    # os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    data = pd.read_csv("//app"+"//States//"+place+".csv")
    isAllModelsSame = True
    flood = []
    limit = 2600
    if river == "YES":
        limit = 1400
    x1=list(data["ANNUAL"])
    for i in range(0,len(x1)):
        if x1[i] > limit:
            flood.append('YES')
        else:
            flood.append('NO')
    data["FLOOD"] = flood
    
    # st.write(data)
    # data.cov()
    # data.corr()
    data['FLOOD'].replace(['YES','NO'],[1,0],inplace=True)
    x=data.iloc[:,1:14]
    y=data.iloc[:,-1]
    st.write(data)
    st.header("KNN")
    # Scaling the data between 0 and 1.
    minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
    minmax.fit(x).transform(x)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')
    clf=neighbors.KNeighborsClassifier()
    clf.fit(x_train,y_train)
    st.write("Predicted Values for the Floods:")
    y_predict=clf.predict(x_test)
    st.write("Actual Values for the Floods:")
    st.write(y_test)
    st.write("List of the Predicted Values:")
    st.write(y_predict)

    x_train_std= minmax.fit_transform(x_train)
    x_test_std= minmax.fit_transform(x_test)
    knn_acc=cross_val_score(clf,x_train_std,y_train,cv=3,scoring='accuracy',n_jobs=-1)
    knn_proba=cross_val_predict(clf,x_train_std,y_train,cv=3,method='predict_proba')
    st.write("\nAccuracy Score:%f"%(accuracy_score(y_test,y_predict)*100))
    st.write("Recall Score:%f"%(recall_score(y_test,y_predict)*100))
    try:
        st.write("ROC score:%f"%(roc_auc_score(y_test,y_predict)*100)) 
    except ValueError:
        pass
    
    st.header("LOGISTIC REGRESSION")
    # #Logistic
    x_train_std=minmax.fit_transform(x_train)         # fit the values in between 0 and 1.
    y_train_std=minmax.transform(x_test)
    st.write(x_train)
    st.write(y_train)
    isFlood = False
    y1 = list(data["FLOOD"])
    for i in range(0,len(y1)):
        if y1[i]==1:
            isFlood = True
            break
    if isFlood == True:
        allOne = True
        for i in range(0,len(y1)):
            if y1[i]==0:
                allOne = False
                break
        if allOne==True:
            isFlood =False

    if isFlood == False:
        st.write("\nAccuracy score: 100.000000")
        st.write("recall score: 0.000000")
    else:
        isAllModelsSame = False
        lr=LogisticRegression()
        lr.fit(x_train,y_train)
        lr_acc=cross_val_score(lr,x_train_std,y_train,cv=3,scoring='accuracy',n_jobs=-1)
        lr_proba=cross_val_predict(lr,x_train_std,y_train,cv=3,method='predict_proba')
        y_pred=lr.predict(x_test)
        st.write("Actual Flood Values:")
        st.write(y_test.values)
        st.write("List of the Predicted Values:")
        st.write(y_pred)
        
        st.write("\naccuracy score: %f"%(accuracy_score(y_test,y_pred)*100))
        st.write("recall score: %f"%(recall_score(y_test,y_pred)*100))
        st.write("roc score: %f"%(roc_auc_score(y_test,y_pred)*100))

    #SVM
    st.header("SUPPORT VECTOR MACHINE")
    svc=SVC(kernel='rbf',probability=True)
    st.write(x_train)
    st.write(y_train)
    isFlood = False
    y1 = list(data["FLOOD"])
    for i in range(0,len(y1)):
        if y1[i]==1:
            isFlood = True
            break
    if isFlood == True:
        allOne = True
        for i in range(0,len(y1)):
            if y1[i]==0:
                allOne = False
                break
        if allOne==True:
            isFlood =False
    if isFlood == False:
        st.write("\nAccuracy score: 100.000000")
        st.write("recall score: 0.000000")
    else:
        isAllModelsSame = False
        svc_classifier=svc.fit(x_train,y_train)
        svc_acc=cross_val_score(svc_classifier,x_train_std,y_train,cv=3,scoring="accuracy",n_jobs=-1)
        svc_proba=cross_val_predict(svc_classifier,x_train_std,y_train,cv=3,method='predict_proba')
        svc_scores=svc_proba[:,1]
        y_pred=svc_classifier.predict(x_test)
        st.write("Actual Flood Values:")
        st.write(y_test.values)
        st.write("Predicted Flood Values:")
        st.write(y_pred)
        st.write("\naccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
        st.write("recall score:%f"%(recall_score(y_test,y_pred)*100))
        st.write("roc score:%f"%(roc_auc_score(y_test,y_pred)*100))
    
    #Decision Tree
    st.header("Decision Tree")
    y1 = list(data["FLOOD"])
    for i in range(0,len(y1)):
        if y1[i]==1:
            isFlood = True
            break
    if isFlood == True:
        allOne = True
        for i in range(0,len(y1)):
            if y1[i]==0:
                allOne = False
                break
        if allOne==True:
            isFlood =False
    if isFlood == False:
        st.write("\nAccuracy score: 100.000000")
        st.write("recall score: 0.000000")
    else:
        isAllModelsSame = False
        dtc_clf=DecisionTreeClassifier()
        dtc_clf.fit(x_train,y_train)
        dtc_clf_acc=cross_val_score(dtc_clf,x_train_std,y_train,cv=3,scoring="accuracy",n_jobs=-1)
        dtc_clf_acc
        st.write("Actual Values:")
        st.write(y_test.values)
        st.write("Predicted Values:")
        y_pred=dtc_clf.predict(x_test)
        st.write(y_pred)
        st.write("\naccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
        st.write("recall score:%f"%(recall_score(y_test,y_pred)*100))
        st.write("roc score:%f"%(roc_auc_score(y_test,y_pred)*100))

    #Random Forest
    st.header("Random Forest")
    y1 = list(data["FLOOD"])
    for i in range(0,len(y1)):
        if y1[i]==1:
            isFlood = True
            break
    if isFlood == True:
        allOne = True
        for i in range(0,len(y1)):
            if y1[i]==0:
                allOne = False
                break
        if allOne==True:
            isFlood =False
    if isFlood == False:
        st.write("\nAccuracy score: 100.000000")
        st.write("recall score: 0.000000")
    else:
        isAllModelsSame = False
        rmf=RandomForestClassifier(max_depth=3, random_state=0)
        rmf_clf=rmf.fit(x_train,y_train)
        rmf_clf
        rmf_clf_acc=cross_val_score(rmf_clf,x_train_std,y_train,cv=3,scoring="accuracy",n_jobs=-1)
        rmf_proba=cross_val_predict(rmf_clf,x_train_std,y_train,cv=3,method='predict_proba')
        rmf_clf_acc
        rmf_proba
        
        st.write("\naccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
        st.write("recall score:%f"%(recall_score(y_test,y_pred)*100))
        st.write("roc score:%f"%(roc_auc_score(y_test,y_pred)*100))

    st.header("Comparing all prediction Models")
    models = []
   
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('LR', LogisticRegression()))
    models.append(('SVC', SVC()))
    models.append(('DT', DecisionTreeClassifier()))
    models.append(('RF', RandomForestClassifier()))
    if isAllModelsSame==True:
        st.write("As there is only one possibility and every model predicts same accuracy, So in this case all models have same accuracy. Try different state to find out which is best.")
    else:
        names = []
        accuracy = []
        recall = []
        roc = []
        for name, model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy.append(accuracy_score(y_test, y_pred))
            recall.append(recall_score(y_test,y_pred))
            roc.append(roc_auc_score(y_test,y_pred))
            names.append(name)
        tr_split = pd.DataFrame({'Name': names, 'Accuracy': accuracy,'Recall':recall,'ROC':roc})
        st.write(tr_split)
        axis = sns.barplot(x = 'Name', y = 'ROC', data =tr_split )
        axis.set(xlabel='Classifier Models', ylabel='ROC Score of the Model')
        for p in axis.patches:
            height = p.get_height()
            axis.text(p.get_x() + p.get_width()/2, height + 0.01, '{:1.4f}'.format(height), ha="center") 
        st.pyplot(plt)
        st.write("The maximum ROC value is", tr_split['ROC'].max())

def main():
   #st_html('index.html')
    st.title("Flood prediction using Machine Learning")
    abc = st.selectbox('Select a State',('Tamilnadu','arunanchal','assam','himachal pradesh','jammu kashmir','jharkhand','punjab','orissa','bihar','telangana','haryana delhi','west bengal','kerala','andaman','uttarkhand','saurashtra region','south interior karnatka'))   
    # ab = st.number_input("Enter rainfall average from march to may") # present years march to may rainfall data on average
    # cd = st.number_input("Average rainfall in past 10 days") #average rainfall in past 10 days of june
    # ef = st.number_input(" Average increase in rainfall from may to june") #average inscrease in rainfall from may to june
    river = st.selectbox('Select wether the given state has large river basin or dams',('YES','NO'))
    if st.button("Submit"):
        model(abc,river)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    main()
