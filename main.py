import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import io
from io import StringIO
from PIL import Image  
from sklearn.tree import export_graphviz
import pydotplus
import streamlit as st

# creation of some variable containers and columns to contain the parts of the webapplication
header = st.container()
dataset = st.container()
model_choice = st.selectbox("Choose a Model", ["Feature explanation","Decision Tree", "Gaussian NB", "SVC"])
features = st.container()
model_training_decision= st.container()
model_training_gaussian = st.container()
model_training_svc = st.container()
matrix, values = st.columns(2)

def get_data():
    # I read the csv file 
    heart_disease_df = pd.read_csv("processed.cleveland.csv")

# I create a function to replace the 5 different possible diagnosises with only 2
def create_groups(value):
    if value == 0:
        return 0
    else:
        return 1

# function to plot a confusion matrix in matplotlib and seaborn made with help of chatgpt by using the prompt
# "how do i plot a confusion matrix in a streamlit app"
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

with header:
    st.title('TaskML: Finn De Peuter')

with dataset:
    st.header('Heart disease dataset')
    st.write('I chose this dataset because I like to learn about medical topics. The dataset consists of multiple pieces but the site said the one most used for machine learning was the processed.cleveland data file.')
    st.write('This file already narrowed down the data to the things that are mostly used for analysis. It narrowed it down to 14 attributes instead of 76. It also structured the data into a readable format because I checked the other "unprocesed files" and these where not structured into csv format yet.')
    st.write("I first renamed the file to be a csv extension, but when I opened it, I noticed it didn't have a header row so the categories weren't specified. Luckily the site lists them in the right order so I could easily add the header myself in the csv file. I did choose to rename the last category to diagnosis, because it was called num on the site and I just found diagnosis an easier term. The type of machine learning I will be performing is classification.")

    heart_disease_df = pd.read_csv("processed.cleveland.csv")
    heart_disease_df['diagnosis'] = heart_disease_df['diagnosis'].apply(create_groups)

    # I use the mean of these columns to replace the null values
    ca = heart_disease_df['ca']
    ca_mean = ca.mean()

    thal = heart_disease_df['thal']
    thal_mean = thal.mean()

    ca.fillna(ca_mean, inplace=True)
    thal.fillna(thal_mean, inplace=True)

    # I separate the features and the results
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    y = heart_disease_df['diagnosis']
    X = heart_disease_df[feature_cols]

    # splitting the data into a 70% train, 30% test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

if model_choice == "Feature explanation":
    with features:
        st.markdown(
        f"""
            These attributes are pretty dificult to understand because they go over very technical aspects of heart related medical things. So I will try to explain them as easy as possible.

            Age, sex, resting blood pressure, cholestoral level, maximum heart rate achieved and fasting blood sugar are features that speak for themselves so I won't cover those.

            Chest pain type has 4 possible options:
            - typical angina = squeezing sensation in the chest, feelings of heaviness, pressure, tightness, burning and radiation to the shoulder and neck that typically lasts for 3 to 15 minutes
            - atypical angina = a sharp, prickling, choking pain that you feel in the chest wall, that is usually tender to palpation and lasts from minutes to hours to all day
            - non-anginal pain = squeezing pain / pressure behind the sternum that has another cause than an angina pectoris
            - asymptomatic = no pain

            The T-wave of an ECG is the wave that occurs right after the QRS complex and is the result of the ventricular repolarization. In easier words the phase in which the myocardial muscle (heart muscle) relaxes again. Abnormalities in this can mean that there is something wrong with the functioning of your hearts normal function.

            Resting electrocardiagraphic results:
            - normal = nothing unusual
            - having ST-T wave abnormality = abnormalities in the ST-T wave that can indicate heart disease
            - showing probable or definite left ventricular hypertrophy = an enormous increase in the myocardial mass on the left caused by too much workload on the heart usually caused by hypertension (high blood pressure)

            Exercise induced angina just means you get pain in the chest from exercising.

            The ST depression induced by exercise means that the ST part of the ECG is abnormally low (= depression). The slope of the peak lies in connection with this as it equals if there is something wrong. Normally it slightly concaves upwards. 
            - upsloping
            - flat
            - downsloping

            can all indicate indicate a coronary ischemia
            = the coronary arteries are narrowed and the heart is not getting enough blood and oxygen

            The number of major vessels colored by fluorosopy:
            A fluoroscopy is a way of taking videos of the movement in the body by using x-rays. This way the doctor can see if your blod flow is in order and there aren't any blockages.

            As I am unsure what exactly is meant by thal, I am unable to explain it but I think it might refer to having a heart birth defect as the values are normal, fixed defect and reversable defect.
            """
            )


if model_choice == "Decision Tree":
    with model_training_decision:
        st.header('Decision Tree')

        # decision tree preparation
        clf = DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(X_train, y_train)

        # predictions
        y_pred_baseline = clf.predict(X_test)

        # show the visual representation of the tree
        st.write("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred_baseline).sum()))
        
        st.write("I am unable to show the decision tree visually because the visgraph doesn't work here on streamlit.")
        with values:
            st.write("The accuracy equals:", metrics.accuracy_score(y_test, y_pred_baseline))
        with matrix:
            # confusion matrix
            true = np.array(y_test)
            predictions = np.array(y_pred_baseline)
            plot_confusion_matrix(true, predictions)

    st.write('The decision tree is the model I chose to be my baseline for this task. It scores pretty decent with an accuracy of 0.75 but I have to keep in mind that overfitting tends to happen in decision trees.')
    st.write('If we look at the confusion matrix to make it more visibly clear, we can see that it predicts wrongly around the same amount in the false positive and the false negatives. The main issue here is the wrongly predicted false negatives. This is the type of wrong predictions that can cause the most harm.')
    st.write("If a person gets told they don't have a presence of heart disease, but it's a false negative, this can have serious consequences. On the other hand the false positives aren't great either because diagnosing someone with something they don't have isn't good medical practice either.")
        

if model_choice == "Gaussian NB":
    with model_training_gaussian:
        st.header('Gaussian NB')

        # Gaussian NB
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)

        y_pred_gaus = gnb.predict(X_test)
        st.write("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred_gaus).sum()))

        with values:
            st.write("Accuracy:",gnb.score(X_test, y_test))
        with matrix:
            true = np.array(y_test)
            predictions_gnb = np.array(y_pred_gaus)
            plot_confusion_matrix(true, predictions_gnb)

    st.write("As we can see the GNB manages to perform way better than the other 2 models I tried. It manages to have to least wrong predictions. It performs the same if we look at the false negatives, but way better than the decision tree if we look at the false positives.")
    st.write("It has a higher accuracy score than both of the other two models.")


if model_choice == "SVC":
    with model_training_svc:
        st.header("SVC: Support Vector Classification")

        # SVC
        clf3 = svm.SVC()
        clf3 = clf3.fit(X_train, y_train)

        y_pred_svc = clf3.predict(X_test)
        st.write("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred_svc).sum()))

        with values:
            st.write("Accuracy:", clf3.score(X_test, y_test))

        with matrix:
            true = np.array(y_test)
            predictions_svc = np.array(y_pred_svc)
            plot_confusion_matrix(true, predictions_svc)

    st.write('The SVC performs the least well out of the 3 models I tried. It has an extremely high amount of false negatives which is absolutely not good in this case.')
    st.write("Now it is possible that it performs badly because I haven't really used customization features on the SVC model and stuck with the basic one. I honestly didn't understand most of the information on the scikit site so that's why I stuck with the basics.")
