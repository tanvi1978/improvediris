# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score_svc = svc_model.score(X_train, y_train)

lr = LogisticRegression(n_jobs=-1)
lr.fit(X_train, y_train)
score_lr = lr.score(X_train, y_train)


rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)
rfc.fit(X_train, y_train)
score_rfc = rfc.score(X_train, y_train)

d = {"SVC": svc_model, "Logistic Regression": lr, "Random Forest Classifier": rfc}

@st.cache()
def prediction(SepalLength, SepalWidth, PetalLength, PetalWidth, model):
  species = d[model].predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"


# Add title widget
st.title("Iris Flower Species Prediction App")  

# Add 4 sliders and store the value returned by them in 4 separate variables.
s_length = st.sidebar.slider("Sepal Length", 0.0, 10.0)
s_width = st.sidebar.slider("Sepal Width", 0.0, 10.0)
p_length = st.sidebar.slider("Petal Length", 0.0, 10.0)
p_width = st.sidebar.slider("Petal Width", 0.0, 10.0)

model = st.sidebar.selectbox("Classifier", ("SVC", "Logistic Regression", "Random Forest Classifier"))

# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'. 
# Print the value of 'species_type' and 'score' variable using the 'st.write()' function.
if st.sidebar.button("Predict"):


	species_type = prediction(s_length, s_width, p_length, p_width, model)
	st.write("Species predicted:", species_type)
	st.write("Accuracy score of this model is:", d[model].score(X_train, y_train))


    