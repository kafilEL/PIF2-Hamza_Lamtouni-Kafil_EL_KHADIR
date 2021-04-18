import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from PIL import Image
import streamlit as st
plt.style.use('fivethirtyeight')

maxUploadSize = 200

# Create a title and SubTitle
st.write("""
# Disease Detector
Detection de maladies PIF 2,\n
\nProjet des etudiants : Hamza Lamtouni, Kafil EL KHADIR
\nAu Professeur : Pr.Francois Meunier
""")
# Open and display an image
image = Image.open(r'data/1.png').convert('RGB')
st.image(image, caption='Machine Learning', use_column_width=True)

# DataSet Selection
selection = st.sidebar.selectbox('Selectionner le jeu de donnees', ('Breast cancer', 'Diabetes', 'Heart'))
st.write(selection)

# Classifier selection
classifier_name = st.sidebar.selectbox('Selectionner le modele de classification',
                                       ("KNN", "Random Forest","Logistic Regression","Naive Bayes(Gaussian)", "Decision Tree"))

# Get the data

if selection == "Diabetes":
    data = pd.read_csv(r'data/diabetes.csv')
    # Check for duplicates and erase them
    data.drop_duplicates(inplace=True)
    st.subheader('Information sur les donnees')
elif selection == 'Heart':
    data = pd.read_csv(r'data/cardio.csv', sep=";")
    data.drop_duplicates(inplace=True)
    data = data.iloc[:, 1:13]
    st.subheader('Information sur les donnees')
else:
    data = pd.read_csv(r'data/breast-cancer.csv')
    data = data.dropna(axis=1)
    data = data.iloc[:, 1:15]

    data['diagnosis'].replace(['M', 'B'], [1, 0], inplace=True)
    st.subheader('Information sur les donnees')



# SHow the data as a table

data_lenght = len(data)
head_number = (st.number_input('Combien de lignes vous voulez afficher ?', 1, data_lenght))
st.dataframe(data.head(head_number))
st.write(data.shape)


# Show some statistics
if st.checkbox('Statistiques descriptives'):
    st.subheader('Statistiques descriptives')
    st.write(data.describe())

# show corelation
st.set_option('deprecation.showPyplotGlobalUse', False)
if st.checkbox('Corellations en "%"'):
    st.subheader('Corellations')
    plt.figure(figsize=(10,10))
    st.write(sns.heatmap(data.iloc[:,0:12].corr(), annot=True, fmt='.0%'))
    st.pyplot()

# Show the data as a chart
if st.checkbox('Afficher la variable cible (y)'):
    # ff0000
    if selection == "Diabetes":
        st.subheader('Nombre de valeurs Y (Outcome Values)')
        sns.countplot(data['Outcome'],label='count',palette=['#00ff40','#ff0000'])
        st.pyplot(plt)
        st.write(data['Outcome'].value_counts())
    elif selection == "Heart":
        st.subheader('Nombre de valeurs Y (Cardio Values)')
        sns.countplot(data['cardio'], label='count',palette=['#00ff40','#ff0000'])
        st.pyplot(plt)
        st.write(data['cardio'].value_counts())
    elif selection == "Breast Cancer":
        st.subheader('Nombre de valeurs Y (Diagnosis Values)')
        sns.countplot(data['diagnosis'], label='count',palette=['#00ff40','#ff0000'])
        st.pyplot(plt)
        st.write(data['diagnosis'].value_counts())
    else :
        st.subheader('Nombre de valeurs Y (Diagnosis Values)')
        sns.countplot(data['diagnosis'], label='count',palette=['#00ff40','#ff0000'])
        st.pyplot(plt)
        st.write(data['diagnosis'].value_counts())

# Split the data into independant X and dependent y
if selection == "Diabetes":
    X = data.iloc[:, 0:8].values
    Y = data.iloc[:, -1].values
elif selection == "Breast cancer":
    X = data.iloc[:, 1:14].values
    Y = data.iloc[:, 0].values

elif selection == 'Heart':
    X = data.iloc[:, 0:11].values
    Y = data.iloc[:, -1].values

else:
    X = data.iloc[:, 0:5].values
    Y = data.iloc[:, -1].values



# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=4)




# Get the feature input from the user
def get_user_input():
    if selection == "Diabetes":
        pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
        glucose = st.sidebar.slider('glucose', 0, 199, 117)
        bloodPressure = st.sidebar.slider('bloodpressure', 0, 122, 72)
        skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
        insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
        bmi = st.sidebar.slider('bmi', 0.0, 67.1, 32.0)
        diabetes_pedigree_function = st.sidebar.slider('dpf', 0.078, 2.42, 0.3725)
        age = st.sidebar.slider('age', 21, 83, 29)

        # Store a dictionnary into a variable
        user_data = {'pregnancies': pregnancies,
                     'glucose': glucose,
                     'bloode_pressure': bloodPressure,
                     'skin_thickness': skin_thickness,
                     'insulin': insulin,
                     'bmi': bmi,
                     'dpf': diabetes_pedigree_function,
                     'age': age
                     }
    elif selection == "Heart":
        age = st.sidebar.slider('age', 10798, 23713, 19468)
        gender = st.sidebar.slider('gender', 0, 2, 1)
        height = st.sidebar.slider('height', 76, 188, 150)
        weight = st.sidebar.slider('weight', 41, 200, 120)
        ap_hi = st.sidebar.slider('ap_hi', 14, 200, 128)
        ap_lo = st.sidebar.slider('ap_lo', 30, 1100, 94)
        cholesterol = st.sidebar.slider('cholesterol', 1, 3, 2)
        gluc = st.sidebar.slider('gluc', 1, 2, 3)
        smoke = st.sidebar.slider('smoke', 0, 1, 0)
        alco = st.sidebar.slider('alco', 0, 1, 1)
        active = st.sidebar.slider('active', 0, 1, 0)

        # Store a dictionnary into a variable
        user_data = {'age': age,
                     'gender': gender,
                     'height': height,
                     'weight': weight,
                     'ap_hi': ap_hi,
                     'ap_lo': ap_lo,
                     'cholesterol': cholesterol,
                     'gluc': gluc,
                     'smoke': smoke,
                     'alco': alco,
                     'active': active
                     }
    elif selection == "Breast cancer":
        radius_mean = st.sidebar.slider('radius_mean', 6.9810, 28.1100, 14.1273)
        texture_mean = st.sidebar.slider('texture_mean', 9.7100, 39.2800, 19.2896)
        perimeter_mean = st.sidebar.slider('perimeter_mean', 43.7900, 188.5000, 91.9690)
        area_mean = st.sidebar.slider('area_mean', 143.5000, 2501.0, 654.8891)
        smoothness_mean = st.sidebar.slider('smoothness_mean', 0.0526, 0.1634, 0.0964)
        compactness_mean = st.sidebar.slider('compactness_mean', 0.0194, 0.3454, 0.1043)

        concavity_mean = st.sidebar.slider('concavity_mean', 0.0, 0.4268, 0.0888)
        concave_points_mean = st.sidebar.slider('concave_points_mean', 0.0, 0.2012, 0.0489)
        symmetry_mean = st.sidebar.slider('symmetry_mean', 0.1060, 0.3040, 0.1812)
        fractal_dimension_mean = st.sidebar.slider('fractal_dimension_mean', 0.0500, 0.0974, 0.0628)
        radius_se = st.sidebar.slider('radius_se', 0.1115, 2.8730, 0.4052)
        texture_se = st.sidebar.slider('texture_se', 0.3602, 4.8850, 1.2169)
        perimetre_se = st.sidebar.slider('perimetre_se', 0.7570, 21.9800, 2.8661)

        # Store a dictionnary into a variable
        user_data = {'radius_mean': radius_mean,
                     'texture_mean': texture_mean,
                     'perimeter_mean': perimeter_mean,
                     'area_mean': area_mean,
                     'smoothness_mean': smoothness_mean,
                     'compactness_mean': compactness_mean,
                     'concavity_mean': concavity_mean,
                     'concave_points_mean': concave_points_mean,
                     'symmetry_mean': symmetry_mean,
                     'fractal_dimension_mean': fractal_dimension_mean,
                     'radius_se': radius_se,
                     'texture_se': texture_se,
                     'perimetre_se': perimetre_se
                     }

    else:
        mean_radius = st.sidebar.slider('mean_radius', 0.0, 28.1100, 14.1273)
        mean_texture = st.sidebar.slider('mean_texture', 0.0, 39.2800, 19.2896)
        mean_perimeter = st.sidebar.slider('mean_perimeter', 0.0, 188.5000, 91.9690)
        mean_area = st.sidebar.slider('mean_area', 0.0, 2501.0, 654.8891)
        mean_smoothness = st.sidebar.slider('mean_smoothness', 0.0, 0.1634, 0.0964)

        # Store a dictionnary into a variable
        user_data = {'mean_radius': mean_radius,
                     'mean_texture': mean_texture,
                     'mean_perimeter': mean_perimeter,
                     'mean_area': mean_area,
                     'mean_smoothness': mean_smoothness
                     }

    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features


# Store the users input into a variable
user_input = get_user_input()

# Set a subeader and diplay the use input
st.subheader('Insertion utilisateur:')
st.write(user_input)

# Create and train the model
if classifier_name == "Random Forest":
    random_forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    random_forest.fit(x_train, y_train)

    # Show the models metrics
    st.subheader('Random Forest Model Test Accuracy score :')

    acc_rf = accuracy_score(y_test, random_forest.predict(x_test))

    st.write(str(accuracy_score(y_test, random_forest.predict(x_test)) * 100), "%")

    # Store the model prediction in a variable
    prediction = random_forest.predict(user_input)
    prediction_proba = random_forest.predict_proba(user_input)
    cm = confusion_matrix(y_test, random_forest.predict(x_test))
    st.subheader("Tester la perfomance du modele via la matrice de confusion")
    st.write(cm)
    tru_negatif = cm[0][0]
    tru_positif = cm[1][1]
    false_negatif = cm[1][0]
    false_positif = cm[0][1]
    st.write('Vraie Negatif de la matrice de confusion :', tru_negatif)
    st.write('Vraie positif de la matrice de confusion :', tru_positif)
    st.write('Faux négatif de la matrice de confusion :', false_negatif)
    st.write('Faux Positif de la matrice de confusion :', false_positif)
    st.success('La formule utilise pour le calcul du score : (vraie Negatif + vraie Positif) / (vraie Negatif + vraie Positif + faux Negatif + faux Positif) ')


    st.write('Model Test Accuracy via la matrice de confusion = {}'.format(
        (tru_positif + tru_negatif) / (tru_negatif + tru_positif + false_positif + false_negatif)))


elif classifier_name=="Naive Bayes(Gaussian)":
    nb = GaussianNB()
    nb.fit(x_train,y_train)
    st.subheader('Naive Bayes Gaussian Model Test Accuracy :')

    acc_nb = accuracy_score(y_test, nb.predict(x_test))

    st.write(str(accuracy_score(y_test,nb.predict(x_test)) * 100), "%")
    prediction = nb.predict(user_input)
    prediction_proba = nb.predict_proba(user_input)
    cm = confusion_matrix(y_test, nb.predict(x_test))
    st.subheader("Tester la perfomance du modele via la matrice de confusion")
    st.write(cm)
    tru_negatif = cm[0][0]
    tru_positif = cm[1][1]
    false_negatif = cm[1][0]
    false_positif = cm[0][1]
    st.write('Vraie Negatif de la matrice de confusion :', tru_negatif)
    st.write('Vraie positif de la matrice de confusion :', tru_positif)
    st.write('Faux négatif de la matrice de confusion :', false_negatif)
    st.write('Faux Positif de la matrice de confusion :', false_positif)
    st.success('La formule utilise pour le calcul du score : (vraie Negatif + vraie Positif) / (vraie Negatif + vraie Positif + faux Negatif + faux Positif) ')


    st.write('Model Test Accuracy via la matrice de confusion = {}'.format(
        (tru_positif + tru_negatif) / (tru_negatif + tru_positif + false_positif + false_negatif)))

elif classifier_name=="Logistic Regression":
    lr = LogisticRegression()
    lr.fit(x_train,y_train)
    st.subheader('Logistic Regression Model Test accuracy :')

    acc_lr = accuracy_score(y_test, lr.predict(x_test))

    st.write(str(accuracy_score(y_test,lr.predict(x_test)) * 100), "%")
    prediction = lr.predict(user_input)
    prediction_proba = lr.predict_proba(user_input)
    cm = confusion_matrix(y_test, lr.predict(x_test))
    st.subheader("Tester accuracy via la matrice de confusion")
    st.write(cm)
    tru_negatif = cm[0][0]
    tru_positif = cm[1][1]
    false_negatif = cm[1][0]
    false_positif = cm[0][1]
    st.write('Vraie Negatif de la matrice de confusion :', tru_negatif)
    st.write('Vraie positif de la matrice de confusion :', tru_positif)
    st.write('Faux négatif de la matrice de confusion :', false_negatif)
    st.write('Faux Positif de la matrice de confusion :', false_positif)
    st.success('La formule utilise pour le calcul du score : (vraie Negatif + vraie Positif) / (vraie Negatif + vraie Positif + faux Negatif + faux Positif) ')


    st.write('Model Test Accuracy via la matrice de confusion = {}'.format(
        (tru_positif + tru_negatif) / (tru_negatif + tru_positif + false_positif + false_negatif)))


elif classifier_name == "Decision Tree":
    dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
    dt.fit(x_train, y_train)
    st.subheader('Decision tree Model Test accuracy :')

    acc_dt = accuracy_score(y_test, dt.predict(x_test))

    st.write(str(accuracy_score(y_test, dt.predict(x_test)) * 100), "%")
    prediction = dt.predict(user_input)
    prediction_proba = dt.predict_proba(user_input)
    cm = confusion_matrix(y_test, dt.predict(x_test))
    st.subheader("Tester accuracy via la matrice de confusion")
    st.write(cm)
    tru_negatif = cm[0][0]
    tru_positif = cm[1][1]
    false_negatif = cm[1][0]
    false_positif = cm[0][1]
    st.write('Vraie Negatif de la matrice de confusion :', tru_negatif)
    st.write('Vraie positif de la matrice de confusion :', tru_positif)
    st.write('Faux négatif de la matrice de confusion :', false_negatif)
    st.write('Faux Positif de la matrice de confusion :', false_positif)
    st.success('La formule utilise pour le calcul du score : (vraie Negatif + vraie Positif) / (vraie Negatif + vraie Positif + faux Negatif + faux Positif) ')


    st.write('Model Test Accuracy via la matrice de confusion = {}'.format(
        (tru_positif + tru_negatif) / (tru_negatif + tru_positif + false_positif + false_negatif)))










else:
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(x_train, y_train)

    st.subheader('KNN Model Test accuracy :')
    acc_knn = accuracy_score(y_test, knn.predict(x_test))
    st.write(str(accuracy_score(y_test, knn.predict(x_test)) * 100), "%")

    cm = confusion_matrix(y_test,knn.predict(x_test))
    st.subheader("Matrice de confusion : ")
    st.write('Tester la perfomance du modele via la matrice de confusion')
    st.write(cm)
    tru_negatif = cm[0][0]
    tru_positif = cm[1][1]
    false_negatif = cm[1][0]
    false_positif = cm[0][1]
    st.write('Vraie Negatif de la matrice de confusion :', tru_negatif)
    st.write('Vraie positif de la matrice de confusion :', tru_positif)
    st.write('Faux négatif de la matrice de confusion :', false_negatif)
    st.write('Faux Positif de la matrice de confusion :', false_positif)
    st.success('La formule utilise pour le calcul du score : (vraie Negatif + vraie Positif) / (vraie Negatif + vraie Positif + faux Negatif + faux Positif) ')


    st.write('Model Test Accuracy via la matrice de confusion = {}'.format(
        (tru_positif + tru_negatif) / (tru_negatif + tru_positif + false_positif + false_negatif)))

    prediction = knn.predict(user_input)
    prediction_proba = knn.predict_proba(user_input)

# Set a subheader and display the classification
st.subheader('Probabilite en % :')
st.write(prediction_proba*100)

st.subheader('Classification :')
st.write(prediction)

if prediction == 1:
    st.warning('English : You probably have a disease. Recomanded : Consult your doctor\n '
               '\nFrancais : Vous avez probablement un risque de maladie , Consultez votre docteur')
else:
    st.success('English : You probably dont have anything .. Stay safe\n'
               '\nFrancais : Vous n"avez probablement aucune maladie ! ')


