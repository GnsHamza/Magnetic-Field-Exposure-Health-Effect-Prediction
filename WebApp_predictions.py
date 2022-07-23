import streamlit as st
import pandas as pd
import pickle


st.write("# Application qui prévoit l'effet d'une quelconque exposition à un champ magnétique")

# Collecter le profil d'entrée
st.sidebar.header("Les caractéristiques de l'exposition")


def caracteristiques_exposition():
    Fréquence = st.sidebar.selectbox("Fréquence", ("Hyperfréquences", "EBF"))
    Induction_magnétique = st.sidebar.slider("Induction magnétique en μT", 0.01, 10.0, 1.0)
    Durée_exposition = st.sidebar.selectbox("Durée d'exposition", ("longue", "modérée", "courte"))
    Type_exposition = st.sidebar.selectbox("Type d'exposition", ("aiguë", "chronique"))
    Distance = st.sidebar.selectbox("Distance", ("grande", "intermédiaire", "petite"))
    Âge = st.sidebar.selectbox("Âge", ("âgé", "adulte", "jeune"))
    Sexe = st.sidebar.selectbox("Sexe", ("masculin", "féminin"))

    data = {
        'Fréquence': Fréquence,
        'Induction_magnétique': Induction_magnétique,
        'Durée_exposition': Durée_exposition,
        'Type_exposition':  Type_exposition,
        'Distance': Distance,
        'Âge': Âge,
        'Sexe': Sexe
    }

    exposition = pd.DataFrame(data, index=[0])
    return exposition


input_df = caracteristiques_exposition()


# Transformer les données d'entrée en données adaptées au model
# Importer la base de données
df = pd.read_csv("MF exposure - data.csv")
caractéristiques_entrées = df.drop(columns=["Effet"])
données_entrées = pd.concat([input_df, caractéristiques_entrées], axis=0)


# Préparation de données (encodage)
variables_categorielles = ["Fréquence", "Durée_exposition", "Type_exposition", "Distance", "Âge", "Sexe"]
for colonne in variables_categorielles:
    dummy = pd.get_dummies(données_entrées[colonne], drop_first=True)
    données_entrées=pd.concat([dummy, données_entrées], axis=1)
    del données_entrées[colonne]


# Prendre uniquement la première ligne
données_entrées = données_entrées[:1]


# Afficher les données transformées
st.subheader("Variables d'entrées encodés")
st.write(données_entrées)

# Importer le model
load_model = pickle.load(open('prévoir_exposition.pkl', 'rb'))

# Appliquer le model aux caractéristiques d'entrée
predictions = load_model.predict(données_entrées)
st.subheader('Résultat de la prévision')
st.write(predictions)

