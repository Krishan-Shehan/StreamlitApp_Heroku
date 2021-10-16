import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


def app():

    data = pd.read_csv("Preprocessed_data.csv")

    classifier_name = st.sidebar.selectbox(
        "Select Classifier",
        (
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "K-Neighbors Classifier",
            "Gaussian Naives Bayes",
            "Neural-Network Classifier",
            "Voting Classifier",
        ),
    )

    with st.sidebar.form(key="my_form"):
        budget = st.number_input("Enter your budget")
        if budget < 0:
            st.error("Enter a valid budget")
        elif budget > 3000000:
            st.error("Maximum budget is 3000000")

        sqfeet = st.number_input("Enter your sqfeet")
        if sqfeet < 0:
            st.error("Enter a valid property area")
        elif sqfeet > 20000:
            st.error("Maximum area allowed is 20000 square feet")

        beds = st.text_input("Enter preffered number of bedrooms")
        baths = st.text_input("Enter preffered number of bathrooms")
        smoking = st.radio("Smoking allowed", ["Yes", "No"])
        wheelchair = st.radio("Wheelchair access", ["Yes", "No"])
        vehicle = st.radio("Electric vehicle charge access", ["Yes", "No"])
        funrnished = st.radio("Furnished", ["Yes", "No"])
        laundry = st.selectbox(
            "Select laundry option",
            (
                "Laundry on site",
                "Laundry in building",
                "W/D in unit",
                "W/D hookups",
                "No laundry on site",
            ),
        )
        parking = st.selectbox(
            "Select parking options",
            (
                "Carport",
                "Street parking",
                "Attached garage",
                "Off-street parking",
                "Detached garage",
                "No parking",
                "Valet parking",
            ),
        )
        state = st.text_input("Enter your state code")
        submit = st.form_submit_button(label="Predict")

    if parking == "Carport":
        parking = 4
    elif parking == "Street parking":
        parking = 1
    elif parking == "Attached garage":
        parking = 0
    elif parking == "Off-street parking":
        parking = 2
    elif parking == "Detached garage":
        parking = 5
    elif parking == "No parking":
        parking = 3
    elif parking == "Valet parking":
        parking = 6

    if laundry == "Laundry on site":
        laundry = 3
    elif laundry == "Laundry in bldg":
        laundry = 4
    elif laundry == "W/D in unit":
        laundry = 0
    elif laundry == "W/D hookups":
        laundry = 2
    else:
        laundry = 1

    smoking = 1 if smoking == "Yes" else 0
    wheelchair = 1 if wheelchair == "Yes" else 0
    vehicle = 1 if vehicle == "Yes" else 0
    funrnished = 1 if funrnished == "Yes" else 0

    def get_classifier(clf_name):
        clf = None
        if clf_name == "Decision Tree Classifier":
            clf = DecisionTreeClassifier(random_state=100)
        elif clf_name == "Random Forest Classifier":
            clf = RandomForestClassifier()
        elif clf_name == "K-Neighbors Classifier":
            clf = KNeighborsClassifier(n_neighbors=3)
        elif clf_name == "Gaussian Naives Bayes":
            clf = GaussianNB()
        elif clf_name == "Neural-Network Classifier":
            clf = MLPClassifier()
        else:
            log_clf = LogisticRegression()
            rnd_clf = RandomForestClassifier()
            knn_clf = KNeighborsClassifier()
            clf = VotingClassifier(
                estimators=[("lr", log_clf), ("rnd", rnd_clf),
                            ("knn", knn_clf)],
                voting="hard",
            )
        return clf

    clf = get_classifier(classifier_name)

    def classify_model(data, model):
        X = data.drop(columns=["type", "pets_allowed"])
        Y = data.values[:, 1] if model == 1 else data.values[:, 12]
        return X, Y

    #### CLASSIFICATION ####

    def multilableclasification_specific(
        budget,
        sqfeet,
        beds,
        baths,
        smoking,
        wheelchair,
        vehicle,
        funrnished,
        laundry,
        parking,
        state,
    ):
        X, Y = classify_model(data, 1)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.25, random_state=1
        )
        clf.fit(X_train, Y_train)
        return clf.predict(
            [
                [
                    budget,
                    sqfeet,
                    beds,
                    baths,
                    smoking,
                    wheelchair,
                    vehicle,
                    funrnished,
                    laundry,
                    parking,
                    state,
                ]
            ]
        )

    def bianryclasification_specific(
        budget,
        sqfeet,
        beds,
        baths,
        smoking,
        wheelchair,
        vehicle,
        funrnished,
        laundry,
        parking,
        state,
    ):
        X, Y = classify_model(data, 0)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.25, random_state=1
        )
        clf.fit(X_train, Y_train)
        return clf.predict(
            [
                [
                    budget,
                    sqfeet,
                    beds,
                    baths,
                    smoking,
                    wheelchair,
                    vehicle,
                    funrnished,
                    laundry,
                    parking,
                    state,
                ]
            ]
        )

    if submit:
        multilabel_pred = multilableclasification_specific(
            budget,
            sqfeet,
            beds,
            baths,
            smoking,
            wheelchair,
            vehicle,
            funrnished,
            laundry,
            parking,
            state,
        )
        bianry_pred = bianryclasification_specific(
            budget,
            sqfeet,
            beds,
            baths,
            smoking,
            wheelchair,
            vehicle,
            funrnished,
            laundry,
            parking,
            state,
        )
        if multilabel_pred == 0:
            multilabel_pred = "TOWN HOUSE"
        if multilabel_pred == 1:
            multilabel_pred = "CONDOMINIUM"
        if multilabel_pred == 2:
            multilabel_pred = "APARTMENT"
        if multilabel_pred == 3:
            multilabel_pred = "DUPLEX"
        if bianry_pred == 0:
            bianry_pred = "PETS NOT ALLOWED"
        if bianry_pred == 1:
            bianry_pred = "PETS ALLOWED"

        html_temp = """
        <div style="background-color:#0CB9A0;padding:1.5px">
        <h1 style="font-family:Courier; color:white;text-align:center; font-size:45px">Housing Type Prediction</h1>
        </div><br>"""

        st.markdown(html_temp, unsafe_allow_html=True)
        st.title(f"{classifier_name}")
        st.markdown(
            "<style>h1{font-family:Courier; color: #1E8054;}</style>",
            unsafe_allow_html=True,
        )

        # st.header("Classifier name")
        # st.write(f'{classifier_name}')

        st.header("Type")
        st.write(f"{multilabel_pred}")

        st.header("Pets allowed ?")
        st.write(f"{bianry_pred}")
