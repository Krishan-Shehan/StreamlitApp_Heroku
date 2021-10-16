import streamlit as st
from multiapp import MultiApp

from apps import classification, association  # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Classification", classification.app)
app.add_app("Association", association.app)
# The main app
app.run()
