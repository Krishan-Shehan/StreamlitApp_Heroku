import streamlit as st


def app():

    item = st.sidebar.selectbox(
        "Select Item",
        (
            "eggs",
            "chocolate",
            "cookies",
            "mineral water",
            "spaghetti",
            "burgers",
            "grated cheese",
            "frozen vegetables",
            "turkey",
        ),
    )

    html_temp = """
        <div style="background-color:#0FD27F;padding:1.5px">
        <h1 style="font-family:Courier; color:white;text-align:center; font-size:45px">Market Basket</h1>
        </div><br>"""

    st.markdown(html_temp, unsafe_allow_html=True)
    st.title("Frequently Bought With..")
    st.markdown(
        "<style>h1{font-family:Courier; color: #1E8054; font-size:40px}</style>",
        unsafe_allow_html=True,
    )

    if item == "chocolate":
        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">eggs</p>'
        st.markdown(original_title, unsafe_allow_html=True)

        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">mineral water</p>'
        st.markdown(original_title, unsafe_allow_html=True)

        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">spaghetti</p>'
        st.markdown(original_title, unsafe_allow_html=True)

    elif item == "cookies":
        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">eggs</p>'
        st.markdown(original_title, unsafe_allow_html=True)

    elif item == "eggs":
        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">chocolate</p>'
        st.markdown(original_title, unsafe_allow_html=True)

        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">cookies</p>'
        st.markdown(original_title, unsafe_allow_html=True)

        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">mineral water</p>'
        st.markdown(original_title, unsafe_allow_html=True)

        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">spaghetti</p>'
        st.markdown(original_title, unsafe_allow_html=True)

    elif item == "mineral water":
        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">chocolate</p>'
        st.markdown(original_title, unsafe_allow_html=True)

        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">eggs</p>'
        st.markdown(original_title, unsafe_allow_html=True)

        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">spaghetti</p>'
        st.markdown(original_title, unsafe_allow_html=True)

    elif item == "spaghetti":
        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">chocolate</p>'
        st.markdown(original_title, unsafe_allow_html=True)

        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">eggs</p>'
        st.markdown(original_title, unsafe_allow_html=True)

        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">mineral water</p>'
        st.markdown(original_title, unsafe_allow_html=True)

    else:
        original_title = '<p style="font-family:Courier; color:white; font-size: 30px; text-align:center ; font-weight: bold">No related items</p>'
        st.markdown(original_title, unsafe_allow_html=True)
