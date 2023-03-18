# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:02:34 2021

@author: User
"""

import train_streamlit as app1
import predict_streamlit as app2
import streamlit as st

st.set_page_config(page_title="Hackathon 2023")
# Define pages based on apps imported.
PAGES = {
    "Train": app1,
    "Predict": app2
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
