# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:32:44 2024

@author: NRN
"""

import streamlit as st

st.set_page_config(layout="wide")
st.title('tekst')

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.4, 1.6, .1, 2.6, .1))
row0_1.header('Long Term Analysis')

row0_2.write('tekst i col 2')
