from cProfile import label

import streamlit as st
import pandas as pd
from io import BytesIO
import zipfile 

from vars import * 
from func import download_preprocess_forecast_general

# Настройка страницы
st.set_page_config(
    page_title="Анализ данных",  
    page_icon="📊",
    layout="wide"
)
st.title("📈 Загрузка и прогноз данных по Туризму")

h = st.number_input(label='Введите горизонт прогнозирования', min_value=1, max_value=3, step=1)
 


st.header("Загрузите ")

datasets_names = ['general', 'income', 'spends', 'socdem', 'duration', 'visits', 'categories']

# Файлоадер для нескольких файлов
uploaded_files = st.file_uploader(
    f"Загрузите zip с файлами CSV или XLSX", 
    type=['zip'], 
    accept_multiple_files=False
)


if uploaded_files is not None:
    with zipfile.ZipFile(uploaded_files, "r") as z:
    
        prognoz_general = download_preprocess_forecast_general(z, h)
        # prognoz_general[1] 
    
        st.dataframe(prognoz_general[1])
        csv_data = prognoz_general[1].to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Скачать CSV",
            data=csv_data,
            file_name="prognoz.csv",
            mime="text/csv"
        )
                
