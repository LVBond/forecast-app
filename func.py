
from ast import alias

import pandas as pd
# import re
# import zipfile
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
# import time 
from datetime import datetime

# from functools import partial
# from IPython.display import display
# import utilsforecast.losses as ufl
# from utilsforecast.evaluation import evaluate
# from utilsforecast.plotting import plot_series
from utilsforecast.preprocessing import fill_gaps
from statsforecast import StatsForecast
# from statsforecast.models import MSTL
from statsforecast import StatsForecast
# from statsforecast.models import CrostonClassic
from statsforecast.models import AutoARIMA

# import copy
# from hierarchicalforecast.core import HierarchicalReconciliation
# from hierarchicalforecast.methods import MiddleOut, BottomUp 
# from hierarchicalforecast.utils import aggregate
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive
# from itertools import product
import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
import streamlit as st 
# import pickle

from vars import * 



def download_preprocess_forecast_general(z, h):

    files = z.namelist()

    # будем сохранять в папку и архив
    # folder_name_full = f'{folder_name}_lvbond_prognoz_{datetime.now().strftime("%Y-%m-%d")}_forecasts'
    # archive_name = f'{folder_name_full}.zip'
    # os.makedirs(folder_name_full, exist_ok=True)
    data_common = {}
    
    # try:
    #     list_dir = os.listdir(f'{folder_name}/')
    # except: 
    #     print(f'Нет папки {folder_name}')

    # СОБРИРАЕМ ИСТОРИЧЕСКИЕ ДАННЫЕ
    data = pd.DataFrame()
    # заходим во все папки и ищем general
    # for root, dirs, files in os.walk(folder_name):

    
        # print(root, 'dirs', dirs, 'files', files)

    for file in files:
        if "general" in file:
            # file_path = os.path.join(root, file)

            if file.endswith(".csv"):
                with z.open(file) as f:
                    data_1 = pd.read_csv(f, sep='\t') 

            # для каждого файла проверяем столбцы    
            for col_name in common_header:
                if col_name not in data_1.columns:
                    print(f'Нет колонки {col_name} в файле {file}.csv')
                    return None
            for col_name in headers['general']:
                if col_name not in data_1.columns:
                    print(f'Нет колонки {col_name} в файле {file}.csv')
                    return None
            # только строки за месяц 
            data_1 = data_1[data_1['Флаг. Детализация по году или кварталу'] == 'месяц']
            data_1 = data_1[data_1['Флаг. Детализация по региону прибытия'] == 'N'] # прогнозируем на уровне региона
            data_1 = data_1[data_1['Флаг. Детализация по домашнему региону'] == 'Субъекты РФ'] # в конце получить ВСЕ РЕГИОНЫ!
            data_1 = data_1[data_1['Домашний регион (откуда приехали)'] != 'Все регионы'] 


            data = pd.concat([data, data_1])
# СОБРАЛИ ИСТОРИЧЕСКИЕ ДАННЫЕ  
            
            data_historical = data.copy() # для визуализации
    
  
    # ОТДЕЛЬНЫЙ ДАТАФРЕЙМ ДЛЯ КАЖДОГО y
    for y_col_name in y_dict['general']:
        
        data_y =  data[['Дата. Последний день периода согласно типу', y_col_name] + cut_col]
        data_y = data_y.groupby(['Дата. Последний день периода согласно типу'] + cut_col, as_index = False).sum()

        data_y['unique_id'] = data_y[cut_col].agg('_'.join, axis=1)

        data_y = data_y.rename(columns={
            'Дата. Последний день периода согласно типу': 'ds',
            y_col_name: 'y'
            })
        # display(data_y)

        # что столбец 'ds' содержит только даты - можно туда, где загружали дату 
        try:
            data_y['ds'] = pd.to_datetime(data_y['ds'])
            # на случай, если данные не отсортированы по дате
            data_y = data_y.sort_values('ds')
        except Exception:
            print(f"Ошибка: столбец **'Дата. Последний день периода согласно типу'** не конвертируется в datetime")
            return None
    

        # что столбец 'y' содержит только числа
        if not pd.api.types.is_numeric_dtype(data_y['y']):
            try:
                data_y['y'] = pd.to_numeric(data_y['y'], errors='raise')
            except Exception:
                print(f'Ошибка: столбец **{y_col_name}** содержит нечисловые значения и не может быть конвертирован')
                return None
        

        # добавляем даты, где их нет
        Y_df = fill_gaps(
            data_y[['ds', 'unique_id', 'y']],
            freq='M',
        )
        # добавляем значение в словарь 
        data_common[y_col_name] = {}
        data_common[y_col_name]['Y_df'] = Y_df.copy().fillna(0)
                


    def arima_prognozing_df_general(data, h):  # data - словарь со структурой и иерархией 

        # model = StatsForecast(models=[AutoARIMA(season_length=12)], freq='M')
        model = StatsForecast(models=[Naive(alias='AutoARIMA')], freq='M')
        # FIXME взять конкретную ариму  

        # прогнозируем все уровни 
        Y_hat_df = model.forecast(df=data['Y_df'], h=h, fitted=True)
        Y_fitted_df = model.forecast_fitted_values().fillna(0)

        Y_hat_df['AutoARIMA'] = Y_hat_df['AutoARIMA'].clip(lower=0).round(0).astype(int)
        Y_fitted_df['AutoARIMA'] = Y_fitted_df['AutoARIMA'].clip(lower=0).round(0).astype(int)
        

        return Y_hat_df.reset_index()
    
    forecasts_df = {}
    for y in y_dict['general']:
        forecasts_df[y] = arima_prognozing_df_general(data=data_common[y], h=h)
    # return forecasts_df
    
    # STAGE 2 
    # получили прогноз general - собираем его по форме и далее используем для прогноза других датасетов
    # Собираем general в исходный вид 
    # проходим циклом по y и собираем прогнозы в один датафрейм 
    forecast_stage2 = pd.DataFrame()
    for y in forecasts_df:
        # разделить unique_id по слэшу на несколько колонок 

        # st.write(forecasts_df[y].index)
      
        a_split = forecasts_df[y]['unique_id'].str.split('_', expand=True)

        # джойн по ширине 
        # соединяем разделенную колонку с остальной частью 
        general_row = a_split.join(forecasts_df[y].iloc[:,[1,2]])

        # переименование колонок
        y_name_old = list(general_row.columns)[-1]
        general_row.rename(columns={
                                0: 'Домашний регион (откуда приехали)', 
                                1: 'Тип путешественника',
                                'ds': 'Дата. Последний день периода согласно типу',
                                y_name_old: y
                            }, inplace=True)
        
        # соединяем прогнозы по всем y в один датафрейм
        if not forecast_stage2.empty:
            forecast_stage2 = forecast_stage2.merge(
                general_row, 
                on=[
                    'Домашний регион (откуда приехали)', 
                    'Тип путешественника',
                    'Дата. Последний день периода согласно типу'
                ])
        else:
            forecast_stage2 = general_row.copy()
    # print(forecast_stage2.columns)
    

    forecast_stage2['Оборот (руб.)'] = forecast_stage2[
        [
        'Сумма по транзакциям (руб.)',
        'Сумма снятия наличных денег во время поездки и в течение одного дня до поездки (руб.)',
        'Сумма переводов (руб.)'  
        ]
    ].sum(axis=1)

    # forecast_stage2['Доля лояльных туристов, %'] = (forecast_stage2['Количество лояльных туристов (чел.)'] / forecast_stage2['Кол-во уникальных туристов (чел.)'] * 100).round(0)
    # чтобы не делить лояльных на нулевых уникальных 
    forecast_stage2['Доля лояльных туристов, %'] = np.where(
        forecast_stage2['Кол-во уникальных туристов (чел.)'] > 0,
        (forecast_stage2['Количество лояльных туристов (чел.)'] / forecast_stage2['Кол-во уникальных туристов (чел.)'] * 100).round(0),
        0
    )
    
    
    forecast_stage2['Сумма снятия наличных денег во время поездки (руб.)'] = 0
    forecast_stage2['Сумма снятия наличных денег за один день до поездки (руб.)'] = 0

    # data - исходный датафрейм
    # forecast_stage2 - с прогнозами
    # собираем в один
    data_empty = pd.DataFrame(columns=data.columns)

    # агрегируем всех туристов по домашнему региону 
    forecast_all_tourists = forecast_stage2.groupby([
            'Дата. Последний день периода согласно типу', 
            'Тип путешественника'
    ], as_index=False).sum(numeric_only=True)
    forecast_all_tourists['Флаг. Детализация по домашнему региону'] = 'Субъекты РФ' # FIXME
    forecast_all_tourists['Домашний регион (откуда приехали)'] = 'Все регионы'

    data_with_forecast = pd.concat([data_empty, forecast_stage2, forecast_all_tourists], ignore_index=True)

    # отсекаем группы менее MIN_GROUP_SIZE
    print(f"Shape ДО фильтра групп <{MIN_GROUP_SIZE} туристов: {data_with_forecast.shape}")
    before_count = len(data_with_forecast)
    data_with_forecast = data_with_forecast[data_with_forecast['Кол-во уникальных туристов (чел.)'] >= MIN_GROUP_SIZE].copy()
    print(f"Shape ПОСЛЕ фильтра: {data_with_forecast.shape} (удалено {before_count - len(data_with_forecast)} строк)")

    

    data_with_forecast['Флаг. Детализация по году или кварталу'] = data_with_forecast['Флаг. Детализация по году или кварталу'].fillna('месяц')
    data_with_forecast['Флаг. Детализация по региону прибытия'] = data_with_forecast['Флаг. Детализация по региону прибытия'].fillna('N')
    data_with_forecast['Регион прибытия (куда приехали)'] = data_with_forecast['Регион прибытия (куда приехали)'].fillna(data['Регион прибытия (куда приехали)'].iloc[0])

    # заполнение дом региона 
    data_with_forecast['Флаг. Детализация по домашнему региону'] = data_with_forecast['Флаг. Детализация по домашнему региону'].fillna('Субъекты РФ')

    # копируем субъекты РФ и все регионы - часть датафрейма
    # заменяем в двух колонках значения на все туристы 
    # конкатенируем с исходным 
    # data_copy = data_with_forecast_res[data_with_forecast_res['Флаг. Детализация по домашнему региону']=='Субъекты РФ'] # уже только эти
    data_copy = data_with_forecast[data_with_forecast['Домашний регион (откуда приехали)']=='Все регионы'] 
    data_copy['Флаг. Детализация по домашнему региону'] = 'Все туристы'
    data_copy['Домашний регион (откуда приехали)'] = 'Все туристы'
    data_with_forecast = pd.concat([data_with_forecast, data_copy])


    # сохранение в заготовленную папку и архив 
    # current_date = datetime.now().strftime('%Y-%m-%d')
    # filename = f'{FOLDER_NAME}_{current_date}_forecast_general.csv'
    # filepath = os.path.join(folder_name_full, filename)
    # data_with_forecast.to_csv(filepath, sep='\t', encoding='utf-8', index=False)
    # print(f"Сохранено в {folder_name_full}: {filename}")


    return data_historical, data_with_forecast, forecasts_df

    


