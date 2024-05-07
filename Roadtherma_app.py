# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:00:27 2024

@author: NRN
"""
#-----------------
#ANGIV version af programmet. Ændres hvis der sker store ændringer. Dette nummer skrives med i configuration filerne
#------------------

import streamlit as st 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import click
import yaml
import pandas as pd
from PIL import Image #til at åbne billeder
from datetime import date #tid dato
import json #til at gemme config filen
import seaborn as sns
import zipfile#til at gemme filer som zip
import io#til at gemme filer som zip

# from config import ConfigState
from data import load_data, create_road_pixels, create_trimming_result_pixels
from utils import split_temperature_data
from export import temperature_to_csv, detections_to_csv, temperature_mean_to_csv, clusters_to_csv
from plotting import plot_statistics, plot_detections, plot_cleaning_results, save_figures
from clusters import create_cluster_dataframe
from road_identification import clean_data, identify_roller_pixels, interpolate_roller_pixels
from detections import detect_high_gradient_pixels, detect_temperatures_below_moving_average
from readers import readers
from cli import _iter_segments
import nrn_functions #funktioner lavet primært til streamlit app
# st.set_page_config(page_title=None, layout="wide")

st.markdown('# Roadtherma user interface')
st.write('')
st.markdown('Program for analysing thermal data obtained during road paving')
st.divider()
#logo i sidebar
logo_image = Image.open('vdlogo_blaa.png')
st.sidebar.image(logo_image, caption=None, width=250)

## VERSION af koden beskrives herunder. Printes nederst ##############
current_version ='version 0.3 - NRN 13-02-2024 - ready for eksternal testing' #det der skrives i configuration filen
versions_log_txt = '''
NRN 14-02-2024  
save as zip file is incluted. Figures is incluted in zip file. 

*version 0.3 - NRN 13-02-2024 - ready for eksternal testing  
Post analysis is updated to include all privious analysis from MAP script.  
A result csv file is added.  
Graphs under statistics is added as an posibility.  
All readers is added*  

*version 0.2 - NRN 08-02-2024  - prototype  
Change sidebar to only contain relevant input.  
change save function so name from input file is added to output name*

*version 0.1 - test fase - NRN 2024*
'''


#%%If we want to get the jobs file as in the original road therma script and save it as a config dictionary, this is tehe way:
# import sys
# sys.path.append(r'C:\Users\B306742\OneDrive - Vejdirektoratet\Dokumenter\roadtherma-master\roadtherma')
# åben jobs filen. resulterer i en dictionary med info i
# import yaml
# jobs_file_path = r'C:\Users\B306742\OneDrive - Vejdirektoratet\Dokumenter\roadtherma-master\jobs.yaml'
# st.write('jobs filen der bruges nu:', jobs_file_path)

# with open(jobs_file_path) as f:
#     jobs_yaml = f.read()
# jobs = yaml.load(jobs_yaml, Loader=yaml.FullLoader) 

# from config import ConfigState
# config_state = ConfigState()
# for n, item in enumerate(jobs): #der kan være flere jobs i en yaml fil. Men vi har normalt kun en. Derfor lopper vi kun over et job her. Hvis man havde flere kørte den resten af koden inde i dette loop
#     print('item')
#     config = config_state.update(item['job']) #får info fra jobs.yaml filen ud
#     # process_job(n, config)
    

#%% ----- SIDEBAR -----------------
# Default jobs værdier
config = {} #starter en ny config dictionary hvis alle værdierne kommer fra App. 
config['Date of analysis'] = date.today().strftime('%Y-%m-%d')
config['version'] = current_version
config_default_values = {'pixel_width':0.25, 'roadwidth_threshold':50, 'autotrim_temperature':40, 'lane_threshold': 150,
                         'roller_detect_enabled':False, 'roller_detect_temperature':50, 'roller_detect_interpolation':False,
                         'gradient_enabled':True, 'gradient_tolerance':10, 'plotting_segments':1, 'show_plots':True,
                         'save_figures':True, 'write_csv':True,'autotrim_enabled':False, 'autotrim_percentage':0.2,
                         'roadwidth_adjust_left': 1, 'roadwidth_adjust_right':1, 'lane_enabled':True,'moving_average_enabled':True,
                         'moving_average_window':100, 'moving_average_percent': 90, 'gradient_statistics_enabled':False,
                         'cluster_npixels':0, 'cluster_sqm':0, 'tolerance':[5,20,1]}

st.sidebar.divider()
st.sidebar.markdown('# configuration options')
#-----
#her skal man kunne loade en fil ind der kan overksrive felter i config_default_values 
# with open('configuration_values.json') as f: #hvis en fil skal åbnes direkte og ikke igennem streamlit
#     config_values = json.load(f)#hvis en fil skal åbnes direkte og ikke igennem streamlit
    
config_values_available = st.sidebar.checkbox('Load configuration values from file')
if config_values_available:
    config_values_path = st.sidebar.file_uploader('Upload file with configuration values. OBS: right now this only work with json files created in this program. Do we want a function with yaml files aswell?') 
    if config_values_path is not None:
        config_values = json.load(config_values_path)
        st.sidebar.write(config_values)
        #Overskriver parameterværdier i config_default_values med den fra config_values
        for k in config_default_values.keys():
            if k in config_values.keys():
                config_default_values[k] = config_values[k]
#----
        
#herunder skrives parameterværdierne ind. 
if config_default_values['pixel_width'] == 0.25:  index_default = 0 
elif config_default_values['pixel_width'] == 0.03:  index_default = 1
config['pixel_width'] = st.sidebar.selectbox('Pixel width in meters.', [0.25, 0.03], index=index_default)
if config['pixel_width'] == 0.25: config_default_values['roadwidth_adjust_left']=1; config_default_values['roadwidth_adjust_right']=1
if config['pixel_width'] == 0.03: config_default_values['roadwidth_adjust_left']=8; config_default_values['roadwidth_adjust_right']=8


config['roadwidth_threshold'] = st.sidebar.number_input('Threshold temperature used when estimating the road width (roadwidth_threshold)', value=config_default_values['roadwidth_threshold'], step=1) #roadwidth_threshold: 50 # Threshold temperature used when estimating the road width.
config['lane_threshold'] = st.sidebar.number_input('Threshold temperature used when detecting the paving lane (lane_threshold)', value=config_default_values['lane_threshold'] , step=1)#     lane_threshold: 150.0      # Threshold temperature used when detecting the paving lane.
config['gradient_enabled'] = st.sidebar.toggle('gradient_enabled', value=config_default_values['gradient_enabled'] )#gradient_enabled: True             # Whether or not to make detections using the "gradient" method.
config['gradient_tolerance'] = st.sidebar.number_input('gradient_tolerance', value=config_default_values['gradient_tolerance'] , step=1)# gradient_tolerance: 10.0 # Tolerance on the temperature difference during temperature gradient detection.    

#--- config indstillinger der ikke skal kunne ændres i appen, men stadig skal være i config filen. Gemmer den oprindelige widget så den er nem at sætte ind igen --- 
config['show_plots'] = config_default_values['show_plots'] #st.sidebar.toggle('show_plots', value=config_default_values['show_plots'] )# Whether or not to show plots of data cleaning and enabled detections. default=True
config['save_figures'] = config_default_values['save_figures'] #st.sidebar.toggle('save_figures', value=config_default_values['save_figures'] )# Whether or not to save the generated plots as png-files instead of showing them.
config['write_csv'] = config_default_values['write_csv'] #st.sidebar.toggle('write_csv', value=config_default_values['write_csv'] )# Whether or not to write csv files for post-analysis.
config['autotrim_enabled'] =config_default_values['autotrim_enabled']# st.sidebar.toggle('autotrim_enabled', value=config_default_values['autotrim_enabled'])# Whether or not to use autotrimming. If set to False the values in the four manual_trim_* entries is used to crop the data.
config['autotrim_temperature'] = config_default_values['autotrim_temperature']#st.sidebar.number_input('Temperature threshold for the data trimming step (autotrim_temperature)', value=config_default_values['autotrim_temperature'], step=1) #autotrim_temperature: 40.0 # Temperature threshold for the data trimming step.
config['cluster_npixels'] = config_default_values['cluster_npixels']#st.sidebar.number_input('cluster_npixels', value=config_default_values['cluster_npixels'])# cluster_npixels: 0    # Minimum amount of pixels in a cluster. Clusters below this value will be discarded.
config['cluster_sqm'] = config_default_values['cluster_sqm'] #st.sidebar.number_input('cluster_sqm', value=config_default_values['cluster_sqm'])# cluster_sqm: 0.0             # Minimum size of a cluster in square meters. Clusters below this value will be discarded.
#st.sidebar.write('Range of tolerance temperature values to use when plotting percentage of road that is comprised:')#     tolerance: [5, 20, 1]        # Range of tolerance temperature values '[<start>, <end>, <step size>]' to use when plotting percentage of road that is comprised of high gradients vs gradient tolerance.
#c1, c2, c3 = st.sidebar.columns(3)
#with c1:
#    val_1 = st.number_input('start', value=config_default_values['tolerance'][0])
#with c2: val_2=st.number_input('end', value=config_default_values['tolerance'][1])
#with c3: val_3=st.number_input('stepsize', value=config_default_values['tolerance'][2])
#config['tolerance'] = [val_1, val_2, val_3] 
config['tolerance'] = config_default_values['tolerance']
config['lane_to_use'] = 'warmest' #st.sidebar.selectbox('Use the "coldest" or "warmest" lane for detections.', ['coldest', 'warmest'], index=1)#     lane_to_use: warmest       # Whether to use the "coldest" or "warmest" lane for detections.
config['lane_enabled'] = config_default_values['lane_enabled']# st.sidebar.toggle('lane_enabled. Whether or not to try and identify lanes', value=config_default_values['lane_enabled'])# lane_enabled: True         # Whether or not to try and identify lanes.
config['plotting_segments'] = config_default_values['plotting_segments']  #st.sidebar.number_input('plotting_segments', value=config_default_values['plotting_segments'] , step=1)# Number of segments that the data is partitioned into before being plotted. If set to 1 the entire dataset is plotted into figure (for each plot-type).
config['roller_detect_enabled'] = config_default_values['roller_detect_enabled']# st.sidebar.toggle('roller_detect_enabled. Whether or not to use roller_detection.', value=config_default_values['roller_detect_enabled']) #roller_detect_enabled: False  # Whether or not to use roller_detection.
config['roller_detect_temperature'] = config_default_values['roller_detect_temperature']# st.sidebar.number_input('roller_detect_temperature', value=config_default_values['roller_detect_temperature'] , step=1)#roller_detect_temperature: 50      # Threshold temperature used in roller-detection (road temperature pixels below this temperature is categorized as roller).
config['roller_detect_interpolation'] = config_default_values['roller_detect_interpolation']#st.sidebar.toggle('roller_detect_interpolation',value=config_default_values['roller_detect_interpolation'] ) #roller_detect_interpolation: True  # If set to True the pixels identified as being the roller is interpolated with mean temperature of the road
config['autotrim_percentage'] =config_default_values['autotrim_percentage']# st.sidebar.number_input('autotrim_percentage', value=config_default_values['autotrim_percentage'] )#autotrim_percentage: 0.2   # Percentage threshold of data below <autotrim_temperature> in order for an edge longitudinal line to be removed.TF_time
config['moving_average_window'] = config_default_values['moving_average_window'] # st.sidebar.number_input('moving_average_window', value=config_default_values['moving_average_window'] , step=1)#moving_average_window: 100.0       # Windowsize in meter to calculate (centered) moving average.
config['moving_average_percent'] = config_default_values['moving_average_percent'] #st.sidebar.number_input('moving_average_percent', value=config_default_values['moving_average_percent'])# moving_average_percent: 90.0       # Percentage used to detect low temperatures, i.e., road pixels where pixel < "percentage of moving average temperature"
config['moving_average_enabled'] = config_default_values['moving_average_enabled']#st.sidebar.toggle('moving_average_enabled', value=config_default_values['moving_average_enabled'])#moving_average_enabled: True  # Whether or not to make detections using the "moving average" method.
config['roadwidth_adjust_left'] = config_default_values['roadwidth_adjust_left'] # st.sidebar.number_input('Additional number of pixels to cut off left edge after estimating road width (roadwidth_adjust_left)', value=config_default_values['roadwidth_adjust_left'] , step=1)
config['roadwidth_adjust_right'] = config_default_values['roadwidth_adjust_right'] # st.sidebar.number_input('Additional number of pixels to cut off right edge after estimating road width ( roadwidth_adjust_right)', value=config_default_values['roadwidth_adjust_right'] , step=1)
config['title'] = 'Example plot of test section'  # String used as title in the figures created. Mandatory
title= config['title']
########---------------------------------------------------------------------



#%% -- Upload data ---
st.subheader('Select data to analyze')
# Data loades ind ved hjælp af en counter der gemmes i session_state variablen. Dette gør at 
# der ikke skal loades en dataframe hver gang der ændres på en parameter. Godt når der bruges store dataframes 

if 'count' not in st.session_state: #starter emd at indsætte count værdierne i session state vaiablen. 
    st.session_state.count = 0
if 'count_new' not in st.session_state:
    st.session_state.count_new = 0

    
def counter_func():
    #funktion der tæller hvis der er ændret i reader eller file upload.
    #denne bruges sammen med on_change således at der tælles hvis der ændres i disse widgets
    st.session_state.count = st.session_state.count+1
    return 

col1, col2 = st.columns(2)
config['reader'] = None #starter med en tom
#navnene på alle readers i readers.py gemmes her så de kan vælges
reader_list = ['voegele_M30','TF_time_K', 'voegele_example','voegele_M119', 'voegele_taulov','TF_old',
               'TF_new', 'TF_notime','TF_time', 'TF_time_new','moba','moba2','moba3']
               
with col1:
    st.write('Reader is used to load input file. The choice of reader depends on the input file\'s formatting.')
    config['reader'] = st.selectbox('Define which reader to use', reader_list,index=None, placeholder="Choose an option",key='reader',#['voegele_M30','TF_time_K']
                                    on_change=counter_func )
    

#herunder oploades data
if 'uploaded_data' not in st.session_state: #starter med at være tom
    st.session_state['uploaded_data']=None

with col2:
    st.write('Input file must be a csv file and in uft-8 encoding. If this is not the case, change it manually.')
    uploaded_file = st.file_uploader('Choose input file', key='uploadFile', on_change=counter_func )

#load den oploadede fil ind baseret på readers
if st.session_state.count != st.session_state.count_new:
    # st.write('count er ikke lig count ny')
    st.session_state.count_new= st.session_state.count_new+1
    if config['reader'] is None:
        st.write('You have to choose a reader before data is loaded.')
    elif uploaded_file is not None:
        #uploaded_file er "stien" til den uplodede data. Nogle filers readers giver både dataframe og tekst
        if config['reader']=='TF_time_K':
            st.session_state['uploaded_data'], additional_text = load_data(uploaded_file, config['reader'])
            st.write(additional_text)
        else:
            st.session_state['uploaded_data'] = load_data(uploaded_file, config['reader'])
            config['input data'] = st.session_state.uploadFile.name
        #printer den uploaded dataframe 
        st.dataframe(st.session_state['uploaded_data'])
elif st.session_state.count == st.session_state.count_new:
    # st.write('count = count_new så df bliver bare stående')
    st.dataframe(st.session_state['uploaded_data'])
    #st.write('count = count_new')


df = st.session_state['uploaded_data']#gemmer denne dataframe til brug i resten af koden. 

#=== Hvis man vil hente en datafil direkte kan dette også gøres således ===
# config['reader'] = 'voegele_M30' #her skal reader angives
# file_path = config['file_path'] #her skal stien skrives 
# df = load_data(file_path, config['reader'])
# title = config['title']
# print('Processing data file #{} - {}'.format(n, title))
# print('Path: {}'.format(file_path))
# st.write('Input filen er:', file_path) 
#--------------------------------------------------------------------------

st.divider()
### Herunder kører hele programmet ##############
######Istedet for process_job så sættes funktion ind herunder.  ######

figures = {}

#%% ## finding the road
st.subheader('Trimming data and identifying road')
st.write('When data is loaded correctly start trimming data and identifying the road. ')
run_trimming_checkbox = st.checkbox('run this section')

if run_trimming_checkbox:
    st.write('The parameters used for trimming are specified here')
    with st.form(key='columns_in_form'):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            config['manual_trim_transversal_start'] = st.number_input('manual_trim_transversal_start', value=2.5)
        with c2:
            config['manual_trim_transversal_end'] = st.number_input('manual_trim_transversal_end', value=9.5)
        with c3:
            config['manual_trim_longitudinal_start'] = st.number_input('manual_trim_longitudinal_start', value=0)
        with c4:
            config['manual_trim_longitudinal_end'] = st.number_input('manual_trim_longitudinal_end', value=df.distance.iloc[-1])
        submitButton = st.form_submit_button(label = 'Calculate') #når denne trykkes sendes de nye værdier til programmet
    
    #Her deles dataframen ind i en df med temperatur dta og en med resten af kolonnerne.
    temperatures, metadata = split_temperature_data(df)
    # st.write(temperatures)#plot af dataframe
    
    # Initial trimming & cleaning of the dataset
    #giver dataframe der er trimmet i siderne og længden baseret på info i jobs filen og hvis der er en anden kørebane. + 
    temperatures_trimmed, trim_result, lane_result, roadwidths = clean_data(temperatures, metadata, config)
    
    #-- herunder plottes rådata og trimmed data ud fra de parametre der sættes
    fig_heatmaps, (ax1, ax2) = plt.subplots(ncols=2, sharey = True)
    plt.suptitle('Raw data', fontsize=10)
    ax1.set_title('All data'); ax2.set_title('Trimmed data')
    nrn_functions.heatmaps_temperature_pixels(ax1, temperatures.values, metadata.distance, config['pixel_width'], include_colorbar=False)
    ax1.set_ylabel('Distance [m]'); ax1.set_xlabel('Road width [meters]')
    #inkluder grænser på rå data figuren
    ax1.axvline(config['manual_trim_transversal_start'],color='k' );  ax1.axvline(config['manual_trim_transversal_end'],color='k' )
    #inkluder longitudinal grænser også
    ax1.axhline(config['manual_trim_longitudinal_start'], color='k'); ax1.axhline(config['manual_trim_longitudinal_end'], color='k')
    ax1.set_ylim([ metadata.distance.iloc[-1],0 ])#når longitudinal linjer tilføjes sætter vi også en yaxis lim
    
    trimmed_data_df = temperatures_trimmed.values
    nrn_functions.heatmaps_temperature_pixels(ax2, trimmed_data_df, metadata.distance[trim_result[2]:trim_result[3]], config['pixel_width'])
    ax2.set_xlabel('Road width [meters]')
    plt.tight_layout()
    st.pyplot(fig_heatmaps)
    #---- slut på plot ---- 
    
    st.write('Based on the temperatures and chosen trimming values, the paved road section is identified.')
    #laver en dataframe med True der hvor den varme vej er og False der hvor der er under 50 grader
    road_pixels = create_road_pixels(temperatures_trimmed.values, roadwidths)
        
    if config['roller_detect_enabled']: #hvis den er slået til 
        roller_pixels = identify_roller_pixels(
            temperatures_trimmed.values, road_pixels, config['roller_detect_temperature'])
        if config['roller_detect_interpolation']:
            interpolate_roller_pixels(temperatures_trimmed.values, roller_pixels, road_pixels)
            #herunder plottes det der er identificeret som vej. 
            #her giver ikke vej = 0, vej = 1 og roller = 2
            pixel_category = create_trimming_result_pixels(
                temperatures.values, trim_result, lane_result[config['lane_to_use']],
                roadwidths, roller_pixels, config
            )
            fig_heatmap1, (ax1) = plt.subplots(ncols=1)
            ax1.set_title('Identified road')
            nrn_functions.heatmap_identified_road(ax1, pixel_category, metadata.distance, config['pixel_width'], categories=['non-road pixels', 'Road pixels', 'Pixels with temperature below {}'.format(config['roller_detect_temperature'])])
            st.pyplot(fig_heatmap1)
    elif ~config['roller_detect_enabled']:
        pixel_category1 = nrn_functions.create_identified_road_pixels(temperatures.values, trim_result, lane_result[config['lane_to_use']],roadwidths)
        fig_heatmap1, (ax1) = plt.subplots(ncols=1, )
        ax1.set_title('Identified road')
        nrn_functions.heatmap_identified_road(ax1, pixel_category1, metadata.distance, config['pixel_width'], categories=['non-road', 'road'])
        ax1.set_ylabel('Distance [m]'); ax1.set_xlabel('Road width [meters]')
        st.pyplot(fig_heatmap1)
#--------

st.subheader('Run analysis')
st.write('When the trimming is ok, run rest of the code by checking the box below. ')
run_script_checkbox = st.checkbox('run rest of the code')

#%% Herunder køres programmet baseret på trimningen ovenover 
if run_script_checkbox: 
    # Calculating detections
    #Her regnes og sammenlignes med moving average af arealet rindt om hver pixel
    moving_average_pixels = detect_temperatures_below_moving_average(
        temperatures_trimmed,
        road_pixels,
        metadata,
        percentage=config['moving_average_percent'],
        window_meters=config['moving_average_window']
        )
    #Her kigges der efter store spring i temperatur med naboer.
    gradient_pixels, clusters_raw = detect_high_gradient_pixels(
        temperatures_trimmed.values,
        roadwidths,
        config['gradient_tolerance'],
        diagonal_adjacency=True
    )
    
      
    # Plot detections results
    for k, (start, end) in _iter_segments(temperatures_trimmed, config['plotting_segments']):
        kwargs = {
            'config': config,
            'metadata': metadata.iloc[start:end, :],
            'temperatures_trimmed': temperatures_trimmed.iloc[start:end, :],
            'roadwidths': roadwidths[start:end],
            'road_pixels': road_pixels[start:end, :],
            'moving_average_pixels': moving_average_pixels[start:end, :],
            'gradient_pixels': gradient_pixels[start:end, :],
            }
        plot_detections(k, figures, **kwargs)
    
    st.pyplot(figures['moving_average']) #plot af moving average resultaterne
    if config['gradient_enabled']: st.pyplot(figures['gradient'])
    
    
#%% analyse efter moving average er udført 
st.subheader('Statistis')
# if run_script_checkbox: 
#     # Plot statistics in relating to the gradient detection algorithm
#     if config['gradient_statistics_enabled']:
#         figures['stats'] = plot_statistics(
#             title,
#             temperatures_trimmed,
#             roadwidths,
#             road_pixels,
#             config['tolerance']
#         )
#         # st.pyplot(figures['stats'])

#Der laves to seperate figurer således at den tidskrævende del kan vælges fra
config['gradient_statistics_enabled'] = st.toggle('Plot percentage high gradient as a function of tolerance (gradient_statistics_enabled). OBS: this is quite time consuming', value=config_default_values['gradient_statistics_enabled']) #gradient_statistics_enabled: True # Whether or not to calculate and plot gradient statistics

if config['gradient_statistics_enabled']:
    figures['stats_gradient'] = nrn_functions.plot_statistics_gradientPlot(title,temperatures_trimmed,roadwidths,road_pixels,config['tolerance'])
    st.pyplot(figures['stats_gradient'])

st.toggle('Plot distribution of temperatures', value=False, key='plot_temp_dist')
if st.session_state.plot_temp_dist == True:
    c1, c2 = st.columns(2)
    with c1: x_lower = st.number_input('lower limit', value=np.min(temperatures_trimmed.values[road_pixels]))
    with c2: x_higher = st.number_input('higher limit', value = np.max(temperatures_trimmed.values[road_pixels]))
    figures['stats_tempDist'] = nrn_functions.plot_statistics_TempDistributionPlot(title, temperatures_trimmed, road_pixels, limits=[x_lower,x_higher])  
    st.pyplot(figures['stats_tempDist'])


#%%-------
st.subheader('post analysis')

if run_script_checkbox:
    # DET MATTEOS SCRIPT GØR 
    #counting nuber of pixels in the road
    number_1 = np.count_nonzero(road_pixels)
    #count all elements in dataframe
    number_2 = np.count_nonzero(moving_average_pixels)
    #ratio 
    ratio = number_2/number_1
    
    
    st.markdown('Number of pixels identified as road: **{}**'.format(number_1))
    st.markdown('Number of pixels detected with moving average method: **{}**'.format(number_2))
    st.markdown('Ratio af pixels below {}% of moving average temperature: **{}%**'.format(config['moving_average_percent'], np.round(ratio*100,2) ))
    
    #Regner intervaller
    statistics_dataframe = nrn_functions.summary_as_MAP(temperatures_trimmed, road_pixels, moving_average_pixels, filename=st.session_state.uploadFile.name)
     

    txt = """ 
    | | 10 degrees gap |20 degrees gap| 30 degrees gap|
    |:-- |:-----| :-----|:-----|
    |Interval with maximum number of temperatures | {gap10}  | {gap20} | {gap30}| 
    | Percentage | {degree10_perc} % | {degree20_perc} %| {degree30_perc} %| 
    """.format(degree10_perc=statistics_dataframe['Percent with 10 degrees gap'][0],
                degree20_perc =statistics_dataframe['Percent with 20 degrees gap'][0],
                degree30_perc =statistics_dataframe['Percent with 30 degrees gap'][0],
                gap10 = statistics_dataframe['10 degrees gap'][0],
                gap20 = statistics_dataframe['20 degrees gap'][0],
                gap30 = statistics_dataframe['30 degrees gap'][0])
    st.write('#')
    st.markdown(txt)
    st.write('#')
    
#%%---- Herunder gemmes csv når man er klar
st.subheader('Save results')
st.write('When the analysis is good enough, the result can be saved either as one combined zip file or individually.')
#----- 

#Herunder er gemme funktion lavet med download knapper

if run_script_checkbox:
   #Laver de dataframes der gemmes senere -- 
   save_raw_temp_df = nrn_functions.temperature_to_csv( temperatures_trimmed, metadata, road_pixels)
   save_ma_detections_df = nrn_functions.detections_to_csv( temperatures_trimmed, metadata, road_pixels, moving_average_pixels)
   save_gradient_detections_df = nrn_functions.detections_to_csv( temperatures_trimmed, metadata, road_pixels, gradient_pixels)
   save_mean_temp_df = nrn_functions.temperature_mean_to_csv(temperatures_trimmed, road_pixels)
   
   input_file_name = st.session_state.uploadFile.name[:-4]+'_'#fjerner .csv
   
   st.markdown('The default name to appear in all saved files is extracted from the input file. It can be changed here. This affects all files in the zip folder and the individual files.')
   input_file_name = st.text_input('Change default name', value=input_file_name)

# if run_script_checkbox:
#     st.write('gem figurer - hvis det skal gøres enkeltvis ')
#     #gemmer figurer
#     fn = 'heatmap_of_trimming.png' #navnet på figur
#     img = io.BytesIO() #allokerer plads så det er mere effektivt
#     fig_heatmaps.savefig(img, format='png') #gemmer figuren ned 
    
#     btn = st.download_button(
#         label="Download image",
#         data=img,
#         file_name=fn,
#         mime="image/png"
#         )
    
if run_script_checkbox:
    
    st.markdown('### It is posible to download all files in one folder')
    st.markdown('This will download several result files, configuration file and uploaded data. If you which to see the individual files before saving look at the individual file below ')
    raw_data_df = st.session_state['uploaded_data'] #den uploadede datafil
    
    #herunder kan det hele gemmes i zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "x") as csv_zip:
        csv_zip.writestr(input_file_name+"Results.csv", pd.DataFrame(statistics_dataframe).to_csv())
        csv_zip.writestr(input_file_name+"raw_data_on_road.csv", pd.DataFrame(save_raw_temp_df).to_csv())
        csv_zip.writestr(input_file_name+"moving_average_detections.csv", pd.DataFrame(save_ma_detections_df).to_csv())
        csv_zip.writestr(input_file_name+"gradient_detections.csv", pd.DataFrame(save_gradient_detections_df).to_csv())
        csv_zip.writestr(input_file_name+"mean_temperatures.csv", pd.DataFrame(save_mean_temp_df).to_csv())
        csv_zip.writestr(input_file_name+"raw_data.csv", pd.DataFrame(raw_data_df).to_csv())
        csv_zip.writestr(input_file_name+"configuration_values.json", json.dumps(config, indent=1))
        #gemmer figurer
        fig_name = input_file_name+'heatmap_of_trimming.png'
        csv_zip.write(fig_name, fig_heatmaps.savefig(fig_name, format='png') )
        fig_name = input_file_name+'identified_road.png'
        csv_zip.write(fig_name, fig_heatmap1.savefig(fig_name, format='png') )
        for fi in figures.keys():
            fig_name = input_file_name+fi+'.png'
            csv_zip.write(fig_name, figures[fi].savefig(fig_name, format='png') )
        
        
    st.download_button(
        label="Download zip",
        data=buf.getvalue(),
        file_name=input_file_name+"analysis_results.zip",
        mime="application/zip",
        )

if run_script_checkbox:
    st.markdown('### It is posible to download individual files')
    
    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    
    
    #
    
    
    ### Output med de ting vi gerne vil have fra entreprenørerne 
    st.write('#')
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        st.markdown('Save the results of post analysis of data. You can look at the data below before saving.')
    save_name0 = st.text_input('output name', value=input_file_name+'Results.csv')#som default skal alle filer hedde navnen på inout filen + mere
    with c2:
        st.download_button(
            label='Save results',
            data = convert_df(statistics_dataframe),
            file_name=save_name0,
            mime = 'text/csv'
            )
    with st.expander('See dataframe before saving'):
        st.dataframe(statistics_dataframe)
        
    #raw temperatures in pixels belonging to road
    st.write('#')
    
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        st.write('Save raw temperatures in pixels belonging to road. You can look at the data below before saving.')
    save_name = st.text_input('output name', value=input_file_name+'raw_data_on_road.csv')#som default skal alle filer hedde navnen på inout filen + mere
            
    with c2:
        st.download_button(
            label='Save raw temperatures in road',
            data = convert_df(save_raw_temp_df),
            file_name=save_name,
            mime = 'text/csv'
            )
    with st.expander('See dataframe before saving'):
        st.dataframe(save_raw_temp_df)
    
    #Save all pixels categorized as either non-road, road or detected as having a temperature below a moving average
    st.write('#')#laver luft imellem knapperne
    c1, c2 = st.columns([0.5, 0.5])
    save_name1 = st.text_input('output name', value=input_file_name+'moving_average_detections.csv')
    with c2:
        st.download_button(
            label='Save MA results',
            data = convert_df(save_ma_detections_df),
            file_name=save_name1,
            mime = 'text/csv'
            )
    with c1:
        st.write('Save all pixels categorized as either non-road, road or detected as having a temperature below a moving average. You can look at the data below before saving. ')
    with st.expander('See dataframe before saving'):
        st.dataframe(save_ma_detections_df)
        
    #Save all pixels categorized as either non-road, road or gradient
    st.write('#')
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        st.write('Save all pixels categorized as either non-road, road or gradient. You can look at the data below before saving.')
    save_name2 = st.text_input('output name', value=input_file_name+'gradient_detections.csv')
    with c2:
        st.download_button(
            label='Save gradient results',
            data = convert_df(save_gradient_detections_df),
            file_name=save_name2,
            mime = 'text/csv'
            )
    with st.expander('See dataframe before saving'):
        st.dataframe(save_gradient_detections_df)
        
    #Save the mean temperature for each distance
    st.write('#')
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        st.write('Save the mean temperature for each distance. You can look at the data below before saving.')
    save_name3 = st.text_input('output name', value=input_file_name+'mean_temperatures.csv')
    with c2:
        st.download_button(
            label='Save mean temperatures',
            data = convert_df(save_mean_temp_df),
            file_name=save_name3,
            mime = 'text/csv'
            )
    with st.expander('See dataframe before saving'):
        
        st.dataframe(save_mean_temp_df)
    
    #Save config file Save the used parameter values in a configuration file. This enables usto run the exact same analysis again.
    st.write('#')
    config_json = json.dumps(config, indent=1)
    c1, c2 = st.columns([0.5, 0.5])
    with c1: 
        st.write('Save the used parameter values in a configuration file. This enables usto run the exact same analysis again. You can look at the file below before saving.')
    
    save_name4 = st.text_input('output name', value=input_file_name+'configuration_values.json')
    with c2:
        st.download_button(
            label='Save configuration file',
            file_name=save_name4,
            mime = "application/json",
            data = config_json
            )
    with st.expander('See configuration file before saving'):
        st.write(config)




#%% Herunder er gemme funktionen der virker med en output sti. Dette virker kun når scriptet køres lokalt på en computer

# def submit_save_func():
#     #Denne funktion udføres når der kligges på submin knappen. 
#     #At gemme det i en funktion gør at det kun gøres når der klikkes på submit knappen og ikke ellers. 
#     if st.session_state.save_raw_temp:
#         # df = nrn_functions.temperature_to_csv( temperatures_trimmed, metadata, road_pixels)
#         txt = st.session_state.save_path+'\\raw_temp_on_road.csv'
#         save_raw_temp_df.to_csv(txt)
#         # st.write('file saved as {}'.format(txt))
        
#     if st.session_state.save_ma_detections:
#         txt = save_path+'\\moving_average_detections.csv'
#         # df = nrn_functions.detections_to_csv( temperatures_trimmed, metadata, road_pixels, moving_average_pixels)
#         save_ma_detections_df.to_csv(txt)
#         # st.write('file saved as {}'.format(txt))
        
#     if st.session_state.save_gradient_detections:
#         txt = save_path+'\\gradient_detections.csv'
#         # df = nrn_functions.detections_to_csv( temperatures_trimmed, metadata, road_pixels, gradient_pixels)
#         save_gradient_detections_df.to_csv(txt)

#     if st.session_state.save_mean_temp:
#         txt = save_path+'\\mean_temperatures.csv'
#         # df = temperature_mean_to_csv(temperatures_trimmed, road_pixels)
#         save_mean_temp_df.to_csv(txt)

    

# if run_script_checkbox:
#     with st.form('save output'):
#         save_path = st.text_input('File location to save csv files', key='save_path')#gemmes som st.session_state.save_path
        
#         st.write(' Check the box for the files you which to save ')
#         save_raw_temp = st.checkbox('Save raw temperatures in pixels belonging to road', value=False, key='save_raw_temp')
#         with st.expander('See dataframe before saving'):
#             save_raw_temp_df = nrn_functions.temperature_to_csv( temperatures_trimmed, metadata, road_pixels)
#             st.dataframe(save_raw_temp_df)
            
#         save_ma_detections = st.checkbox('Save all pixels categorized as either non-road, road or detected as having a temperature below a moving average.',
#                                          value=False, key = 'save_ma_detections')
#         with st.expander('See dataframe before saving'):
#             save_ma_detections_df = nrn_functions.detections_to_csv( temperatures_trimmed, metadata, road_pixels, moving_average_pixels)
#             st.dataframe(save_ma_detections_df)
            
#         save_gradient_detections= st.checkbox('Save all pixels categorized as either non-road, road or gradient', value=False,
#                                               key='save_gradient_detections')
#         with st.expander('See dataframe before saving'):
#             save_gradient_detections_df = nrn_functions.detections_to_csv( temperatures_trimmed, metadata, road_pixels, gradient_pixels)
#             st.dataframe(save_gradient_detections_df)
            
#         save_mean_temp = st.checkbox('Save the mean temperature for each distance', value=False, key='save_mean_temp')
#         with st.expander('See dataframe before saving'):
#             save_mean_temp_df = nrn_functions.temperature_mean_to_csv(temperatures_trimmed, road_pixels)
#             st.dataframe(save_mean_temp_df)
        
#         save_configuration_file =st.checkbox('Save the used parameter values in a configuration file. This enables usto run the exact same analysis again.',
#                                              value=False, key='save_configuration_file')
#         with st.expander('See configuration file before saving'):
#             st.write(config)
        
#         save_submitted = st.form_submit_button('Save the chosen files', on_click=submit_save_func)
#     #END form 
   
#     # #Herunder printes hvilke ting der er gemt i formen 
#     if save_raw_temp:
#         txt = st.session_state.save_path+'\\raw_temp_on_road.csv'
#         st.write('file saved as {}'.format(txt))
        
#     if save_ma_detections:
#         txt = save_path+'\\moving_average_detections.csv'
#         st.write('file saved as {}'.format(txt))
        
#     if save_gradient_detections:
#         txt = save_path+'\\gradient_detections.csv'
#         st.write('file saved as {}'.format(txt))
        
#     if st.session_state.save_mean_temp:
#         txt = save_path+'\\mean_temperatures.csv'
#         st.write('file saved as {}'.format(txt))
        
#     if save_configuration_file:
#         txt = save_path+'\configuration_values.json'
#         txt = save_path+'\configuration_values.json'
#         with open(txt, 'w') as f:
#             json.dump(config, f, indent=1)
#         st.write('configuration information file is saved at {}'.format(txt))


#%%
st.divider()
txt = '''
*This application is developed for The Danish Road Directorate with the purpose of analysing thermal data obtained during road paving.*  
*It builds upon the tool road therma found on https://github.com/roadtools/roadtherma*
'''
st.markdown(txt)

with st.expander('Version log'):
    st.markdown('**Version log**')
    
    st.markdown(versions_log_txt)
