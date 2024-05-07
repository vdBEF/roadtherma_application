# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:15:37 2024

@author: NRN
"""

# import streamlit as st


# st.header('Thermgrafi App')
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# import csv

#%% #henter jobs filen på samme måde som RoadTherma script
#Vær i stand til at bruge filerne i roadtherma mappen
import sys
sys.path.append(r'C:\Users\B306742\OneDrive - Vejdirektoratet\Dokumenter\roadtherma-master\roadtherma')

### åben jobs filen. resulterer i en dictionary med info i
import yaml
# jobs_file_path = r'C:\Users\B306742\OneDrive - Vejdirektoratet\Dokumenter\roadtherma-master\jobs.yaml'
jobs_file_path = r'K:\DT\BBM\BEF\NRN\Infrared cameras analysis\dashboard test\jobs_til_KVS_20230922_XBB_TF_Entr29_Hldv125_Hoved_1ud_1.yaml'


with open(jobs_file_path) as f:
    jobs_yaml = f.read()
jobs = yaml.load(jobs_yaml, Loader=yaml.FullLoader) 

from config import ConfigState
config_state = ConfigState()
for n, item in enumerate(jobs): #der kan være flere jobs i en yaml fil. Men vi har normalt kun en. Derfor lopper vi kun over et job her. Hvis man havde flere kørte den resten af koden inde i dette loop
    print(item)
    config = config_state.update(item['job']) #får info fra jobs.yaml filen ud

#%%
title = config['title']
file_path = config['file_path']
file_path = r'K:\DT\BBM\BEF\NRN\Infrared cameras analysis\dashboard test\KVS_20230922_XBB_TF_Entr29_Hldv125_Hoved_1ud_1.csv'

#Funktion der henter data alt efter hvilken reader der er angivet
from readers import readers
if config['reader']=='TF_time_K':
    df, str1 = readers[config['reader']](file_path)
else:
    df = readers[config['reader']](file_path)
    
#%%
from utils import split_temperature_data
temperatures, metadata = split_temperature_data(df)

#%% Data is cleaned. If autotrim_enabled: False it is done by

width = config['pixel_width']
length = len(temperatures.columns)
transversal = np.arange(0, length * width, width)#array med tal der svare til længden ved pixel 1=0.25, pixel 2= 0.5, pixel 3=0.75 etc. 
longi_start = config['manual_trim_longitudinal_start']
longi_end = config['manual_trim_longitudinal_end']
trans_start = config['manual_trim_transversal_start']
trans_end = config['manual_trim_transversal_end']

def _interval2indices(distance, start, end):
    indices, = np.where(distance > start)
    start_idx = min(indices)
    indices, = np.where(distance < end)
    end_idx = max(indices)
    return start_idx, end_idx
#dette regner hvor mange kolonner der skal skæres fra for at vi starter de rigtige meter inde. 
# i jobs filen sætter man hvor meget der skal skæres væk i siderne. 
column_start, column_end = _interval2indices(transversal, trans_start, trans_end)
# Det samme i længde retningen. som default er denne sat til 0, altså fra starten og 9999, altså helt til slutningen. 
row_start, row_end = _interval2indices(metadata.distance.values, longi_start, longi_end)

trim_result = column_start, column_end, row_start, row_end #denne variabel skal bruges senere

#denne dataframe er trimmet i siderne og længden baseret på ovenstående. 
temperatures_trimmed = temperatures.iloc[row_start:row_end, column_start:column_end]

#---
#nrn plot 
import nrn_functions
fig_heatmaps, (ax1, ax2) = plt.subplots(ncols=2)
ax1.set_title('Raw data')
nrn_functions.heatmaps_temperature_pixels(ax1, temperatures.values, metadata.distance, config['pixel_width'], include_colorbar=False)
#inkluder grænser på rå data figuren
ax1.axvline(config['manual_trim_transversal_start'],color='k' )
ax1.axvline(config['manual_trim_transversal_end'],color='k' )
ax1.axhline(config['manual_trim_longitudinal_start'], color='k'); ax1.axhline(config['manual_trim_longitudinal_end'], color='k')
#når longitudinal linjer tilføjes sætter vi også en yaxis lim
ax1.set_ylim([ metadata.distance.iloc[-1],0 ])
ax2.set_title('Trimmed data')
trimmed_data_df = temperatures_trimmed.values
nrn_functions.heatmaps_temperature_pixels(ax2, trimmed_data_df, metadata.distance[trim_result[2]:trim_result[3]], config['pixel_width'])
plt.tight_layout()
#---

#%% der undersøges om der er en nylavet lane ved siden af. Dette gøres i 
def detect_paving_lanes(df, threshold):
    """
    Detect lanes the one that is being actively paved during a two-lane paving operation where
    the lane that is not being paved during data acquisition has been recently paved and thus
    having a higher temperature compared to the surroundings.
    """
    df = df.copy(deep=True)
    df_temperature, _df_rest = split_temperature_data(df) #sikre os at der kun er temperatur kolonner i data. Er der fordi vi har gjort det tidligere.
    pixels = df_temperature.values #laver om til array
    seperators = _calculate_lane_seperators(pixels, threshold)
    if seperators is None:
        lanes = {
                'warmest': (0, pixels.shape[1]),
                'coldest': (0, pixels.shape[1])
                }
    else:
        lanes = _classify_lanes(df_temperature.values, seperators)
    return lanes
def _calculate_lane_seperators(pixels, threshold):
    # mean for each longitudinal line:
    mean_temp = np.mean(pixels, axis=0)

    # Find the first longitudinal mean that is above threshold starting from each edge
    above_thresh = (mean_temp > threshold).astype('int')
    start = len(mean_temp) - len(np.trim_zeros(above_thresh, 'f'))
    end = - (len(mean_temp) - len(np.trim_zeros(above_thresh, 'b')))

    # If there are longitudinal means below temperature threshold in the middle
    # it is probably because there is a shift in lanes.
    below_thresh = ~ above_thresh.astype('bool')
    if sum(below_thresh[start:end]) == 0:
        return None

    if sum(below_thresh[start:end]) > 0:
        # Calculate splitting point between lanes
        (midpoint, ) = np.where(mean_temp[start:end] == min(mean_temp[start:end]))
        midpoint = midpoint[0] + start
        return (start, midpoint, end)
    return None
def estimate_road_width(pixels, threshold, adjust_left, adjust_right):
    """
    Estimate the road length of each transversal line (row) of the temperature
    heatmap data.
    """
    road_widths = []
    for idx in range(pixels.shape[0]):#looper over alle rækker i temperatures_trimmed
        start = _estimate_road_edge_right(pixels[idx, :], threshold)
        end = _estimate_road_edge_left(pixels[idx, :], threshold)
        road_widths.append((start + adjust_left, end - adjust_right))
    return road_widths

def _estimate_road_edge_right(line, threshold):
    cond = line < threshold #roadwidth_threshold: 50 # Threshold temperature used when estimating the road width.
    count = 0
    while True:
        if any(cond[count:count + 3]):
            count += 1
        else:
            break
    return count


def _estimate_road_edge_left(line, threshold):
    cond = line < threshold
    count = len(line)
    while True:
        if any(cond[count - 3:count]):
            count -= 1
        else:
            break
    return count

lane_result = detect_paving_lanes( temperatures_trimmed, config['lane_threshold'])# i road_identification
lane_start, lane_end = lane_result[config['lane_to_use']]
temperatures_trimmed = temperatures_trimmed.iloc[:, lane_start:lane_end]
# regner bredten af kørebanen i pixel
roadwidths = estimate_road_width( temperatures_trimmed.values, config['roadwidth_threshold'],
    config['roadwidth_adjust_left'],
    config['roadwidth_adjust_right']
)

def create_road_pixels(pixels_trimmed, roadwidths):
    road_pixels = np.zeros(pixels_trimmed.shape, dtype='bool')
    for idx, (road_start, road_end) in enumerate(roadwidths):
        road_pixels[idx, road_start:road_end] = 1
    return road_pixels
#laver en dataframe med True der hvor den varme vej er og False der hvor der er under 50 grader
road_pixels = create_road_pixels(temperatures_trimmed.values, roadwidths)

#%%  de her linjer forstår jeg ikke helt... 
def identify_roller_pixels(pixels, road_pixels, temperature_threshold):
    below_threshold = pixels < temperature_threshold
    roller_pixels = road_pixels & below_threshold
    return below_threshold, roller_pixels
aa, roller_pixels = identify_roller_pixels(temperatures_trimmed.values, road_pixels, config['roller_detect_temperature'])

def interpolate_roller_pixels(temperature_pixels, roller_pixels, road_pixels):
    non_roller_road_pixels = road_pixels & ~roller_pixels #gover vejen med høje temperaturer
    points = np.where(non_roller_road_pixels)
    values = temperature_pixels[points]
    points_interpolate = np.where(roller_pixels)
    # values_interpolate = griddata(points, values, points_interpolate, method='linear')
    # temperature_pixels[points_interpolate] = 200.0 # values_interpolate
    temperature_pixels[points_interpolate] = np.mean(temperature_pixels[points])

if config['roller_detect_interpolation']:
    interpolate_roller_pixels(temperatures_trimmed.values, roller_pixels, road_pixels)
    
#%% så kigges der på pixels som har en temperatur procent under det rullende gennemsnit
# dette gøres i detections:
#moving_average_pixels = detect_temperatures_below_moving_average( temperatures_trimmed, road_pixels, metadata,
    # percentage=config['moving_average_percent'], window_meters=config['moving_average_window'])
temperature_pixels = temperatures_trimmed.values 

#window der bruges i df.rolling regnes her. 
window_meters=config['moving_average_window']

def longitudinal_resolution(distance): #from utils.py
    t = distance.diff().describe() #kigger på skridt længden i distance kolonnen. .describe() giver statistik
    longitudinal_resolution = t['50%'] #tager medianen af skridtlængderne
    return longitudinal_resolution
#metadata= dataframe med resterne fra datasættet. metadata.distance = distance kolonnen
a = longitudinal_resolution(metadata.distance)
window = int(round(window_meters / a))#hvor mange pixel der er på 100 meter. 

#så regnes moving average i detections._calc_moving_average(df, road_pixels, window)
df = temperatures_trimmed.copy(deep=True)
df.values[~ road_pixels] = 'NaN' #sætter alt der ikke er på vejen lig NaN
min_periods = int(window * 0.80) # There must be 80% of <window> number of pixels
#her laves moving average af temperaturene. .mean() betyder at det er rullende gennemsnit
#center=True betyder at der tages 200 punkter før og 200 punkter efter. svarende til 50 meter med de værdier der er givet her. 
#roll across the rows, 
df_avg = df.rolling(window, center=True,min_periods=min_periods).mean()
moving_average = df_avg.mean(axis=1)
moving_average_values = np.tile(moving_average.values, (temperature_pixels.shape[1], 1)).T #array med mov_avg værdier i 26 kolonner
percentage=config['moving_average_percent'] 
ratio = percentage / 100
#Her undersøges det om temperatur værdierne er under 90% af moving average værdien. 
moving_average_pixels = temperature_pixels < (ratio * moving_average_values)
moving_average_pixels[~ road_pixels] = 0  # non-road pixels are not part of the detection

#%% Så kigges der på forskellen i temperatur
#detections.detect_high_gradient_pixels(temperature_pixels, roadwidths,  tolerance, diagonal_adjacency=True)
#gradient_pixels, clusters_raw = detect_high_gradient_pixels( temperatures_trimmed.values,roadwidths,
    # config['gradient_tolerance'], diagonal_adjacency=True)
def _detect_longitudinal_gradients(idx, offsets, temperatures, gradient_pixels, tolerance):
    start, end = offsets[idx]
    next_start, next_end = offsets[idx + 1]
    start = max(start, next_start)
    end = min(end, next_end)

    temperature_slice = temperatures[idx, start:end]
    temperature_slice_next = temperatures[idx + 1, start:end]

    (indices, ) = np.where(np.abs(temperature_slice - temperature_slice_next) > tolerance)
    indices += start
    gradient_pixels[idx, indices] = 1
    gradient_pixels[idx + 1, indices] = 1

    edges = _calc_edges(idx, indices, idx + 1, indices)
    return edges
def _calc_edges(rowidx1, colidx1, rowidx2, colidx2):
    edges = np.zeros((4, len(colidx2)))
    edges[0, :] = rowidx1
    edges[1, :] = colidx1
    edges[2, :] = rowidx2
    edges[3, :] = colidx2
    return edges
def _detect_transversal_gradients(idx, offsets, temperatures, gradient_pixels, tolerance):
    start, end = offsets[idx]
    temperature_slice = temperatures[idx, start:end]
    (indices, ) = np.where(np.abs(np.diff(temperature_slice)) > tolerance)
    indices += start
    gradient_pixels[idx, indices] = 1
    gradient_pixels[idx, indices + 1] = 1
    edges = _calc_edges(idx, indices, idx, indices + 1)
    return edges
def _detect_diagonal_gradients_left(idx, offsets, temperatures, gradient_pixels, tolerance):
    start, end = offsets[idx]
    next_start, next_end = offsets[idx + 1]


    if  next_start < start:
        new_start = start
        new_next_start = start - 1
    elif start < next_start:
        new_start = next_start + 1
        new_next_start = next_start
    elif start == next_start:
        new_start = start + 1
        new_next_start = next_start


    if next_end < end:
        new_end = next_end + 1
        new_next_end = next_end
    elif end < next_end:
        new_end = end
        new_next_end = end - 1
    elif end == next_end:
        new_end = end
        new_next_end = end - 1

    next_start = new_next_start
    next_end = new_next_end
    start = new_start
    end = new_end


    temperature_slice = temperatures[idx, start:end]
    temperature_slice_next = temperatures[idx + 1, next_start:next_end]

    (indices, ) = np.where(np.abs(temperature_slice - temperature_slice_next) > tolerance)
    gradient_pixels[idx, start:end][indices] = 1
    gradient_pixels[idx + 1, next_start:next_end][indices] = 1

    edges = _calc_edges(idx, indices + start, idx + 1, indices + next_start)
    return edges
def _detect_diagonal_gradients_right(idx, offsets, temperatures, gradient_pixels, tolerance):
    start, end = offsets[idx]
    next_start, next_end = offsets[idx + 1]

    if  next_start < start:
        new_start = start
        new_next_start = start + 1
    elif start < next_start:
        new_start = next_start - 1
        new_next_start = next_start
    elif start == next_start:
        new_start = start
        new_next_start = next_start + 1

    if next_end < end:
        new_end = next_end - 1
        new_next_end = next_end
    elif end < next_end:
        new_end = end
        new_next_end = end + 1
    elif end == next_end:
        new_end = end - 1
        new_next_end = next_end

    next_start = new_next_start
    next_end = new_next_end
    start = new_start
    end = new_end


    temperature_slice = temperatures[idx, start:end]
    temperature_slice_next = temperatures[idx + 1, next_start:next_end]

    (indices, ) = np.where(np.abs(temperature_slice - temperature_slice_next) > tolerance)

    gradient_pixels[idx, start:end][indices] = 1
    gradient_pixels[idx + 1, next_start:next_end][indices] = 1

    edges = _calc_edges(idx, indices + start, idx + 1, indices + next_start)
    return edges

gradient_pixels = np.zeros(temperatures_trimmed.values.shape, dtype='bool') #array med False værdier
edges = []
diagonal_adjacency=True
for idx in range(len(roadwidths) - 1): #loop over hele længden af kørebanen
    # Locate the temperature gradients in the driving direction
    edges.append(_detect_longitudinal_gradients(idx, roadwidths, temperatures_trimmed.values, gradient_pixels, config['gradient_tolerance']))
    # Locate the temperature gradients in the transversal direction
    edges.append(_detect_transversal_gradients(idx, roadwidths, temperatures_trimmed.values, gradient_pixels, config['gradient_tolerance']))

    if diagonal_adjacency:
        edges.append(_detect_diagonal_gradients_right(idx, roadwidths, temperatures_trimmed.values, gradient_pixels, config['gradient_tolerance']))
        edges.append(_detect_diagonal_gradients_left(idx, roadwidths, temperatures_trimmed.values, gradient_pixels, config['gradient_tolerance']))

edges.append(_detect_transversal_gradients(idx + 1, roadwidths, temperatures_trimmed.values, gradient_pixels, config['gradient_tolerance']))

#%%Så bruges en graph modul to find clusters of temperature changes
import networkx as nx
def _create_gradient_graph(edges):
    edge_iterator = _iter_edges(edges)
    gradient_graph = nx.from_edgelist(edge_iterator)
    return gradient_graph
def _iter_edges(edges):
    for edge_array in edges:
        for idx in range(edge_array.shape[1]):
            row1, col1, row2, col2 = edge_array[:, idx]
            yield (row1, col1), (row2, col2)
def _extract_clusters(graph):
    for cluster in sorted(nx.connected_components(graph), key=len, reverse=True):
        cluster = [(int(row), int(col)) for row, col in cluster]
        yield cluster
        
gradient_graph = _create_gradient_graph(edges)#dette giver et graph object.. 
clusters_raw = list(_extract_clusters(gradient_graph)) #giver de clusters der er i graph (?)

#%% # Plot af data efter trimming. Det er plot af de resultater det laves ved cleaning længere oppe 
#data.create_trimming_result_pixels(pixels_raw, trim_result, lane_result, roadwidths, roller_pixels, config)
#pixel_category = create_trimming_result_pixels(temperatures.values, trim_result, lane_result[config['lane_to_use']],
    # roadwidths, roller_pixels, config)
pixel_category = np.zeros(temperatures.values.shape, dtype='int')
trim_col_start, trim_col_end, trim_row_start, trim_row_end = trim_result #dette kommer fra at skære siderne af vejen
lane_start, lane_end = lane_result[config['lane_to_use']]
view = pixel_category[trim_row_start:trim_row_end, trim_col_start:trim_col_end]

figures = {}

from matplotlib.colors import ListedColormap
for longitudinal_idx, (road_start, road_end) in enumerate(roadwidths):
    view[:, lane_start:lane_end][longitudinal_idx, road_start:road_end] = 1

if config['roller_detect_enabled']:
    view[roller_pixels] = 2
    
def _iter_segments(temperatures, number_of_segments):
    if int(number_of_segments) == 1:
        yield '', (0, len(temperatures))
        return
def plot_cleaning_results(config, metadata, temperatures, pixel_category):
    titles = {
        'main': config['title'] + " - data cleaning",
        'temperature_title': "raw temperature data",
        'category_title': "road detection results"
    }
    categories = ['non-road', 'road', 'roller']
    return _plot_heatmaps(
        titles,
        metadata,
        config['pixel_width'],
        temperatures.values,
        pixel_category,
        categories
    )
def _plot_heatmaps(titles, metadata, pixel_width, pixel_temperatures, pixel_category, categories):
    fig_heatmaps, (ax1, ax2) = plt.subplots(ncols=2)
    fig_heatmaps.subplots_adjust(wspace=0.6)
    fig_heatmaps.suptitle(titles['main'])

    # Plot the raw data
    ax1.set_title(titles['temperature_title'])  # 'Raw data'
    _temperature_heatmap(ax1, pixel_temperatures,
                         metadata.distance, pixel_width)
    ax1.set_ylabel('chainage [m]')

    # Plot that shows identified road and high gradient pixels
    ax2.set_title(titles['category_title'])  # 'Estimated high gradients'
    plt.figure(num=fig_heatmaps.number)
    _categorical_heatmap(ax2, pixel_category, metadata.distance,
                        pixel_width, categories)
    return fig_heatmaps

def _temperature_heatmap(ax, pixels, distance, pixel_width):
    """Make a heatmap of the temperature columns in the dataframe."""
    mat = ax.imshow(
        pixels,
        aspect="auto",
        cmap='RdYlGn_r',
        extent=_create_extent(distance, pixels, pixel_width) #floats (left, right, bottom, top), (0, pixels.shape[1] * pixel_width, distance.iloc[-1], distance.iloc[0])
    )
    plt.colorbar(mat, ax=ax, label='Temperature [C]')


def _create_extent(distance, pixels, pixel_width):
    return (0, pixels.shape[1] * pixel_width, distance.iloc[-1], distance.iloc[0])
def _categorical_heatmap(ax, pixels, distance, pixel_width, categories):
    # set limits .5 outside true range
    colors = ["dimgray", "firebrick", "springgreen"]
    mat = ax.imshow(
        pixels,
        aspect='auto',
        vmin=np.min(pixels) - .5,
        vmax=np.max(pixels) + .5,
        cmap=ListedColormap(colors[:len(categories)]),
        extent=_create_extent(distance, pixels, pixel_width)
    )
    # tell the colorbar to tick at integers
    cbar = plt.colorbar(mat, ax=ax, ticks=np.arange(
        np.min(pixels), np.max(pixels) + 1))
    cbar.set_ticklabels(categories)
#Herunder laves selve plottet
for k, (start, end) in _iter_segments(temperatures, config['plotting_segments']):
    kwargs = {
        'config': config,
        'metadata': metadata.iloc[start:end, :],
        'temperatures': temperatures.iloc[start:end, :],
        'pixel_category': pixel_category[start:end, :],
        }
    figures[f'fig_cleanup{k}_'] = plot_cleaning_results(**kwargs)

#%% # Plot af detections results
#Der plottes både en figur med rå data og moving average data hvis dette er sæået til i jobs filen
# og der plottes gradient detection hvis dette er slået til 
def plot_detections(k, figures, **kwargs):
    config = kwargs['config']

    ## Perform and plot moving average detection results along with the trimmed temperature data
    if config['moving_average_enabled']:
        figures[f'moving_average{k}'] = _plot_moving_average_detection(**kwargs)

    ## Perform and plot gradient detection results along with the trimmed temperature data
    if config['gradient_enabled']:
        figures[f'gradient{k}'] = _plot_gradient_detection(**kwargs)
        
def _plot_moving_average_detection(moving_average_pixels, config, temperatures_trimmed, road_pixels, metadata, **_kwargs):
    titles = {
        'main': config['title'] + " - moving average",
        'temperature_title': "Temperatures",
        'category_title': "Moving average detection results"
    }
    categories = ['non-road', 'road', 'detections']
    pixel_category = create_detect_result_pixels(
        temperatures_trimmed.values,
        road_pixels,
        moving_average_pixels
    )
    return _plot_heatmaps(
        titles,
        metadata,
        config['pixel_width'],
        temperatures_trimmed.values,
        pixel_category,
        categories
    )
def create_detect_result_pixels(pixels_trimmed, road_pixels, detection_pixels): #fra data modul
    pixel_category = np.zeros(pixels_trimmed.shape, dtype='int')
    pixel_category[~ road_pixels] = 1
    pixel_category[road_pixels] = 2
    pixel_category[detection_pixels] = 3
    return pixel_category
def _plot_gradient_detection(gradient_pixels, config, temperatures_trimmed, metadata, road_pixels, **_kwargs):
    titles = {
        'main': config['title'] + " - gradient",
        'temperature_title': "Temperatures",
        'category_title': "gradient detection results"
    }
    categories = ['non-road', 'road', 'detections']
    pixel_category = create_detect_result_pixels(
        temperatures_trimmed.values,
        road_pixels,
        gradient_pixels
    )
    return _plot_heatmaps(
        titles,
        metadata,
        config['pixel_width'],
        temperatures_trimmed.values,
        pixel_category,
        categories
    )
## her laves plottet
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
#%% Der laves statistik på gradient detection og dette plottes også 
# Plot statistics in relating to the gradient detection algorithm
import seaborn as sns

def plot_statistics(title, temperatures, roadwidths, road_pixels, tolerance): #fra plotting.py
    tol_start, tol_end, tol_step = tolerance
    tolerances = np.arange(tol_start, tol_end, tol_step)

    fig_stats, (ax1, ax2) = plt.subplots(ncols=2)
    fig_stats.suptitle(title)

    # Plot showing the percentage of road that is comprised of high gradient pixels for a given gradient tolerance
    high_gradients = _calculate_tolerance_vs_percentage_high_gradient(
        temperatures, roadwidths, road_pixels, tolerances)
    ax1.set_title('Percentage high gradient as a function of tolerance')
    ax1.set_xlabel('Threshold temperature difference [C]')
    ax1.set_ylabel('Percentage of road whith high gradient.')
    sns.lineplot(x=tolerances, y=high_gradients, ax=ax1)

    # Plot showing histogram of road temperature
    ax2.set_title('Road temperature distribution')
    ax2.set_xlabel('Temperature [C]')
    distplot_data = temperatures.values[road_pixels]
    sns.histplot(distplot_data, color="m", ax=ax2,
                 stat='density', discrete=True, kde=True)
    return fig_stats

def _calculate_tolerance_vs_percentage_high_gradient(temperatures, roadwidths, road_pixels, tolerances):
    percentage_high_gradients = list()
    nroad_pixels = road_pixels.sum()
    for tolerance in tolerances:
        gradient_pixels, _ = detect_high_gradient_pixels(
            temperatures.values, roadwidths, tolerance)
        percentage_high_gradients.append(
            (gradient_pixels.sum() / nroad_pixels) * 100)
    return percentage_high_gradients

def detect_high_gradient_pixels(temperature_pixels, roadwidths,  tolerance, diagonal_adjacency=True): #fra detections.py
    """
    Return a boolean array the same size as `df_temperature` indexing all pixels
    having higher gradients than what is supplied in `config.gradient_tolerance`.
    The `roadwidths` list contains the road section identified for each transversal
    line (row) in the data.
    """
    gradient_pixels = np.zeros(temperature_pixels.shape, dtype='bool')
    edges = []

    for idx in range(len(roadwidths) - 1):
        # Locate the temperature gradients in the driving direction
        edges.append(_detect_longitudinal_gradients(idx, roadwidths, temperature_pixels, gradient_pixels, tolerance))

        # Locate the temperature gradients in the transversal direction
        edges.append(_detect_transversal_gradients(idx, roadwidths, temperature_pixels, gradient_pixels, tolerance))

        if diagonal_adjacency:
            edges.append(_detect_diagonal_gradients_right(idx, roadwidths, temperature_pixels, gradient_pixels, tolerance))
            edges.append(_detect_diagonal_gradients_left(idx, roadwidths, temperature_pixels, gradient_pixels, tolerance))

    edges.append(_detect_transversal_gradients(idx + 1, roadwidths, temperature_pixels, gradient_pixels, tolerance))

    gradient_graph = _create_gradient_graph(edges)
    clusters_raw = list(_extract_clusters(gradient_graph))
    return gradient_pixels, clusters_raw
#Her plottes faktisk
if config['gradient_statistics_enabled']: #gradient_statistics_enabled: True # Whether or not to calculate and plot gradient statistics
    figures['stats_'] = plot_statistics(
        title,
        temperatures_trimmed,
        roadwidths,
        road_pixels,
        config['tolerance']
    )
    
#%% Her plottes med den nye funktion hvor man kan slå den tidskrævende del til og fra

# figures['stats_gradient'] = nrn_functions.plot_statistics_gradientPlot(title,temperatures_trimmed,roadwidths,road_pixels,config['tolerance'])

x_lower = np.min(temperatures_trimmed.values[road_pixels]); x_higher = np.max(temperatures_trimmed.values[road_pixels])
figures['stats_tempDist'] = nrn_functions.plot_statistics_TempDistributionPlot(title, temperatures_trimmed, road_pixels, limits=[x_lower,x_higher]) 


#%% ###### Så laves output der kan bruges i map script
#dette gøres igennem export.py
import os
#%% rå temp data kun på vejen og resten hedder NaN
#file path kommer fra 
# file_path = config['file_path']. kommer fra starten af scriptet

def temperature_to_csv(file_path, temperatures, metadata, road_pixels):
    temperatures = temperatures.copy()
    filename = os.path.basename(file_path)
    temperatures.values[~road_pixels] = 'NaN'
    df = merge_temperature_data(metadata, temperatures)
    df.to_csv(f"gradient_{filename}.csv")
    
def merge_temperature_data(df_temp, df_rest): #fra utils
    """
    Merge two dataframes containing temperature data and the rest, respectively, into a single dataframe.
    """
    return pd.merge(df_temp, df_rest, how='inner', copy=True, left_index=True, right_index=True)
#Denne gemmer en fil med navnet gradient + filnavnet på input filen
#den gemmer de trimmed_temperatures sat sammen med meta data igen (dist, etc)
# alt der ikke er inden for road får NaN. Så faktisk ikke noget med gradient at gøre .... 
temperature_to_csv(file_path, temperatures_trimmed, metadata, road_pixels)


#%% #heri gemmes resultatet fra sammenligning af temperatur og moving average for hver pixel. Den sættes til 2 hvis temp < 90%*moving average.  
def detections_to_csv(file_path, detection_type, temperatures, metadata, road_pixels, detected_pixels):
    data_width = temperatures.values.shape[1]
    temperatures = temperatures.copy()
    filename = os.path.basename(file_path)
    temperatures.values[~road_pixels] = 0
    temperatures.values[road_pixels] = 1
    temperatures.values[detected_pixels & road_pixels] = 2
    temperatures[f'percentage_{detection_type}'] = 100 * (np.sum(detected_pixels & road_pixels, axis=1) / data_width)
    df = merge_temperature_data(metadata, temperatures)
    df.to_csv(f"area_{detection_type}_{filename}.csv")
    
detections_to_csv( file_path, 'moving_avg', temperatures_trimmed, metadata, road_pixels, moving_average_pixels)

#%% Her gemmes de steder hvor der er en gradient forskel også angivet med 2 taller 
detections_to_csv( file_path, 'gradient', temperatures_trimmed, metadata, road_pixels, gradient_pixels)
#%% gemmer den gennemsnitlige temperatur for hver række på vejen. altså en temp for hver distance 
def temperature_mean_to_csv(file_path, temperatures, road_pixels):
    temperatures = temperatures.copy()
    filename = os.path.basename(file_path)
    temperatures.values[~road_pixels] = 'NaN'
    temperatures['temperature_sum'] = np.nanmean(temperatures.values, axis=1)
    df = temperatures[['temperature_sum']]
    df.to_csv(f"distribution_{filename}.csv")
    
temperature_mean_to_csv(file_path, temperatures_trimmed, road_pixels)
#%%

def detections_MA_D_to_csv(file_path, detection_type, temperatures, metadata, road_pixels, detected_pixels):
    # detection_type='moving_avg'
    data_width1 = temperatures.values.shape[1] #antal kolonner
    temperatures1 = temperatures.copy()
    filename = os.path.basename(file_path)
    temperatures1.values[~road_pixels] = 0 #0 der hvor der ikke er en vej
    temperatures1.values[road_pixels] = 1 # 1 der hvor der er en vej 
    temperatures1.values[detected_pixels & road_pixels] = 2 #2 der hvor der er under 90% moving average
    #detected_pixels & road_pixels = True hvis begge er true. så sum tæller antal true for hver række.
    # delt med data_width1 giver procentdel true for den række på vejen. *100 giver det i %
    temperatures1[f'percentage_{detection_type}'] = 100 * (np.sum(detected_pixels & road_pixels, axis=1) / data_width1) #summen af true i moving average og road
    df2 = merge_temperature_data(metadata, temperatures1) #df med temperature i 0,1,2 og %mæssig dele af vejen der er under MA krav. for hver row. 
    # df2.to_csv(f"area_{detection_type}_{filename}.csv")
    
    
    temperatures2 = temperatures.copy()
    filename = os.path.basename(file_path)
    temperatures2.values[~road_pixels] = 'NaN'
    temperatures2['temperature_sum'] = np.nanmean(temperatures2.values, axis=1) 
    df1 = temperatures2[['temperature_sum']]#gennemsnits værdi af temperatur for hver række. kun på vejen, resten er NaN
    # df1.to_csv(f"distribution_res_{filename}.csv")
    Summary_merged=calculate_Moving_average_results(df1,df2,filename)
    # Summary_merged['Pixels']=Pixels
    Summary_merged.to_csv(f"{detection_type}_{filename}.csv",index=False,sep=';')
    # Summary.to_csv(f"distribution_results_{filename}.csv",index=False,sep=';') 
    print('\n Moving Average Results [%]:',Summary_merged['Moving Average Results [%]'][0])
    
# detections_MA_D_to_csv(file_path,'moving_avg_results', temperatures_trimmed, metadata, road_pixels, moving_average_pixels)

#Herunder er koden skrevet ud...
temperatures = temperatures_trimmed
data_width1 = temperatures.values.shape[1]
temperatures1 = temperatures.copy()
filename = os.path.basename(file_path)
temperatures1.values[~road_pixels] = 0
temperatures1.values[road_pixels] = 1
temperatures1.values[moving_average_pixels & road_pixels] = 2
detection_type = 'moving_avg_results'
b = moving_average_pixels & road_pixels
a = np.sum(moving_average_pixels & road_pixels, axis=1)
aa = a/data_width1
temperatures1[f'percentage_{detection_type}'] = 100 * (np.sum( moving_average_pixels & road_pixels, axis=1) / data_width1)
df2 = merge_temperature_data(metadata, temperatures1)
temperatures2 = temperatures.copy()
filename = os.path.basename(file_path)
temperatures2.values[~road_pixels] = 'NaN'
temperatures2['temperature_sum'] = np.nanmean(temperatures2.values, axis=1) 
df1 = temperatures2[['temperature_sum']]#gennemsnits værdi af temperatur for hver række. kun på vejen, resten er NaN
# så bruges calculate moving averag eresults. Den er lang. Summary_merged=calculate_Moving_average_results(df1,df2,filename)

#%% DET MATTEOS SCRIPT GØR 
#moving_average csv filen regnes og gemmes i df
data_width = temperatures.values.shape[1]
temperatures = temperatures.copy()
filename = os.path.basename(file_path)
temperatures.values[~road_pixels] = 0
temperatures.values[road_pixels] = 1
temperatures.values[moving_average_pixels & road_pixels] = 2
temperatures[f'percentage_{detection_type}'] = 100 * (np.sum(moving_average_pixels & road_pixels, axis=1) / data_width)
df = merge_temperature_data(metadata, temperatures)

#%%
# Det den dem af MAP script der finder intervaller
#den bruger summen for hver row som input fil (den der hedder disttibution)
df = nrn_functions.temperature_mean_to_csv(temperatures_trimmed, road_pixels)
#% Select only rows with temperature above 80.
df = df[df['temperature_sum']>80]
IRfiles={m:[] for m in range(20)} #laver 20 tomme dataframes
bins_list=[]
ref=200
scale=1
for m in range(20): #for hver af dataframsne
    min=0+m
    max=200+m
    bins2 = [p/scale for p in range(min, max,20)]
    bins_list.append(bins2)
    IRfiles[m] = df['temperature_sum'].value_counts(bins=bins2)
    #problemet er at cut giver catagoric dTypes. bruger value_count istedet
    # IRfiles[m]=pd.cut(x= df['temperature_sum'], bins=bins2, include_lowest=True).value_counts()
    IRfiles[m]=IRfiles[m].to_frame()
    IRfiles[m].reset_index(inplace=True)
    IRfiles[m]['temperature_sum'] = IRfiles[m][IRfiles[m].columns[1]]
    
    IRfiles[m]['Percentage []']=[IRfiles[m]['temperature_sum'][x]/IRfiles[m]['temperature_sum'].sum()*100 for x in range(int(len(IRfiles[m])))]
Results=pd.concat(IRfiles,axis=0).sort_values(by=['index'])
Maks=Results.loc[Results['Percentage []'].idxmax()]

#%%
# Så gøres det for intervaller der er 10 grader
IRfiles3={m:[] for m in range(10)}
bins_list=[]
for m in range(40):
    min=0+m
    max=200+m
    bins3 = [p/scale for p in range(min, max, 10)]
    bins_list.append(bins3)
    IRfiles3[m]=pd.cut(x= df['temperature_sum'], bins=bins3, include_lowest=True).value_counts()
    IRfiles3[m]=IRfiles3[m].to_frame()
    IRfiles3[m].reset_index(inplace=True)
    
    IRfiles3[m]['Percentage [%]']=[IRfiles3[m]['temperature_sum'][x]/IRfiles3[m]['temperature_sum'].sum()*100 for x in range(int(len(IRfiles3[m])))]
    IRfiles3[m]['Percentage [%] 10C gap']=IRfiles3[m]['Percentage [%]']
    IRfiles3[m]['Temperature [°C] 10C gap']=IRfiles3[m]['temperature_sum']
    del IRfiles3[m]['temperature_sum']
    del IRfiles3[m]['Percentage [%]']
Results3=pd.concat(IRfiles3,axis=0).sort_values(by=['index'])
Maks3=Results3.loc[Results3['Percentage [%] 10C gap'].idxmax()]

#Så gøres det for intervaller der er 30 grader
IRfiles4={m:[] for m in range(30)}
bins_list=[]
for m in range(30):
    min=0+m
    max=200+m
    bins3 = [p/scale for p in range(min, max, 30)]
    bins_list.append(bins3)
    IRfiles4[m]=pd.cut(x= df['temperature_sum'], bins=bins3, include_lowest=True).value_counts()
    IRfiles4[m]=IRfiles4[m].to_frame()
    IRfiles4[m].reset_index(inplace=True)
    IRfiles4[m]['Percentage [%]']=[IRfiles4[m]['temperature_sum'][x]/IRfiles4[m]['temperature_sum'].sum()*100 for x in range(int(len(IRfiles4[m])))]
    IRfiles4[m]['Percentage [%] 30C gap']=IRfiles4[m]['Percentage [%]']
    IRfiles4[m]['Temperature [°C] 30C gap']=IRfiles4[m]['temperature_sum']
    del IRfiles4[m]['temperature_sum']
    del IRfiles4[m]['Percentage [%]']
Results4=pd.concat(IRfiles4,axis=0).sort_values(by=['index'])
Maks4=Results4.loc[Results4['Percentage [%] 30C gap'].idxmax()]

#MA ratio
number_1 = np.count_nonzero(road_pixels)
#count all elements in dataframe
number_2 = np.count_nonzero(moving_average_pixels)
#ratio 
ratio = number_2/number_1

#%% gemmer som MAP output
a = filename[:-4].split('_')
summary_df = pd.DataFrame({'Entreprise':a[4], 'Contractor':a[2], 'Mix Type':a[0],
                           'Road ID':a[5], 'Date':a[1], 'Device':a[3],
                           'Number of Pavers':a[7], 'Position of Paver':a[8],
                           'Road segment':a[6]}, index=range(0,1) )
summary_df['Moving Average Results [%]'] = np.round(ratio*100,2)
summary_df['10°C gap'] = Maks3.loc['index']
summary_df['Percent with 10°C gap'] = np.round(Maks3.loc['Percentage [%] 10C gap'],2) 
summary_df['20°C gap'] = Maks.loc['index']
summary_df['Percent with 20°C gap'] = np.round(Maks.loc['Percentage [%]'],2)
summary_df['30°C gap'] = Maks4.loc['index']
summary_df['Percent with 30°C gap'] = np.round(Maks4.loc['Percentage [%] 30C gap'],2)

#%%
statistics_dataframe = nrn_functions.summary_as_MAP(temperatures_trimmed, road_pixels, moving_average_pixels, filename=filename)