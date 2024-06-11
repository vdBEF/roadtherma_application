# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:11:20 2024

@author: NRN
"""
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
from plotting import _calculate_tolerance_vs_percentage_high_gradient

def heatmaps_temperature_pixels(ax, pixels, distance, pixel_width, include_colorbar=True):
    """
    Funktion til at plotte heatmaps i toppen af streamlit koden. 
    Kopier af original road therma funktioner. 
    
    Make a heatmap of the temperature columns in the dataframe.
    
    ax: den aksel på et subplot figuren tilgår 
    pixels: den dataframe der skal vises på heatmappet. Indeholder temperatur værdier der bruges til
        farvekoden og vejen opløst i pixels
    distance: serie med distance værdierne der passer til pixels dataframen
    pixel_width: angiver bredten af en pixel. 
    """
    mat = ax.imshow(
        pixels,
        aspect="auto",
        cmap='RdYlGn_r',
        extent=(0, pixels.shape[1] * pixel_width, distance.iloc[-1], distance.iloc[0]) #floats (left, right, bottom, top), (0, pixels.shape[1] * pixel_width, distance.iloc[-1], distance.iloc[0])
    )
    if include_colorbar==True:
        plt.colorbar(mat, ax=ax, label='Temperature [C]')
    
def heatmap_identified_road(ax, pixel_category, distance, pixel_width, categories):
    """
    Figur der minder om   plot_cleaning_results(config, metadata, temperatures, pixel_category), men designet til illustrationer
    _categorical_heatmap(ax, pixels, distance, pixel_width, categories)
    
    resultatet kan sættes ind på en af subplots exkser
    
    ax: den aksel på et subplot figuren tilgår 
    pixel_category : dataframe der har de forskellige ting man vil illustrere givet ved kategorier
    distance: serie med distance værdierne der passer til pixels dataframen
    pixel_width: angiver bredten af en pixel. 
    
    """
    colors = ["dimgray", "firebrick", "springgreen"]
    mat = ax.imshow(
        pixel_category,
        aspect='auto',
        vmin=np.min(pixel_category) - .5, 
        vmax=np.max(pixel_category) + .5,
        cmap=ListedColormap(colors[:len(categories)]),
        extent=(0, pixel_category.shape[1] * pixel_width, distance.iloc[-1], distance.iloc[0])
    )
    # tell the colorbar to tick at integers
    cbar = plt.colorbar(mat, ax=ax, ticks=np.arange(
        np.min(pixel_category), np.max(pixel_category) + 1))
    cbar.set_ticklabels(categories)
    
#%## plot af kun identificeret vej og ikke med roller på
def create_identified_road_pixels(pixels_raw, trim_result, lane_result, roadwidths):
    """
    som create_trimming_result_pixels, bare kun med vej og ikke vej
    """
    pixel_category = np.zeros(pixels_raw.shape, dtype='int')
    trim_col_start, trim_col_end, trim_row_start, trim_row_end = trim_result
    lane_start, lane_end = lane_result
    view = pixel_category[trim_row_start:trim_row_end, trim_col_start:trim_col_end]

    for longitudinal_idx, (road_start, road_end) in enumerate(roadwidths):
        view[:, lane_start:lane_end][longitudinal_idx, road_start:road_end] = 1

    return pixel_category


def temperature_to_csv( temperatures, metadata, road_pixels):
    """
    Denne funktion retunerer de rå temperatur data kun for de pixels der er vejen. 
    Kombineres med metadata også
    
    Retunere den dataframe der skal gemmes som csv. 
    

    """
    temperatures = temperatures.copy()
    temperatures.values[~road_pixels] = 'NaN'
    df = pd.merge(metadata,temperatures, how='inner', copy=True, left_index=True, right_index=True)
    return df 

def detections_to_csv( temperatures, metadata, road_pixels, detected_pixels):
    """
    Denne funktion retunerer en dataframe med 0 for ikke vej pixel,
    1 for vej pixels og 2 for dedekterede pixels
    
    temperatures: den trimmede temperatur df
    metadata: df med meta data
    road_pixels: df med True der hvor der er vej
    detect_pixels: df med True der hvor der er dedekteret
    """
    data_width = temperatures.values.shape[1]
    temperatures = temperatures.copy()
    temperatures.values[~road_pixels] = 0
    temperatures.values[road_pixels] = 1
    temperatures.values[detected_pixels & road_pixels] = 2
    temperatures[f'percentage'] = 100 * (np.sum(detected_pixels & road_pixels, axis=1) / data_width)
    df = pd.merge(metadata, temperatures , how='inner', copy=True, left_index=True, right_index=True)
    return df 

def temperature_mean_to_csv(temperatures, road_pixels):
    """
    Funktion der retunere dataframe med gennemsnits temperatur for hver distance for vejen. 
    """
    temperatures = temperatures.copy()
    temperatures.values[~road_pixels] = 'NaN'
    temperatures['temperature_sum'] = np.nanmean(temperatures.values, axis=1)
    df = temperatures[['temperature_sum']]
    return df


def plot_statistics_gradientPlot(title, temperatures, roadwidths, road_pixels, tolerance):
    """
    Tager udgangspunkt i plotting.plot_statistics , men dele op så de to plots er hver for sig. 
    Den ene er meget tidskrævende og skal derfor kunne slås fra. Denne del 
    plotter "Percentage high gradient as a function of tolerance" 

    """
    tol_start, tol_end, tol_step = tolerance
    tolerances = np.arange(tol_start, tol_end, tol_step)

    fig_stats, ax1 = plt.subplots()
    ax1.set_title(title)

    # Plot showing the percentage of road that is comprised of high gradient pixels for a given gradient tolerance
    high_gradients = _calculate_tolerance_vs_percentage_high_gradient(
        temperatures, roadwidths, road_pixels, tolerances)
    ax1.set_title('Percentage high gradient as a function of tolerance')
    ax1.set_xlabel('Threshold temperature difference [C]')
    ax1.set_ylabel('Percentage of road whith high gradient.')
    sns.lineplot(x=tolerances, y=high_gradients, ax=ax1)

    return fig_stats

def plot_statistics_TempDistributionPlot(title, temperatures, road_pixels, limits):
    """
    Tager udgangspunkt i plotting.plot_statistics , men dele op så de to plots er hver for sig. 
    Den ene er meget tidskrævende og skal derfor kunne slås fra. 
    Denne del plotter "Road temperature distribution"
    
    limits = [x_lower, x_higher]
    histplot: kde=False gør st den er en del hurtigere
    """

    fig_stats, ax1 = plt.subplots()
    ax1.set_title(title)

    # Plot showing histogram of road temperature
    ax1.set_title('Road temperature distribution')
    ax1.set_xlabel('Temperature [C]')
    distplot_data = temperatures.values[road_pixels]
    sns.histplot(distplot_data, color="m", ax=ax1,
                 stat='density', discrete=True, kde=False)
    ax1.set_xlim(limits)
    return fig_stats

def summary_as_MAP(temperatures_trimmed, road_pixels, moving_average_pixels, filename):
    """
    Function which create the same summary statistics at MAP old script
    
    #Hvert datasæt i dataframe gennemgås. Det undersøges hvor mange datapunkter der er i forskellig etemperatur intervaller med fast længde på X grader. 
    #Der laves således at intervallerne er løbende, eg. for interval af X=20: 0-20, 1-21, 2-22... 
    #Dette samles i resultat dataframe. Derefter findes det interval med flest datapunkter og denne gemmes i Maks og senere summary_df 

    Derudover udregnes ratio af pixels på vejen der er detected i moving average metoden 
    13-02-2024: Streamlit laver fejl når man bruger pd.cut() da den ikke kan arbejde med Catagorial dTypes
    Derfor laves koden lidt om så den bruger value_counts() istedet. 
    
    """
    df = temperature_mean_to_csv(temperatures_trimmed, road_pixels)
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

    # Så gøres det for intervaller der er 10 grader
    IRfiles3={m:[] for m in range(10)}
    bins_list=[]
    for m in range(40):
        min=0+m
        max=200+m
        bins3 = [p/scale for p in range(min, max, 10)]
        bins_list.append(bins3)
        IRfiles3[m] = df['temperature_sum'].value_counts(bins=bins3)
        #problemet er at cut giver catagoric dTypes. bruger value_count istedet
        # IRfiles3[m]=pd.cut(x= df['temperature_sum'], bins=bins3, include_lowest=True).value_counts()
        IRfiles3[m]=IRfiles3[m].to_frame()
        IRfiles3[m].reset_index(inplace=True)
        IRfiles3[m]['temperature_sum'] = IRfiles3[m][IRfiles3[m].columns[1]]
        
        IRfiles3[m]['Percentage []']=[IRfiles3[m]['temperature_sum'][x]/IRfiles3[m]['temperature_sum'].sum()*100 for x in range(int(len(IRfiles3[m])))]
        IRfiles3[m]['Percentage [] 10C gap']=IRfiles3[m]['Percentage []']
        IRfiles3[m]['Temperature [] 10C gap']=IRfiles3[m]['temperature_sum']
        del IRfiles3[m]['temperature_sum']
        del IRfiles3[m]['Percentage []']
    Results3=pd.concat(IRfiles3,axis=0).sort_values(by=['index'])
    Maks3=Results3.loc[Results3['Percentage [] 10C gap'].idxmax()]
    
    #Så gøres det for intervaller der er 30 grader
    IRfiles4={m:[] for m in range(30)}
    bins_list=[]
    for m in range(30):
        min=0+m
        max=200+m
        bins3 = [p/scale for p in range(min, max, 30)]
        bins_list.append(bins3)
        IRfiles4[m] = df['temperature_sum'].value_counts(bins=bins3)
        #problemet er at cut giver catagoric dTypes. bruger value_count istedet
        # IRfiles4[m]=pd.cut(x= df['temperature_sum'], bins=bins3, include_lowest=True).value_counts()
        IRfiles4[m]=IRfiles4[m].to_frame()
        IRfiles4[m].reset_index(inplace=True)
        IRfiles4[m]['temperature_sum'] = IRfiles4[m][IRfiles4[m].columns[1]]
        
        IRfiles4[m]['Percentage []']=[IRfiles4[m]['temperature_sum'][x]/IRfiles4[m]['temperature_sum'].sum()*100 for x in range(int(len(IRfiles4[m])))]
        IRfiles4[m]['Percentage [] 30C gap']=IRfiles4[m]['Percentage []']
        IRfiles4[m]['Temperature [] 30C gap']=IRfiles4[m]['temperature_sum']
        del IRfiles4[m]['temperature_sum']
        del IRfiles4[m]['Percentage []']
    Results4=pd.concat(IRfiles4,axis=0).sort_values(by=['index'])
    Maks4=Results4.loc[Results4['Percentage [] 30C gap'].idxmax()]

    #MA ratio
    number_1 = np.count_nonzero(road_pixels)
    #count all elements in dataframe
    number_2 = np.count_nonzero(moving_average_pixels)
    #ratio 
    ratio = number_2/number_1
    
    #gemmer som MAP output
    a = filename[:-4].split('_')
    summary_df = pd.DataFrame({'Entreprise':a[4], 'Contractor':a[2], 'Mix Type':a[0],
                               'Road ID':a[5], 'Date':a[1], 'Device':a[3],
                               'Number of Pavers':a[7], 'Position of Paver':a[8],
                               'Road segment':a[6]}, index=range(0,1) )
    summary_df['Moving Average Results [%]'] = np.round(ratio*100,1)
    summary_df['10 degrees gap'] = Maks3.loc['index']
    summary_df['Percent with 10 degrees gap'] = np.round(Maks3.loc['Percentage [] 10C gap'],2) 
    summary_df['20 degrees gap'] = Maks.loc['index']
    summary_df['Percent with 20 degrees gap'] = np.round(Maks.loc['Percentage []'],2)
    summary_df['30 degrees gap'] = Maks4.loc['index']
    summary_df['Percent with 30 degrees gap'] = np.round(Maks4.loc['Percentage [] 30C gap'],2)


    return summary_df
