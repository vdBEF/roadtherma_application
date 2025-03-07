import numpy as np
from scipy.interpolate import griddata

from utils import split_temperature_data, merge_temperature_data


def clean_data(temperatures, metadata, config):
    """
    Clean and prepare data by running cleaning routines contained in this module, i.e.,
        - trim_temperature_data
        - detect_paving_lanes
        - estimate_road_width

    and return the results of these operations, together with a trimmed version of the
    temperature data.
    """
    if config['autotrim_enabled']:
        trim_result = trim_temperature_data(
            temperatures.values,
            config['autotrim_temperature'],
            config['autotrim_percentage']
        )
    else:
        trim_result = crop_temperature_data(temperatures, metadata, config)

    column_start, column_end, row_start, row_end = trim_result
    temperatures_trimmed = temperatures.iloc[row_start:row_end, column_start:column_end]

    lane_result = detect_paving_lanes(
        temperatures_trimmed,
        config['lane_threshold']
    )

    lane_start, lane_end = lane_result[config['lane_to_use']]
    temperatures_trimmed = temperatures_trimmed.iloc[:, lane_start:lane_end]
    roadwidths = estimate_road_width(
        temperatures_trimmed.values,
        config['roadwidth_threshold'],
        config['roadwidth_adjust_left'],
        config['roadwidth_adjust_right']
    )
    return temperatures_trimmed, trim_result, lane_result, roadwidths


def crop_temperature_data(temperatures, metadata, config):
    width = config['pixel_width']
    length = len(temperatures.columns)
    transversal = np.arange(0, length * width, width)
    longi_start = config['manual_trim_longitudinal_start']
    longi_end = config['manual_trim_longitudinal_end']
    trans_start = config['manual_trim_transversal_start']
    trans_end = config['manual_trim_transversal_end']

    column_start, column_end = _interval2indices(transversal, trans_start, trans_end)
    row_start, row_end = _interval2indices(metadata.distance.values, longi_start, longi_end)
    return column_start, column_end, row_start, row_end+1 #2025-02-01 - JLB - resolves the last line is removed problem


def _interval2indices(distance, start, end):
    indices, = np.where(distance > start)
    start_idx = min(indices)
    indices, = np.where(distance < end)
    end_idx = max(indices)
    return start_idx, end_idx


def trim_temperature_data(pixels, threshold, autotrim_percentage):
    """
    Trim the temperature heatmap data by removing all outer rows and columns that only contains
    `autotrim_percentage` temperature values above `threshold`.
    """
    column_start, column_end = _trim_temperature_columns(pixels, threshold, autotrim_percentage)
    row_start, row_end = _trim_temperature_columns(pixels.T, threshold, autotrim_percentage)
    return column_start, column_end, row_start, row_end


def _trim_temperature_columns(pixels, threshold, autotrim_percentage):
    for idx in range(pixels.shape[1]):
        pixel_start = idx
        if not _trim(pixels, idx, threshold, autotrim_percentage):
            break


    for idx in reversed(range(pixels.shape[1])):
        pixel_end = idx
        if not _trim(pixels, idx, threshold, autotrim_percentage):
            break

    return pixel_start, pixel_end + 1 # because this is used in slicing so we need to adjust


def _trim(pixels, column, threshold_temp, autotrim_percentage):
    above_threshold = sum(pixels[:, column] > threshold_temp)
    above_threshold_pct = 100 * (above_threshold / pixels.shape[0])
    if above_threshold_pct > autotrim_percentage:
        return False

    return True


def detect_paving_lanes(df, threshold):
    """
    Detect lanes the one that is being actively paved during a two-lane paving operation where
    the lane that is not being paved during data acquisition has been recently paved and thus
    having a higher temperature compared to the surroundings.
    """
    df = df.copy(deep=True)
    df_temperature, _df_rest = split_temperature_data(df)
    pixels = df_temperature.values
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
    #start = len(mean_temp) - len(np.trim_zeros(above_thresh, 'f')) #JLB 03/07 - remove cold lane detection
    #end = - (len(mean_temp) - len(np.trim_zeros(above_thresh, 'b'))) #JLB 03/07 - remove cold lane detection
    start=0 #JLB 03/07 - remove cold lane detection
    end=len(mean_temp) #JLB 03/07 - remove cold lane detection
    
    # If there are longitudinal means below temperature threshold in the middle
    # it is probably because there is a shift in lanes.
    below_thresh = ~ above_thresh.astype('bool')
    if sum(below_thresh[start:end]) == 0:
        return None

    if sum(below_thresh[start:end]) > 0:
        # Calculate splitting point between lanes
        (midpoint, ) = np.where(mean_temp[start:end] == min(mean_temp[start:end]))
        midpoint = midpoint[0] + start
        return None #(start, midpoint, end) #JLB 03/07 - remove cold lane detection
    return None


def _classify_lanes(pixels, seperators):
    start, midpoint, end = seperators
    f_mean = pixels[:, start:midpoint].mean()
    b_mean = pixels[:, midpoint + 1:end].mean()
    # columns = df_temperature.columns
    if f_mean > b_mean:
        warm_lane = (0, midpoint + 1)  # columns[:midpoint + 1]
        cold_lane = (midpoint, pixels.shape[1])  # columns[midpoint:] # We exclude the seperating column
    else:
        warm_lane = (midpoint, pixels.shape[1])
        cold_lane = (0, midpoint + 1)

    return {'warmest': warm_lane,
            'coldest': cold_lane}


def estimate_road_width(pixels, threshold, adjust_left, adjust_right):
    """
    Estimate the road length of each transversal line (row) of the temperature
    heatmap data.
    """
    road_widths = []
    for idx in range(pixels.shape[0]):
        start = _estimate_road_edge_right(pixels[idx, :], threshold)
        end = _estimate_road_edge_left(pixels[idx, :], threshold)
        road_widths.append((start + adjust_left, end - adjust_right))
    return road_widths


def _estimate_road_edge_right(line, threshold):
    cond = line < threshold
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


def identify_roller_pixels(pixels, road_pixels, temperature_threshold):
    below_threshold = pixels < temperature_threshold
    roller_pixels = road_pixels & below_threshold
    return roller_pixels


def interpolate_roller_pixels(temperature_pixels, roller_pixels, road_pixels):
    non_roller_road_pixels = road_pixels & ~roller_pixels
    points = np.where(non_roller_road_pixels)
    values = temperature_pixels[points]
    points_interpolate = np.where(roller_pixels)
    # values_interpolate = griddata(points, values, points_interpolate, method='linear')
    # temperature_pixels[points_interpolate] = 200.0 # values_interpolate
    temperature_pixels[points_interpolate] = np.mean(temperature_pixels[points])



def trimguess(temperatures, config):    

    autotrim1=1
    if autotrim1==1:

        #TEMP=temperatures[temperatures>=config['roadwidth_threshold']]
        TEMP=temperatures[temperatures>=100]
# Temp1=TEMP.dropna(how='all',axis=1)
        Temp1=TEMP
# print(Temp1.mean())
        if config['pixel_width']==0.03:
            Limit=100
        else:
            #Limit=250
            Limit=200
        print(((Temp1.count())))
        print(((Temp1.count()>=Limit)))
        n=0
        for i in range(np.size(Temp1,1)-1):
    # n=n+1 
            # print(i)
            if (Temp1.count()>=Limit).iloc[i]==True: # indsat iloc 01102024
                break
            n=n+1
        print('n=',n)
        k=n
        print(range(np.size(Temp1,1)-n-1))
        for i in range(np.size(Temp1,1)-n-1):
            # print(i+n)
            if (Temp1.count()>=Limit).iloc[i+n]==False: # indsat iloc 01102024
                break
            k=k+1
        print('k=',k)     
        # o=k

        print('k-n',k-n)
        if k-n<15 and config['pixel_width']==0.03 or k-n<5 and config['pixel_width']==0.25:
            TrimWarning='Auto trim error. Do a manual trim.'
            n=0
            k=np.size(Temp1,1)-3
        else:
            TrimWarning=''

       
        if config['pixel_width']==0.25: #or config['pixel_width']==0.03 :
            StartTrim=(n-1)*config['pixel_width']
            if StartTrim<0:
                StartTrim=0.0
            EndTrim=(k+1)*config['pixel_width']
        elif config['pixel_width']==0.03:
            StartTrim=round((n-4)*config['pixel_width'],2)
            EndTrim=round((k+4)*config['pixel_width'],2)
            if StartTrim<0:
                StartTrim=0.0
        print('StartTrim=',StartTrim)
        print('EndTrim=',EndTrim)      
        #if config['pixel_width']==0.25 or config['pixel_width']==0.03 :
            #StartTrim=(n-1)*config['pixel_width']
            #if StartTrim<0:
                #StartTrim=0.0
            #EndTrim=(k+1)*config['pixel_width']        
        #print('StartTrim=',StartTrim)
        #print('EndTrim=',EndTrim)
        
        return StartTrim, EndTrim, TrimWarning

