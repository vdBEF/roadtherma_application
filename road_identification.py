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


def trimguess(temperatures, config, TwoLane, ForceTrim, Less, High):    

    autotrim1=1
    if autotrim1==1:
        
        TEMP=temperatures[temperatures>=100]

        Temp1=TEMP # Temperatures over 100C
        Temp2=temperatures

        if config['pixel_width']==0.03:
            Limit=100
        else:
            Limit=200
   
        n=0
        for i in range(np.size(Temp1,1)-1):

            if (Temp1.count()>=Limit).iloc[i]==True: # indsat iloc 05092024
                break
            n=n+1
        # print('n=',n)
        k=n
        print(range(np.size(Temp1,1)-n-1))
        for i in range(np.size(Temp1,1)-n-1):
            if (Temp1.count()>=Limit).iloc[i+n]==False: # indsat iloc 05092024
                break
            k=k+1
        # print('k=',k)     
        # print('k-n',k-n)
        if k-n<=30 and config['pixel_width']==0.03 or k-n<5 and config['pixel_width']==0.25:
            TrimWarning='Auto trim error. Do a manual trim.'
            n=0
            k=np.size(Temp1,1)-4 # minus 4 fordi +4 til sidst
            print('k=max',k)
            kn=1
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
        
        
        p1=0
        p2=0
        p3=0
        p4=0
    

        
        cond1=(Temp1.count()).iloc[n+1]>500
        if (config['pixel_width']==0.25):
            cond= (n<11 or k>np.shape(Temp2)[1]-10)
        else:
            cond= (n<20 or k>np.shape(Temp2)[1]-20)

        if (config['pixel_width']==0.25):
            cond2=Temp2.std().iloc[n:k].mean()<36 # måske 30 bedre
           
        else:
            p1=Temp2.std().iloc[n:k].std()<10
            p2=(Temp2.mean().iloc[n:k].var()<300 and Temp2.std().iloc[n:k].std()<25 )
            p3=(Temp2.mean().iloc[n:n+2].mean()>90 or Temp2.mean().iloc[k-2:k].mean()>90)
            p4=(Temp2.mean().iloc[n+2:n+6].mean()>85 or Temp2.mean().iloc[k-6:k-2].mean()>85 )
            
            cond2=p1 or (p2 and (p3 or p4))
        print('p1',p1)
        print('p2',p2)
        print('p3',p3)
        print('p4',p4)
        print('cond',cond)
        print('cond1',cond1)
        print('cond2',cond2)
        if ForceTrim==True:
            wd=0
        else:
            wd=5
        if Less==True:
            LK=0.8
        if High==True:
            LK=1.025
        else:
            LK=1
        if abs(StartTrim-EndTrim)>wd and (TwoLane==False and ((cond==True and cond2==True) or (ForceTrim==True))):# or (cond1==True and cond2==True)):
        # if TwoLane==False: #and ((cond==True and cond2==True) or (ForceTrim==True))) # Always turned on

            pp=0
            # lane to the right
            if pp==0:#Temp2.mean().iloc[0]<60 and Temp2.mean().iloc[k-1]>60:
                
                print('TWO PAVED LANES')

                TT=[]
                for i in range(0,np.size(Temp2,1)):
                
                    TT.append(Temp2.mean()[Temp2.mean()>50])
                    # TT.append((Temp2.iloc[i]<120).value_counts()[True])
                # print(len(TT))
               
                N=len(Temp2.mean())-len(TT) 
                
                # y=0
                # TTT=[]
                TT=Temp2.mean() 
                print('len(TT)',len(TT))
                # if config['pixel_width']==0.03:  
                
             
                    
                LLN=Temp2.mean()[Temp2.mean()>50].mean()
                LLK=Temp2.mean()[Temp2.mean()>50].mean()
                print('LLN',LLN)
                print('LLK',LLK)    
                # print('LL',LL)
                # nT=0
                
                ii=0 
                nnnT=0
                nT=0+nnnT
                nnT=nT
                print('nT',nT)
                LL=Temp2.mean()[Temp2.mean()>50].mean()
                LL=LL*LK
                for p in range(100):
                        l=p    
                        if abs(TT.iloc[(ii+p)])>LL:

                            if TT.iloc[(ii+p)]<LL:
   
                                l=l+1
                                print('l',l)
                                nnnT=nnT+l

                        
                            else: 
                               nT=nnT+l
 
                               if len(TT)>200 and (n<=11 or len(TrimWarning)>0):
                                   nT=nnT+l
                                   print('TF nT',nT)
                                   print('LL',LL)
                                   break
                               elif len(TT)>200: 
                                   nT=nnT+l-4
                                   print('TF nT2',nT)
                                   print('LL',LL)
                               else:
                                    nT=nnT+l
                                    print('nT',nT)
                                    print('LL',LL)
                                    break
                               break    
                    # break    
                    
                #if n-nT<=5 and n<=2 and len(TT)>200:
                 #   print('LL*1.2') 
                  #  for p in range(100):

                   #     l=p
                    #    if abs(TT.iloc[(ii+p)])>LLN :

                     #       if TT.iloc[(ii+p)]<LL+2:
                           
                      #          print('l',l)
                        #        nT=nnT+l

                       #     else: #TT.iloc[(ii+1)]<TT.iloc[(ii+2)]:
                         #     nT=nnT+l

                          #    if len(TT)>200 and (n<=5 or len(TrimWarning)>0):
                           #         nT=nnT+l-4
                            #        print('TF nT',nT)
                             #       print('LL',LL)
                              #      break
                              #else:
                               #     nT=nnT+l-1
                                #    print('nT',nT)
                                 #   print('LL',LL)
                                  #  break
                              #break 
  
     
                kkkT=len(TT)-1
                kT=kkkT

                kkT=kT
                # print('start kT',kT)
                LL=Temp2.mean()[Temp2.mean()>50].mean()
                LL=LL*LK
                # print('abs(TT.iloc[kT])',abs(TT.iloc[kT]),kT)
                for p in range(1,150,1):

                    l=p
                    # print('abs(TT.iloc[kT-(p)])',abs(TT.iloc[kT-(p)]),kT-(p))

                    cn=abs(TT.iloc[(kT-(p+1))])<LL
                    
                    if cn:
                        # l=l+1
                        kkkT=kkT-l
                        # print('p',p)
                        # print('LL',LL)
                        # # print('abs(TT.iloc[kT])',abs(TT.iloc[kT]),kT)
                        # print('abs(TT.iloc[kT-(p-1)])',abs(TT.iloc[kT-(p-1)]),kT-(p-1))
                        # print('abs(TT.iloc[kT-(p)])',abs(TT.iloc[kT-(p)]),kT-(p))
                        
                    else:
                            l=p+1                               
                            kT=kkT-l
                            # if TT.iloc[(kT-(p))]>LL:
                            #     break
                            # l=l-2
                            if len(TT)-kkkT+2>=nT:
                                rr=nT-5
                            else:
                                rr=len(TT)-(kkkT+5)
                            for p in range(rr):
                                print('abs(TT.iloc[kkkT-(p+2)])',abs(TT.iloc[kkkT-(p+2)]),kkkT-(p+2))
                                if abs(TT.iloc[(kkkT-(p+2))])<LL: 
                                    kT=kkkT-(p+2)
                                    if abs(TT.iloc[kT+2])>LL:
                                        break
                            
                            
                            if len(TT)>200 and (k>=270 or len(TrimWarning)>0):
                                kT=kkT-l+4
                                print('TF kT',kT)
                                print('LL',LL)
                                if (kkT-l)>272:
                                    kT=280
                                break    
                            elif len(TT)>200: 
                                kT=kkT-l+4
                                print('TF nT2',nT)
                                print('LL',LL)        
                                    
                            break
                            break               
                

                print('End nT',nT)
                print('End kT',kT)  
   
            
                if config['pixel_width']==0.25: #or config['pixel_width']==0.03 :
                    if nT>n and n>6:
                        n=n
                    else:
                        n=nT
                    if kT<k and k<len(TT)-10:
                        k=k
                        EndTrim=(k+1)*config['pixel_width']
                    else:
                        k=kT
                        EndTrim=(k+2)*config['pixel_width']
                    StartTrim=(n-1)*config['pixel_width']
                    if StartTrim<0:
                        StartTrim=0.0
                    # EndTrim=(k+1)*config['pixel_width']
                
                elif config['pixel_width']==0.03:
                    if nT-n<15 and nT>n and nT>30: 
                        n=n
                        StartTrim=round((n-4)*config['pixel_width'],2)
                        
                    else:
                        print('n=nT')
                        n=nT
                        StartTrim=round((n-4)*config['pixel_width'],2)
                      
                    if kT<=k and k<(len(TT)-40):# and kT<k-20:
                        k=kT
                        EndTrim=round((k+4)*config['pixel_width'],2)
                    
                    else:
                        print('k=kT ',kT)
                        k=kT
                        EndTrim=round((k+4)*config['pixel_width'],2)

                    if StartTrim<0:
                        StartTrim=0.0
                print('StartTrim=',StartTrim)
                print('EndTrim=',EndTrim)        
        
            if k-n<30 and config['pixel_width']==0.03 or k-n<5 and config['pixel_width']==0.25:
                TrimWarning='Auto trim error. Do a manual trim.'
                n=0
                k=np.size(Temp1,1)-4 # minus 4 fordi +4 til sidst
            else:
                TrimWarning=''        
        
        return StartTrim, EndTrim, TrimWarning


#old version
#def trimguess(temperatures, config):    

    #autotrim1=1
   # if autotrim1==1:

        #TEMP=temperatures[temperatures>=config['roadwidth_threshold']]
        #TEMP=temperatures[temperatures>=100]
# Temp1=TEMP.dropna(how='all',axis=1)
       # Temp1=TEMP
# print(Temp1.mean())
        #if config['pixel_width']==0.03:
         #   Limit=100
        #else:
            #Limit=250
         #   Limit=200
        #print(((Temp1.count())))
        #print(((Temp1.count()>=Limit)))
        #n=0
        #for i in range(np.size(Temp1,1)-1):
    # n=n+1 
            # print(i)
           # if (Temp1.count()>=Limit).iloc[i]==True: # indsat iloc 01102024
          #      break
         #   n=n+1
        #print('n=',n)
        #k=n
        #print(range(np.size(Temp1,1)-n-1))
        #for i in range(np.size(Temp1,1)-n-1):
            # print(i+n)
          #  if (Temp1.count()>=Limit).iloc[i+n]==False: # indsat iloc 01102024
         #       break
        #    k=k+1
       # print('k=',k)     
        # o=k

        #print('k-n',k-n)
        #if k-n<15 and config['pixel_width']==0.03 or k-n<5 and config['pixel_width']==0.25:
           # TrimWarning='Auto trim error. Do a manual trim.'
          #  n=0
         #   k=np.size(Temp1,1)-4
        #else:
         #   TrimWarning=''

       
        #if config['pixel_width']==0.25: #or config['pixel_width']==0.03 :
            #StartTrim=(n-1)*config['pixel_width']
           # if StartTrim<0:
          #      StartTrim=0.0
         #   EndTrim=(k+1)*config['pixel_width']
        #elif config['pixel_width']==0.03:
            #StartTrim=round((n-4)*config['pixel_width'],2)
           # EndTrim=round((k+4)*config['pixel_width'],2)
         #   if StartTrim<0:
          #      StartTrim=0.0
        #print('StartTrim=',StartTrim)
        #print('EndTrim=',EndTrim)      
        #if config['pixel_width']==0.25 or config['pixel_width']==0.03 :
            #StartTrim=(n-1)*config['pixel_width']
            #if StartTrim<0:
                #StartTrim=0.0
            #EndTrim=(k+1)*config['pixel_width']        
        #print('StartTrim=',StartTrim)
        #print('EndTrim=',EndTrim)
        
        #return StartTrim, EndTrim, TrimWarning

