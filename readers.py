# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:40:50 2024

@author: b306460
"""

import re
from datetime import datetime
import csv
import pandas as pd
import numpy as np
import sys





#%% Readers opdelt i kamera typer:

def _read_Voegele(filename):

    list_functions=[_read_voegele_1,_read_voegele_2]    
    # list_functions=[_read_voegele_2]  
    
    for function in list_functions:
        
        try:
            df, str1, res=function(filename)
            print(df)
            df=df.replace(',','.')
            print(df)
            # Finding nan rows and deleting them: (100325)
            DL=[]    
            for i in range(len(df)):
                if any(df.iloc[i].isna())==True:
                    DL.append(i)
            print('DL',DL)           
            df.drop(DL,inplace=True)
        
            # df, str1, res=_read_voegele_1(filename)
        except:
            # df, str1, res=_read_voegele_2(filename)
            pass
        else:
            break
    else:
        raise Exception("No function succeeded.")
        # str1="No function succeeded."
    return df, str1#, res






def _read_TF(filename):

    list_functions=[_read_TF_time,_read_TF_new,_read_TF_old]    

    for function in list_functions:
        
        try:
            df, str1, res=function(filename)
            # Finding nan rows and deleting them: (100325)
            DL=[]    
            for i in range(len(df)):
                if any(df.iloc[i].isna())==True:
                    DL.append(i)
            print('DL',DL)
            df.drop(DL,inplace=True)
        except:
            pass
        else:
            break
    else:
        raise Exception("No function succeeded.")
        # str1="No function succeeded."
    return df, str1#, res




def _read_Moba(filename):

    list_functions=[_read_moba1,_read_moba2,_read_moba3]    
    # list_functions=[_read_moba2,_read_moba1,_read_moba3]    
    for function in list_functions:
        try:
            df, str1, res=function(filename)
            # Finding nan rows and deleting them: (100325)
            DL=[]    
            for i in range(len(df)):
                if any(df.iloc[i].isna())==True:
                    DL.append(i)
            print('DL',DL)
            df.drop(DL,inplace=True)
        except:
            pass
        else:
            break
    else:
        raise Exception("No function succeeded.")
        # str1="No function succeeded."
    return df, str1#, res
    # return df

def _read_ALL(filename):
    # list_functions=[_read_all]
    # for function in _read_all:
        # try:
            df, str1, res = _read_all(filename)
            # print(t[0])
            # df=t[0]
            
            # str1=t[1]
            # res=t[2]
            
            # Finding nan rows and deleting them:
            DL=[]
            for i in range(len(df)):
                if any(df.iloc[i].isna())==True:
                    DL.append(i)
            print(DL)           
            df.drop(DL,inplace=True)  
            
        # except:
            # pass
        # else:
            # raise Exception("No function succeeded.")
    # else:
    #     raise Exception("No function succeeded.")
    #     str1="No function succeeded."
            return df, str1#, res    


#%% Voegele
temperatures_voegele = ['T{}'.format(n) for n in range(52)]
VOEGELE_BASE_COLUMNS = ['time', 'distance', 'latitude', 'longitude']
def _convert_vogele_timestamps(df, formatting):
    df['time'] = pd.to_datetime(df.time, format=formatting)
    
    

def _read_voegele_1(filename):
# vogele_M30 og vogele_taulov
    print('vogele_M30 or vogele_taulov')
    temperatures_voegele = ['T{}'.format(n) for n in range(52)]
    VOEGELE_BASE_COLUMNS = ['time', 'distance', 'latitude', 'longitude']
    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
    try:
        dfT = pd.read_csv(filename,delimiter=',', encoding='utf_8_sig',skiprows=5,nrows=10,quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)
    except UnicodeDecodeError:
        dfT = pd.read_csv(filename,delimiter=',',encoding='cp1252',skiprows=5,nrows=10,quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)
    print(dfT)
    if  bool(re.search('"',str(dfT.loc[0][0])))==True:
        for col in dfT.columns:
            dfT[col] = dfT[col].apply(lambda x:x.strip(''))
    print(dfT)
    print(dfT.loc[0][0])            
    print(len(dfT.loc[0][0]) ) 
    try:
        df = pd.read_csv(filename, skiprows=3, delimiter=',', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True,encoding='cp1252')
    except:
        df = pd.read_csv(filename, skiprows=3, delimiter=',', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)
    print(df)
    for col in df.columns:
        if col == 'time':
            df[col] = df[col].apply(lambda x:x.strip('"'))
            if len(dfT.loc[0][0])>=31:
                try:
                    df['time'] = pd.to_datetime(df.time, format="%d-%m-%Y %H:%M:%S UTC + 02:00")
                except:
                    df['time'] = pd.to_datetime(df.time, format="%d/%m/%Y %H:%M:%S UTC + 02:00") # NOTE only difference between this and _read_vogele_taulov is the UTC-part here (ffs!)
            else:
                df['time'] = pd.to_datetime(df.time, format="%d/%m/%Y %H:%M:%S")
        elif col in set(temperatures_voegele) | {'distance', 'latitude', 'longitude'}:
            
            #df[col] = df[col].astype('str').apply(lambda x:x.strip('"')).astype('float')
            # df[col] = df[col].astype('str').apply(lambda x:x.strip('"'))
            try: 
                df[col] = df[col].astype('str').apply(lambda x:x.strip('"')).astype('float')
            except:
                df.replace('""',np.nan,inplace=True)
                df[col] = df[col].astype('str').apply(lambda x:x.strip('"')).astype('float')
    # print(df)
    # if len(dfT.loc[0][0])==31:
    #     try:
    #         df['time'] = pd.to_datetime(df.time, format="%d-%m-%Y %H:%M:%S UTC + 02:00")
    #     except:
    #         df['time'] = pd.to_datetime(df.time, format="%d/%m/%Y %H:%M:%S UTC + 02:00") # NOTE only difference between this and _read_vogele_taulov is the UTC-part here (ffs!)
    # else:
    #     df['time'] = pd.to_datetime(df.time, format="%d/%m/%Y %H:%M:%S")
    # str1="\n All data used \n"
    # res=0
    
    str2=len(df)
    dftest=df.copy()    
    for n in range(len(temperatures_voegele)): 
        # print(df[df['T{}'.format(n)] > 1000])
        df = df.drop(df[df['T{}'.format(n)] > 1000].index)
        df=df.reset_index(drop=True)
        # str1 =
    res=str2-len(df) 
    if    res >=1 :
            #res virker ikke...
        str1="\n"+str(res)+" Row(s) removed because temperatures > 1000\N{DEGREE SIGN}C \n"
        # str1="Rows are removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    else:
        str1="\nAll data used (T<1000\N{DEGREE SIGN}C) \n"
    
    
    print(df)

    for n in range(len(temperatures_voegele)): 
        df['T{}'.format(n)]=df['T{}'.format(n)].astype(float)
           
    return df, str1, res


def _read_voegele_2(filename):
    # Voegele_roller_example og m119
    print('Voegele_roller_example or voegele_m119')
    temperatures_voegele = ['T{}'.format(n) for n in range(52)]
    VOEGELE_BASE_COLUMNS = ['time', 'distance', 'latitude', 'longitude']
    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
    try:
        dfT = pd.read_csv(filename,delimiter=';',skiprows=5,nrows=10)
        BL=any("," in s for s in re.findall(',',str(dfT.iloc[0,1])))
    except UnicodeDecodeError:
        dfT = pd.read_csv(filename,delimiter=';',encoding='cp1252',skiprows=5,nrows=10)
        BL=any("," in s for s in re.findall(',',str(dfT.iloc[0])))
        
    if  any('"' in s for s in re.findall('"',dfT.iloc[0,0]))==True:#bool(re.search('"',str(dfT.loc[0][0])))==True:
        for col in dfT.columns:
            dfT[col] = dfT[col].apply(lambda x:x.strip('"'))
    print(dfT)
    try: 
        BL2=any("." in s for s in re.findall('.',str(dfT.iloc[0,7])))
    except:
        BL2=any("." in s for s in re.findall('.',str(dfT.iloc[0])))
    
    if BL==True and BL2==False:#bool(re.search(',',str(dfT.loc[0][1])))==True:
        try:            
            df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',',quoting=csv.QUOTE_NONE)
            print(df)
            f=1
        except:
            df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',',encoding='cp1252',quoting=csv.QUOTE_NONE) #encoding='cp1252'
            print(df)
            f=1
    elif BL==True and BL2==True:
        try:            
            df = pd.read_csv(filename, skiprows=2, delimiter=',', names=columns, decimal='.',quoting=csv.QUOTE_NONE)
            print(df)
            f=1
        except:
            df = pd.read_csv(filename, skiprows=2, delimiter=',', names=columns, decimal='.',encoding='cp1252',quoting=csv.QUOTE_NONE) #encoding='cp1252'
            print(df)
            f=1
            
        
    for col in df.columns:
            if col in {'distance', 'latitude', 'longitude'}|set(temperatures_voegele):
               try: 
                   df[col] = df[col].astype('str').apply(lambda x:x.strip('"'))
                   df[col] = df[col].astype('str').apply(lambda x:x.replace(',','.')).astype(float)
               except:
                   df.replace('""',np.nan,inplace=True)
                   df[col] = df[col].astype('str').apply(lambda x:x.strip('"'))
                   df[col] = df[col].astype('str').apply(lambda x:x.replace(',','.')).astype(float)
            if col in {'time'}:
               try: 
                   df[col] = df[col].astype('str').apply(lambda x:x.strip('"'))
                   
               except:
                   df.replace('""',np.nan,inplace=True)
                   df[col] = df[col].astype('str').apply(lambda x:x.strip('"'))




    if f!=1:
        try:            
            df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',')
        except:
            df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',',encoding='cp1252')
        print(df)
    if len(dfT.loc[0][0])==31:
        try:
            try:
                df['time'] = pd.to_datetime(df.time, format="%d-%m-%Y %H:%M:%S UTC + 02:00")
            except:
                df['time'] = pd.to_datetime(df.time, format="%d-%m-%Y %H:%M:%S UTC + 01:00")
        except:
            try:    
                df['time'] = pd.to_datetime(df.time, format="%d/%m/%Y %H:%M:%S UTC + 02:00")
            except:
                df['time'] = pd.to_datetime(df.time, format="%d/%m/%Y %H:%M:%S UTC + 01:00")
            # print(df.dtypes)
    elif BL2==True:
        try:
            print(11)
            df['time'] = pd.to_datetime(df.time, format="%d-%m-%Y %H:%M:%S UTC + 01:00")
        except:
            print(1)
            df['time'] = pd.to_datetime(df.time, format="%d/%m/%Y %H:%M:%S UTC + 01:00")   
    else:
        df['time'] = pd.to_datetime(df.time, format="%d/%m/%Y %H:%M:%S")
    # str1="\n All data used \n"
    # res=0
    
    str2=len(df)
    dftest=df.copy()    
    for n in range(len(temperatures_voegele)): 
        # print(df[df['T{}'.format(n)] > 1000])
        df = df.drop(df[df['T{}'.format(n)] > 1000].index)
        df=df.reset_index(drop=True)
        # str1 =
    res=str2-len(df) 
    if    res >=1 :
            #res virker ikke...
        str1="\n"+str(res)+" Row(s) removed because temperatures > 1000\N{DEGREE SIGN}C \n"
        # str1="Rows are removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    else:
        str1="\nAll data used (T<1000\N{DEGREE SIGN}C) \n"
    for n in range(len(temperatures_voegele)): 
        df['T{}'.format(n)]=df['T{}'.format(n)].astype(float)
        
    print(df)
    return df, str1, res



#%% TF


def _read_TF_new(filename):
    print('TF_new')
    temperatures = ['T{}'.format(n) for n in range(281)]
    # print(temperatures)
    columns = ['distance'] + ['time'] + ['latitude'] + ['longitude'] + temperatures
    df = pd.read_csv(filename, skiprows=5, delimiter=',', names=columns)
    df['time'] = [x[:-6] for x in df.time]
    df['time'] = pd.to_datetime(df.time, format=' %Y-%m-%dT%H:%M:%S')
    del df['T280']
    # str1="\n All data used \n"
    # res=0
    
    str2=len(df)
    dftest=df.copy()    
    for n in range(len(temperatures)): 
        # print(df[df['T{}'.format(n)] > 1000])
        df = df.drop(df[df['T{}'.format(n)] > 1000].index)
        df=df.reset_index(drop=True)
        # str1 =
    res=str2-len(df) 
    if    res >=1 :
            #res virker ikke...
        str1="\n"+str(res)+" Row(s) removed because temperatures > 1000\N{DEGREE SIGN}C \n"
        # str1="Rows are removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    else:
        str1="\nAll data used (T<1000\N{DEGREE SIGN}C) \n"
    
    
    
    return df, str1, res

def _read_TF_time(filename):
    print('TF_time')

    try:
        with open(filename, newline='') as f:    
            Tcsv_reader = csv.reader(f)
            Tfirst_line = next(Tcsv_reader)
            Tsecond_line = next(Tcsv_reader)
            Tthird_line = next(Tcsv_reader)
            T4_line = next(Tcsv_reader)
            T5_line = next(Tcsv_reader)
            T6_line = next(Tcsv_reader)
            T7_line = next(Tcsv_reader)
            T8_line = next(Tcsv_reader)
            T9_line = next(Tcsv_reader)
            print(T8_line[0:10]) 
    except:
        T8_line=pd.read_csv(filename,delimiter=';', skiprows=8,nrows=1)   
       
    temperatures = ['T{}'.format(n) for n in range(280)]
    columns = ['distance'] +['time'] + ['latitude'] + ['longitude'] + temperatures

    if bool(re.search(',',str(T8_line)))==True:
        
        df = pd.read_csv(filename, skiprows=5, delimiter=';', names=columns,thousands='.',decimal=',')
    else:
        df = pd.read_csv(filename, skiprows=5, delimiter=';', names=columns,thousands=',',decimal='.')
     
    print(df)  
    try:
        df['time']=[x.replace(' ','') for x in df.time]
        df['time'] = pd.to_datetime(df.time, format='%H:%M:%S')
    except:
        print('fejl time')
    #df['time']=[x.replace(' ','') for x in df.time]    
    #df['time'] = pd.to_datetime(df.time, format='%H:%M:%S')
    print(df) 
#del df['T280']

    str2=[]
    # for n in range(len(temperatures)):
    #     str2.append(len(df[df['T{}'.format(n)] > 1000].index))
        
        
    #     # if (df['T{}'.format(n)].max())<1000:
    #     #     str2.append("DB")
    #     # else:
    #     #     # str2[n]="Række(r) er fjernet pga temperatur højere end 1000C" 
    #     #     str2.append("DS")
    # res=np.max(str2)
    # res=str2.count("DS")
    # if    res >=1 :
    #         #res virker ikke...
    #     str1="\n"+str(res)+" Row(s) removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    #     # str1="Rows are removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    # else:
    #     str1="\nAll data used (T<1000\N{DEGREE SIGN}C) \n"
    str2=len(df)
    dftest=df.copy()    
    for n in range(len(temperatures)): 
        # print(df[df['T{}'.format(n)] > 1000])
        df = df.drop(df[df['T{}'.format(n)] > 1000].index)
        df=df.reset_index(drop=True)
        # str1 =
    res=str2-len(df) 
    if    res >=1 :
            #res virker ikke...
        str1="\n"+str(res)+" Row(s) removed because temperatures > 1000\N{DEGREE SIGN}C \n"
        # str1="Rows are removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    else:
        str1="\nAll data used (T<1000\N{DEGREE SIGN}C) \n"
    
    return df, str1, res




def _read_TF_old(filename):
    
    print('TF_old')
    temperatures = ['T{}'.format(n) for n in range(141)]
    columns = ['distance'] + temperatures + ['distance_again']
    df = pd.read_csv(filename, skiprows=7, delimiter=',', names=columns)
    del df['distance_again']
    del df['T140'] # This is the last column of the dataset (which is empty)
    # str1="\n All data used \n"
    # res=0
    
    str2=len(df)
    dftest=df.copy()    
    for n in range(len(temperatures)): 
        # print(df[df['T{}'.format(n)] > 1000])
        df = df.drop(df[df['T{}'.format(n)] > 1000].index)
        df=df.reset_index(drop=True)
        # str1 =
    res=str2-len(df) 
    if    res >=1 :
            #res virker ikke...
        str1="\n"+str(res)+" Row(s) removed because temperatures > 1000\N{DEGREE SIGN}C \n"
        # str1="Rows are removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    else:
        str1="\nAll data used (T<1000\N{DEGREE SIGN}C) \n"
    
    
    return df, str1, res











#%% Moba


def _temperatures_moba(filename):
    print('før sensor')
    print(pd.read_csv(filename))
    sensors = pd.read_csv(filename, sep=';', skiprows=13, nrows=1)
    print(sensors)
    sensors.columns = ['name', 'number', 'none']
    n_sens = sensors.number + 1
    temperatures_MOBA = ['T{}'.format(n) for n in range(int(n_sens))]
    return temperatures_MOBA


MOBA_BASE_COLUMNS = ['index','distance', 'speed','temporary_time', 'longitude', 'latitude']
           
           
           
           
def _rows_moba(filename):
  
   test = pd.read_csv(filename, index_col = False, sep=';', skiprows=27)
   rows = 0
   for i in range(test.index.size):
        if type(test.index[i]) == int:
               rows += 1
        elif test.index[i].isdigit() is True:
               rows += 1
        else:
            break
                
   return rows

                 
def _read_moba1(filename): 
    print('Moba1')               
    with open(filename, newline='') as f:
        csv_reader = csv.reader(f)
        
        _csv_headings = next(csv_reader)
        
        first_line = next(csv_reader)
        for row in csv_reader:
                    
            if row[0][0:5]=='Index':
                        
                skip=csv_reader.line_num
                
                break
       
    date = first_line[1].split(' ')  
    date[0] = re.sub(r'\s+', '', first_line[2]) 
    date[1] = str(datetime.datetime.strptime(date[1], '%B').month)
          
    date = '-'.join(date)
    temperatures_MOBA = _temperatures_moba(filename)
    MOBA_BASE_COLUMNS = ['index','distance', 'speed','temporary_time', 'longitude', 'latitude']
    columns = MOBA_BASE_COLUMNS + ['signal_quality', 'satellites'] + temperatures_MOBA
    try:
        df = pd.read_csv(filename,
                    skiprows=skip,
                    nrows=_rows_moba(filename),
                    delimiter=';',
                    names=columns,
                    quoting=csv.QUOTE_NONE,
                    quotechar='"',
                    doublequote=True)
    except UnicodeDecodeError:
        df = pd.read_csv(filename,
                    skiprows=skip,
                    nrows=_rows_moba(filename),
                    delimiter=';',
                    names=columns,
                    quoting=csv.QUOTE_NONE,
                    quotechar='"',
                    doublequote=True, encoding='cp1252')

    df = df.drop(labels='index', axis=1) 
    df['date'] = date
    df.drop(columns=[temperatures_MOBA[-1]],inplace=True)
    df = df.dropna()
    df=df.reset_index(drop=True)           
    df['temporary_time'] = df['date'] + [' '] + df['temporary_time']
    df = df.drop(labels='date', axis=1)
    # df.insert(loc=2, column='time', value=[datetime.datetime.strptime(df['temporary_time'][i],'%Y-%m-%d %H:%M:%S')for i in range(df.shape[0])])
    df.insert(loc=1, column='time', value=[datetime.datetime.strptime(df['temporary_time'][i],'%Y-%m-%d %H:%M:%S')for i in range(df.shape[0])])
    df['time'] = pd.to_datetime(df['time'])
    df = df.drop(labels='temporary_time', axis=1)
    df = df.drop(labels='speed', axis=1)
    df = df.drop(labels='satellites', axis=1)
    df = df[['time', 'distance','latitude', 'longitude', 'signal_quality']+ temperatures_MOBA[:len(temperatures_MOBA)-1]]
    df['distance'] = df['distance'].astype(float)
    df[temperatures_MOBA[:len(temperatures_MOBA)-1]]=df[temperatures_MOBA[:len(temperatures_MOBA)-1]].apply(pd.to_numeric, errors='coerce', axis=1).fillna(0, downcast='infer')
       # df = df.drop(labels=temperatures_MOBA[-1], axis=1)
    
    str2=len(df)
    dftest=df.copy()    
    for n in range(len(temperatures_MOBA)-1): 
        # print(df[df['T{}'.format(n)] > 1000])
        df = df.drop(df[df['T{}'.format(n)] > 1000].index)
        df=df.reset_index(drop=True)
        # str1 =
    res=str2-len(df) 
    if    res >=1 :
            #res virker ikke...
        str1="\n"+str(res)+" Row(s) removed because temperatures > 1000\N{DEGREE SIGN}C \n"
        # str1="Rows are removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    else:
        str1="\nAll data used (T<1000\N{DEGREE SIGN}C) \n"
    
    
    # str1="\n All data used \n"
    # res=0
    return df, str1, res     
                



# def _sensors_moba2(filename):
#          sensors2 = pd.read_csv(filename, sep = ';', skiprows=5, nrows=1, encoding='cp1252')
#          sensors2=sensors2.iloc[:, list(range(2)) + [-1]]
#          sensors2.columns = ['name', 'number', 'none']
#          n_sens2 = sensors2.number 
#          return n_sens2


def _temp_MOBA2(filename):
         sensors2 = pd.read_csv(filename, sep = ';', skiprows=5, nrows=1,encoding='cp1252')
         sensors2=sensors2.iloc[:, list(range(2)) + [-1]]
         sensors2.columns = ['name', 'number', 'none']
         n_sens2 = sensors2.number 
         temperatures_MOBA2 = ['T{}'.format(n) for n in range(int(n_sens2))]
         return temperatures_MOBA2

 #temperatures_MOBA = ['T{}'.format(n) for n in range(_sensors_moba(filename))]
MOBA_BASE_COLUMNS2 = ['time','distance', 'latitude', 'longitude', 'altitude','humidity','pressure','air temp','wind speed']


def _rows_moba2(filename):
    try:
             test2 = pd.read_csv(filename, sep = ';', skiprows=33,quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)
    except UnicodeDecodeError:
             test2 = pd.read_csv(filename, sep = ';', skiprows=33,encoding='cp1252',quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)
         # print(test2.head())
         # print(test2['Altitude [m]'])
         # print(range(int(len(test2['Altitude [m]']))))
    rows2 = 0
         #Finder første række index hvor det er nan.
    rows2=test2.loc[pd.isna(test2['Altitude [m]']), :].index[0]
         # for t in range(int(len(test2['Altitude [m]']))):
         #     if np.isnan(test2['Altitude [m]'].loc[t]) == False:
               
         #         rows2 = rows2 + 1
         #         print(rows2)
         #     else: 
         #         # print(np.isnan(test2['Altitude [m]'][i])==True)
         #         # print(rows2)
         #         break
            
    return rows2
        
def _read_moba2(filename):
    print('Moba2')
    temperatures_MOBA2 = _temp_MOBA2(filename)
    columns = MOBA_BASE_COLUMNS2 + temperatures_MOBA2 + ['ScreedWidthLeft', 'ScreedWidthRight'] 
     # print(_rows_moba2(filename))
    try:
        df = pd.read_csv(filename, skiprows=34,nrows=_rows_moba2(filename), delimiter=';', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)    
    except UnicodeDecodeError:
        df = pd.read_csv(filename, skiprows=34,nrows=_rows_moba2(filename), delimiter=';', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True,encoding='cp1252')  
     # df = pd.read_csv(filename, skiprows=34,nrows=_rows_moba2(filename), delimiter=';', names=columns)    
    df['time']=[datetime.datetime.strptime(df['time'][i],'%d.%m.%Y %H:%M:%S')for i in range(df.shape[0])] 
    # str1="\n All data used \n"
    # res=0
     # print(df)
    str2=len(df)
    dftest=df.copy()    
    for n in range(len(temperatures_MOBA2)): 
        # print(df[df['T{}'.format(n)] > 1000])
        df = df.drop(df[df['T{}'.format(n)] > 1000].index)
        df=df.reset_index(drop=True)
        # str1 =
    res=str2-len(df) 
    if    res >=1 :
            #res virker ikke...
        str1="\n"+str(res)+" Row(s) removed because temperatures > 1000\N{DEGREE SIGN}C \n"
        # str1="Rows are removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    else:
        str1="\nAll data used (T<1000\N{DEGREE SIGN}C) \n" 
     
     
    return df, str1, res





# def _sensors_moba3(filename):
#     try:
#        sensors = pd.read_csv(filename, sep=';', skiprows=5, nrows=1)
#        # print(sensors)
       
#     except UnicodeDecodeError:
#        # print(e)
#        sensors = pd.read_csv(filename, sep=';', skiprows=5, nrows=1,encoding='iso8859_10')
#        # print(sensors)
#     sensors.columns = ['name', 'number']
#     # print(sensors)
#     n_sens = sensors.number + 1
#     return n_sens

def _temperatures_moba3(filename):
    try:
        sensors = pd.read_csv(filename, sep=';', skiprows=5, nrows=1)
    except UnicodeDecodeError:
        sensors = pd.read_csv(filename, sep=';', skiprows=5, nrows=1,encoding='cp1252')
    sensors.columns = ['name', 'number']
    n_sens = sensors.number + 1
    # n_sens = sensors.number
    temperatures_MOBA = ['T{}'.format(n) for n in range(int(n_sens))]
    return temperatures_MOBA

MOBA_BASE_COLUMNS = ['time',
                      'distance',
                      'latitude',
                      'longitude', 
                      'altitude',
                      'Sensor[Single IR Spot 1]',
                      'Sensor[Single IR Spot 2]',
                      'Humidity [%]',
                      'Pressure [hPa]',
                      'AirTemperature [°C]',
                      'WindSpeed [km/h]']


def _read_moba3(filename):
    print('Moba3')
    # try
    temperatures_MOBA = _temperatures_moba3(filename)
    # except UnicodeDecodeError:
    #     temperatures_MOBA = _temperatures_moba3(filename,encoding='cp1252')
    columns = MOBA_BASE_COLUMNS + temperatures_MOBA
    print(temperatures_MOBA)
    
    try:
        df = pd.read_csv(filename,
        index_col = False,
        skiprows=34,
        delimiter=';',
        names=columns,
        quoting=csv.QUOTE_NONE,
        quotechar='"',
        doublequote=True)
    except UnicodeDecodeError:
        df = pd.read_csv(filename,
        index_col = False,
        skiprows=34,
        delimiter=';',
        names=columns,
        quoting=csv.QUOTE_NONE,
        quotechar='"',
        doublequote=True,encoding='cp1252')
        # doublequote=True,encoding='uft_8_sig')
    # print(df)
    df.insert(loc=1, column='time2', value=[datetime.datetime.strptime(df['time'][i],'%d.%m.%Y %H:%M:%S')for i in range(df.shape[0])])
    df = df[['time2', 'distance','latitude', 'longitude', 'Sensor[Single IR Spot 1]']+ temperatures_MOBA]
    df=df.rename(columns = {'time2':'time'})
    df = df.drop(labels=temperatures_MOBA[-1], axis=1)
    df = df.drop_duplicates('distance')
    df=df.reset_index(drop=True)
    # str1="\n All data used \n"
    # res=0
    str2=len(df)
    dftest=df.copy()    
    for n in range(len(temperatures_MOBA)-1): 
        # print(df[df['T{}'.format(n)] > 1000])
        df = df.drop(df[df['T{}'.format(n)] > 1000].index)
        df=df.reset_index(drop=True)
        # str1 =
    res=str2-len(df) 
    if    res >=1 :
            #res virker ikke...
        str1="\n"+str(res)+" Row(s) removed because temperatures > 1000\N{DEGREE SIGN}C \n"
        # str1="Rows are removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    else:
        str1="\nAll data used (T<1000\N{DEGREE SIGN}C) \n"
    
    return df, str1, res   

#%% ALL

def _read_all(filename):

    StartLine=0
    try:
        with open(filename, newline='',encoding='cp1252') as f:    
            Tcsv_reader = csv.reader(f)
            for i in range(40):
                line1 = next(Tcsv_reader)
                # print(line)
                StartLine=StartLine+1
                for kk in line1:
                    # if k.isnumeric():
                    if kk.count('Time')==1 or kk.count('Tidspunkt')==1 or kk.count('Distance')==1 or kk.count('Time;')==1 or kk.count('Distance;')==1 or kk.count('time')==1 or kk.count('Stationing')==1:
                        print(StartLine)
                        print(kk)
                        lline1=kk
                       
                       
                        startline1=StartLine
                        break
    except:
        #print(line1)
        print('Fail')        
    print('111111111111111')   
    StartLine=0
    try:
        with open(filename, 'r',encoding='cp1252') as f:
            lineK=f.readlines()
            print(lineK)
            for i in range(40):
                
                line1=lineK[i]
                
                StartLine=StartLine+1
                
                if line1.count('Time')==1 or line1.count('Tidspunkt')==1 or line1.count('Distance')==1 or line1.count('Time;')==1 or line1.count('Distance;')==1 or line1.count('time')==1 or line1.count('Stationing')==1:
            
                    print(StartLine)
                    print(line1)
                    lline=line1
                    startline=StartLine
                    # break
                
    except:
        print('Fail')   

    if startline>=startline1:
        startline=startline
    # lline=lline
    else:
        startline=startline1    
        lline=lline1


    with open(filename, newline='',encoding='cp1252') as f:
       Tcsv_reader = csv.reader(f)
       for i in range(40):
          line = next(Tcsv_reader)
               
       for k in line:   
           if k.count('"')>=1:
              q=0
              DQ=True
           else:    
              q=csv.QUOTE_NONE
              DQ=False
     

    with open(filename, newline='',encoding='cp1252') as f:
       # Tcsv_reader = csv.reader(f,delimiter='|')
       delim2=''
       try:
           dialect = csv.Sniffer().sniff(f.read(1024))
           delim=dialect.delimiter
           delim2=delim
       except:
           if lline.count(';')>=1:
               delim=';'
           elif lline.count(',')>=1:
               delim=','
           else:
               delim=','    
      
    if delim!=',' and delim!=';':
       print('dialect fail')
       if lline.count(';')>=1:
           delim=';'
       elif lline.count(',')>=1:
           delim=','
       else:
           delim=delim2


    with open(filename, 'r',encoding='cp1252') as f:
       lineK=f.readlines()

    # if lineK[50].count('.')==3 and lineK[50].count(',')>50:
    #    dec='.'
    #    th=','
    if lineK[50].count(';')>10 and delim==';' and lineK[50].count('.')>10:    
        dec='.'
        th=',' 

    elif lineK[50].count(';')>10 and delim==';' and lineK[50].count(',')>10:    
        dec=','
        th='.'       
    elif lineK[50].count(',')>lineK[50].count('.') and lineK[50].count(':')<=3:
        dec='.'
        th=','   
    elif lineK[50].count(',')>lineK[50].count('.'):
       dec=','
       th='.'
    else:
       dec='.'
       th=','


    try:

       ddf = pd.read_csv(filename, skiprows=startline+1,delimiter=delim,decimal=dec,quoting=q,doublequote=DQ, quotechar='"')     
    except:
       ddf = pd.read_csv(filename, skiprows=startline+1,encoding='cp1252',delimiter=delim,decimal=dec,quoting=q,doublequote=DQ,quotechar='"') 
    r,c=ddf.shape

    try:
        condi=ddf.iloc[1][1].astype(str).count('"')>=2
    except:
        try:
            condi=ddf.iloc[1][1].count('"')>=2
        except:    
            condi=ddf.iloc[1][0].count('"')>=2

    print('222222')
    if c<2 or condi:
       q=3
       try:

           ddf = pd.read_csv(filename, skiprows=startline+1,delimiter=delim,decimal=dec,quoting=q,doublequote=DQ, quotechar='"')     
       except:
           ddf = pd.read_csv(filename, skiprows=startline+1,encoding='cp1252',delimiter=delim,decimal=dec,quoting=q,doublequote=DQ,quotechar='"') 


       r,c=ddf.shape

    # r=len(lineK)-startline # måde at før nrows på uden at skulle loade det hele i ddf
# Columns information

    dis=re.search('dist|Dist|Stat|stat',lline)
    tim=re.search('Tim|tim|Tids|tids',lline)   
    lat=re.search('lat|Lat|Bred|bred',lline)
    long=re.search('lon|Long|Læng|læng',lline)
    sig=re.search('Signa|signa|qual|Qual',lline)
    alt=re.search('Alti|alti',lline)
    hum=re.search('Humi|humi',lline)
    pre=re.search('Pres|pres',lline)
    air=re.search('AirT|airt',lline)
    win=re.search('Wind|wind',lline)
    sat=re.search('Sate|sate',lline)
    ind=re.search('Inde|inde',lline)
    spe=re.search('Spee|Spee',lline)

    NL=['distance']+ ['time']+['latitude']+['longitude']+['Signal quality']+['altitude']+['Humidity']+['Pressure']+['AirTemperature']+['WindSpeed']+['satellites']+['index']+['speed']

    RF=[]
    k=0
    for i in [dis,tim,lat,long,sig,alt,hum,pre,air,win,sat,ind,spe]:
    # k=k+1
        try:
            RF.append([NL[k],i.span(0)[0]])
        except:
            RF.append([NL[k],'Not used'])
        k=k+1    

    rf=[]
    for i in range(len(RF)):
        if RF[i][1]!='Not used':
            rf.append(RF[i])
    nb=[]
    for i in range(len(rf)):
        nb.append(rf[i][1])
    
    nb.sort()  

    columnInfo=[]
    for i in range(len(nb)):
        for ii in range(len(rf)):
            if nb[i]==rf[ii][1]:
                columnInfo.append(rf[ii][0])

    

    rows = 1
    for i in range(ddf.index.size):
        if pd.notna(ddf.loc[ddf.index[i]][12])==True:
            rows += 1
     # elif ddf.index[i].isdigit() is True:
     #        rows += 1
        else:
            rows +=1
        
            break
    print(rows)       
    
    temperatures_col = ['T{}'.format(n) for n in range(c)] 
       
    columns = columnInfo + temperatures_col
    
   
    if c>100:
           try:    
               df = pd.read_csv(filename, skiprows=startline+1,delimiter=delim,decimal=dec,thousands=th,names=columns)
           # df = pd.read_csv(filename, skiprows=5, delimiter=';', names=columns,thousands='.',decimal=',')
           except:
           # df = pd.read_csv(filename, skiprows=5, delimiter=';', names=columns,thousands=',',decimal='.')
               df = pd.read_csv(filename, skiprows=startline+1,encoding='cp1252',delimiter=delim,decimal=dec,thousands=th,names=columns) 


    else:
           try:    
               df = pd.read_csv(filename, skiprows=startline,delimiter=delim,decimal=dec,thousands=th,names=columns,quoting=q,doublequote=DQ, quotechar='"', nrows=rows)
           # df = pd.read_csv(filename, skiprows=5, delimiter=';', names=columns,thousands='.',decimal=',')
           except:
           # df = pd.read_csv(filename, skiprows=5, delimiter=';', names=columns,thousands=',',decimal='.')
               df = pd.read_csv(filename, skiprows=startline,encoding='cp1252',delimiter=delim,decimal=dec,thousands=th,names=columns,quoting=q,doublequote=DQ, quotechar='"',nrows=rows) 

    for col in df.columns:
        if all(df[col].isna())==True:
            df=df.drop(columns=[col])


    # if all(df[df.columns[-1]].isna())==True:
    #         df=df.drop(columns=[df.columns[-1]])
    #For moba 2 reader...
    # if df.columns[-1]=='T36':
    #     df=df.drop(columns=[df.columns[-1]])
    #     df=df.drop(columns=[df.columns[-1]])
    #     temperatures_col=temperatures_col[0:35]
                
        # print(df)
        # print(df.columns)

    #Definere ny temperatures_col, med korrekt antal T columns
    temperatures_col=['T{}'.format(n) for n in range([x for x in ''.join(df.columns)].count('T'))]

    if lline.count('"')>=1:
            for col in set(temperatures_col) | {'distance', 'latitude', 'longitude'}:
                try:
                    try:
                        df[col] = df[col].astype('str').apply(lambda x:x.strip('"')).astype('float')
                    except:
                    
                        df[col] = df[col].astype('str').apply(lambda x:x.strip('"'))
                except:
                    try:
                        df.replace('""',np.nan,inplace=True)
                        df[col] = df[col].astype('str').apply(lambda x:x.strip('"')).astype('float')
                    except:
                        df.replace('""',np.nan,inplace=True)
                        df[col] = df[col].astype('str').apply(lambda x:x.strip('"'))

    
    if lline[0:40].count('dd.MM.yyyy HH:mm:ss')==1:
            df['time']=pd.to_datetime(df.time, format='mixed')# format='%d.%m.%Y%H:%M:%S')
    elif lline.count('Time stamp')==1:
        df['time'] = pd.to_datetime(df.time, format='%H:%M:%S')
        print('Moba time...')
    elif line[0].count('UTC')==0:        
            df['time']=[x.replace(' ','') for x in df.time]
            # df['time'] = pd.to_datetime(df.time, format='%H:%M:%S')    
            df['time'] = pd.to_datetime(df.time, format="mixed")
            df['time'] =pd.to_datetime(df['time'],format).apply(lambda x: x.time()) # added 080725 to remove the date part of the datetime format.
            

            
    else:
            df['time'] = df['time'].astype('str').apply(lambda x:x.strip('"'))
            try:
                df['time'] = pd.to_datetime(df.time, format="%d-%m-%Y %H:%M:%S UTC + 02:00")
                # df['time'] = pd.to_datetime(df.time, format="mixed")
            except:
                df['time'] = pd.to_datetime(df.time, format="%d/%m/%Y %H:%M:%S UTC + 02:00")
                # df['time'] = pd.to_datetime(df.time, format="dayfirst mixed")
       
    DL=[]
    for row in df.index:
        if any(df.iloc[row]=='Invalid')==True:
            DL.append(row)
    df=df.drop(index=DL)
    df.reset_index(drop=True,inplace=True)   

    try:
        for n in range([x for x in ''.join(df.columns)].count('T')):
            df['T{}'.format(n)] =pd.to_numeric(df['T{}'.format(n)])
    except:
        for n in range([x for x in ''.join(df.columns)].count('T')-1):
            df['T{}'.format(n)] =pd.to_numeric(df['T{}'.format(n)])
    
    str2=[]
    str2=len(df)
   # dftest=df.copy()    
    # for n in range(len(temperatures_col)-1):
    for n in range([x for x in ''.join(df.columns)].count('T')-1):    
       # print(df[df['T{}'.format(n)] > 1000])
       df = df.drop(df[df['T{}'.format(n)] > 1000].index)
       df=df.reset_index(drop=True)
       # str1 =
    res=str2-len(df) 
    if    res >=1 :
           #res virker ikke...
       str1="\n"+str(res)+" Row(s) removed because temperatures > 1000\N{DEGREE SIGN}C \n"
       # str1="Rows are removed because temperatures > 1000\N{DEGREE SIGN}C \n"
    else:
       str1="\nAll data used (T<1000\N{DEGREE SIGN}C) \n"            



    return df, str1, res


#%% 
readers = {
        'Voegele':_read_Voegele,
        'TF':_read_TF,
        'Moba':_read_Moba,
        'Default':_read_ALL}
        # 'Voegele_1':_read_voegele_1,
        # 'Voegele_2':_read_voegele_2,
        # 'TF_old':_read_TF_old,
        # 'TF_new':_read_TF_new,
        # 'TF_time':_read_TF_time,
        # 'Moba1':_read_moba1,
        # 'Moba2':_read_moba2,
        # 'Moba3':_read_moba3,
        # }


#%% TEST
#K:\DT\BBM\BEF\JLB1\Termografi\2024\Entr 2 M40d\Udlægning 29042024 udl2_2
# filename='K:\\DT\\BBM\\BEF\\JLB1\\Termografi\\2024\\Entr 2 M40d\\Udlægning 29042024 udl2_2\\KVS_20240429_NCC_VOG_Entr. 2_M40_spor 1_hoved (2).csv'

# df,str1=_read_Voegele(filename)


# filename='K:\\DT\\BBM\\BEF\\JLB1\\Termografi\\2024\\Entr 2 M40d\\Udlægning 29042024 udl1\\KVS_20240429_XBB_TF_Entr2_M40_Hoved_2ud_1.csv'

# df,str1=_read_TF(filename)


# C:/Users/B306460/Downloads

# filename='C:\\Users\\B306460\\Downloads\\14822232_20240602_082418_-_Udleagningstemperatur-24_06_03-UTC+02_00.csv'
# df,str1=_read_Voegele(filename)


