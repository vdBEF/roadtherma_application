import re
import datetime
import csv
import pandas as pd
import numpy as np

temperatures_voegele = ['T{}'.format(n) for n in range(52)]
VOEGELE_BASE_COLUMNS = ['time', 'distance', 'latitude', 'longitude']

def _convert_vogele_timestamps(df, formatting):
    df['time'] = pd.to_datetime(df.time, format=formatting)


def _read_vogele_example(filename):
    """ Old example code. This is probably not going to be used. """
    columns = VOEGELE_BASE_COLUMNS + temperatures_voegele
    df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',')
    _convert_vogele_timestamps(df, "%d.%m.%Y %H:%M:%S")
    return df


def _read_vogele_M119(filename):
    """
    Data similar to the example file.
    NOTE removed last line in the file as it only contained 'Keine Daten vorhanden'.
    """
    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
    df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',')
    _convert_vogele_timestamps(df, "%d.%m.%Y %H:%M:%S")
    return df


def _read_voegele_roller_example(filename):
    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
    df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',')
    _convert_vogele_timestamps(df, "%d-%m-%Y %H:%M:%S UTC + 02:00") # NOTE only difference between this and _read_vogele_M30 is the usage of "-" instead of "/" here (ffs!)
    return df

#def _read_voegele_roller_example(filename):
#    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
#    df = pd.read_csv(filename, skiprows=2, delimiter=',', names=columns, decimal='.',encoding='865')
#    _convert_vogele_timestamps(df, "%d/%m/%Y %H:%M:%S UTC + 02:00") # NOTE only difference between this and _read_vogele_M30 is the usage of "-" instead of "/" here (ffs!)
#    return df

# def _read_voegele_2(filename):
#     columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
#     df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',',encoding='865')
#     _convert_vogele_timestamps(df, "%H:%M:%S") # NOTE only difference between this and _read_vogele_M30 is the usage of "-" instead of "/" here (ffs!)
#     return df


def _read_vogele_M30(filename):
    """
    NOTE removed last line in the file as it only contained 'No data to display'.
    """
    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
    df = pd.read_csv(filename, skiprows=3, delimiter=',', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True, encoding='latin-1')
    for col in df.columns:
        if col == 'time':
            df[col] = df[col].apply(lambda x:x.strip('"'))
        if col in set(temperatures_voegele) | {'distance', 'latitude', 'longitude'}:
            df[col] = df[col].astype('str').apply(lambda x:x.strip('"')).astype('float')
    _convert_vogele_timestamps(df, "%d/%m/%Y %H:%M:%S UTC + 02:00") # NOTE only difference between this and _read_vogele_taulov is the UTC-part here (ffs!)
    return df

def _read_vogele_taulov(filename):
    """
    NOTE removed last line in the file as it only contained 'No data to display'.
    """
    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
    df = pd.read_csv(filename, skiprows=3, delimiter=',', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)
    for col in df.columns:
        if col == 'time':
            df[col] = df[col].apply(lambda x:x.strip('"'))
        if col in set(temperatures_voegele) | {'distance', 'latitude', 'longitude'}:
            df[col] = df[col].astype('str').apply(lambda x:x.strip('"')).astype('float')
    _convert_vogele_timestamps(df, "%d/%m/%Y %H:%M:%S")
    return df


def _read_TF_old(filename):
    temperatures = ['T{}'.format(n) for n in range(141)]
    columns = ['distance'] + temperatures + ['distance_again']
    df = pd.read_csv(filename, skiprows=7, delimiter=',', names=columns)
    del df['distance_again']
    del df['T140'] # This is the last column of the dataset (which is empty)
    return df


def _read_TF_new(filename):
    temperatures = ['T{}'.format(n) for n in range(281)]
    columns = ['distance'] + ['time'] + ['latitude'] + ['longitude'] + temperatures
    df = pd.read_csv(filename, skiprows=7, delimiter=',', names=columns)
    df['time'] = [x[:-6] for x in df.time]
    df['time'] = pd.to_datetime(df.time, format=' %Y-%m-%dT%H:%M:%S')
    del df['T280']
    return df

def _read_TF_notime(filename):
    temperatures = ['T{}'.format(n) for n in range(281)]
    columns = ['distance'] + ['latitude'] + ['longitude'] + temperatures
    df = pd.read_csv(filename, skiprows=7, delimiter=';', names=columns)
    df['time'] = pd.DataFrame({'time':pd.date_range('01-01-2012 23:50', periods=int(len(df)), freq='1min')})
    del df['T280']
    return df

def _read_TF_time(filename):
    temperatures = ['T{}'.format(n) for n in range(281)]
    columns = ['distance'] +['time'] + ['latitude'] + ['longitude'] + temperatures
    df = pd.read_csv(filename, skiprows=7, delimiter=';', names=columns)
    df['time']=[x.replace(' ','') for x in df.time]
    df['time'] = pd.to_datetime(df.time, format='%H:%M:%S')
    del df['T280']
    # for n in range(len(temperatures)): 
    #     df = df.drop(df[df['T{}'.format(n)] > 1000].index)
    #     df=df.reset_index(drop=True)  
    return df


def _read_TF_time_K(filename):
    temperatures = ['T{}'.format(n) for n in range(280)]
    columns = ['distance'] +['time'] + ['latitude'] + ['longitude'] + temperatures
    df = pd.read_csv(filename, skiprows=7, delimiter=';', names=columns,thousands='.',decimal=',')
    df['time']=[x.replace(' ','') for x in df.time]
    df['time'] = pd.to_datetime(df.time, format='%H:%M:%S')
    #del df['T280']
    
    
    # for n in range(len(temperatures)): 
    #     df = df.drop(df[df['T{}'.format(n)] > 1000].index)
        # df=df.reset_index(drop=True)
    str2=[]
    for n in range(len(temperatures)):
        if (df['T{}'.format(n)].max())<1000:
            str2.append("DB")
        else:
            # str2[n]="Række(r) er fjernet pga temperatur højere end 1000C" 
            str2.append("DS")
        
        res=str2.count("DS")
        if    res >=1 :
            str1="\n Række(r) fjernet pga. temperatur > 1000\N{DEGREE SIGN}C \n"
        else:
            str1="\n Alt data er brugt (T<1000\N{DEGREE SIGN}C) \n"    
    for n in range(len(temperatures)): 
        df = df.drop(df[df['T{}'.format(n)] > 1000].index)
        df=df.reset_index(drop=True)    
        
    return df, str1
     



def _read_TF_time_new(filename):
    temperatures = ['T{}'.format(n) for n in range(36)]
    columns = ['distance'] +['time'] + ['latitude'] + ['longitude'] + temperatures
    df = pd.read_csv(filename, skiprows=2, delimiter=',', names=columns)
    df['time']=[x.replace(' ','') for x in df.time]
    df['time'] = pd.to_datetime(df.time, format='%H:%M:%S')
    del df['T35']
    

    return df

def _sensors_moba(filename):
    sensors = pd.read_csv(filename, sep=';', skiprows=13, nrows=1)
    sensors.columns = ['name', 'number', 'none']
    n_sens = sensors.number + 1
    return n_sens


def _temperatures_moba(filename):
    sensors = pd.read_csv(filename, sep=';', skiprows=13, nrows=1)
    sensors.columns = ['name', 'number', 'none']
    n_sens = sensors.number + 1
    temperatures_MOBA = ['T{}'.format(n) for n in range(int(n_sens))]
    return temperatures_MOBA


MOBA_BASE_COLUMNS = ['index','distance', 'speed','temporary_time', 'longitude', 'latitude']


def _rows_moba(filename):
    test = pd.read_csv(filename, index_col = False, sep=';', skiprows=27)
    rows = 0
    for i in range(test.Index.size):
        if type(test.Index[i]) == int:
            rows += 1
        elif test.Index[i].isdigit() is True:
            rows += 1
        else:
            break
    return rows


def _read_moba(filename):
    with open(filename, newline='') as f:
        csv_reader = csv.reader(f)
        _csv_headings = next(csv_reader)
        first_line = next(csv_reader)

    date = first_line[1].split(' ')
    date[0] = re.sub(r'\s+', '', first_line[2])
    date[1] = str(datetime.datetime.strptime(date[1], '%B').month)
    date = '-'.join(date)
    temperatures_MOBA = _temperatures_moba(filename)
    columns = MOBA_BASE_COLUMNS + ['signal_quality', 'satellites'] + temperatures_MOBA
    df = pd.read_csv(filename,
            skiprows=28,
            nrows=_rows_moba(filename),
            delimiter=';',
            names=columns,
            quoting=csv.QUOTE_NONE,
            quotechar='"',
            doublequote=True)
    df = df.drop(labels='index', axis=1)
    df['date'] = date
    df['temporary_time'] = df['date'] + [' '] + df['temporary_time']
    df = df.drop(labels='date', axis=1)
    df.insert(loc=2, column='time', value=[datetime.datetime.strptime(df['temporary_time'][i],'%Y-%m-%d %H:%M:%S')for i in range(df.shape[0])])
    df['time'] = pd.to_datetime(df['time'])
    df = df.drop(labels='temporary_time', axis=1)
    df = df.drop(labels='speed', axis=1)
    df = df.drop(labels='satellites', axis=1)
    df = df[['time', 'distance','latitude', 'longitude', 'signal_quality']+ temperatures_MOBA]
    df[temperatures_MOBA]=df[temperatures_MOBA].apply(pd.to_numeric, errors='coerce', axis=1).fillna(0, downcast='infer')
    df = df.drop(labels=temperatures_MOBA[-1], axis=1)
    return df

def _sensors_moba2(filename):
    sensors2 = pd.read_csv(filename, sep = ';', skiprows=5, nrows=1)
    sensors2=sensors2.iloc[:, list(range(2)) + [-1]]
    sensors2.columns = ['name', 'number', 'none']
    n_sens2 = sensors2.number 
    return n_sens2


def _temp_MOBA2(filename):
    sensors2 = pd.read_csv(filename, sep = ';', skiprows=5, nrows=1)
    sensors2=sensors2.iloc[:, list(range(2)) + [-1]]
    sensors2.columns = ['name', 'number', 'none']
    n_sens2 = sensors2.number 
    temperatures_MOBA2 = ['T{}'.format(n) for n in range(int(n_sens2))]
    return temperatures_MOBA2

#temperatures_MOBA = ['T{}'.format(n) for n in range(_sensors_moba(filename))]
MOBA_BASE_COLUMNS2 = ['time','distance', 'latitude', 'longitude', 'altitude','humidity','pressure','air temp','wind speed']

def _rows_moba2(filename):
    test2 = pd.read_csv(filename, sep = ';', skiprows=33)
    rows2 = 0
    for i in range(int(len(test2['Altitude [m]']))):
        if np.isnan(test2['Altitude [m]'][i]) == False:
            rows2 += 1
        else: 
            break
    return rows2

def _read_moba2(filename):
    temperatures_MOBA2 = _temp_MOBA2(filename)
    columns = MOBA_BASE_COLUMNS2 + temperatures_MOBA2 + ['ScreedWidthLeft', 'ScreedWidthRight'] 
    df = pd.read_csv(filename, skiprows=34,nrows=_rows_moba2(filename), delimiter=';', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)    
    df['time']=[datetime.datetime.strptime(df['time'][i],'%d.%m.%Y %H:%M:%S')for i in range(df.shape[0])] 
    return df

def _sensors_moba3(filename):
    sensors = pd.read_csv(filename, sep=';', skiprows=5, nrows=1)
    sensors.columns = ['name', 'number']
    n_sens = sensors.number + 1
    return n_sens

def _temperatures_moba3(filename):
    sensors = pd.read_csv(filename, sep=';', skiprows=5, nrows=1)
    sensors.columns = ['name', 'number']
    n_sens = sensors.number + 1
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
    with open(filename, newline='') as f:
        csv_reader = csv.reader(f)
        _csv_headings = next(csv_reader)
        first_line = next(csv_reader)
    temperatures_MOBA = _temperatures_moba3(filename)
    columns = MOBA_BASE_COLUMNS + temperatures_MOBA
    df = pd.read_csv(filename,
        index_col = False,
        skiprows=34,
        delimiter=';',
        names=columns,
        quoting=csv.QUOTE_NONE,
        quotechar='"',
        doublequote=True)
    df.insert(loc=1, column='time2', value=[datetime.datetime.strptime(df['time'][i],'%d.%m.%Y %H:%M:%S')for i in range(df.shape[0])])
    df = df[['time2', 'distance','latitude', 'longitude', 'Sensor[Single IR Spot 1]']+ temperatures_MOBA]
    df=df.rename(columns = {'time2':'time'})
    df = df.drop(labels=temperatures_MOBA[-1], axis=1)
    df = df.drop_duplicates('distance')
    df=df.reset_index(drop=True)
    return df


readers = {
        'voegele_example': _read_vogele_example,
        'voegele_M119': _read_vogele_M119,
        'voegele_roller_example': _read_voegele_roller_example,
        # 'voegele_2':_read_voegele_2,
        'voegele_M30': _read_vogele_M30,
        'voegele_taulov': _read_vogele_taulov,
        'TF_old': _read_TF_old,
        'TF_new': _read_TF_new,
        'TF_notime': _read_TF_notime,
        'TF_time': _read_TF_time,
        'TF_time_new': _read_TF_time_new,
	'TF_time_K': _read_TF_time_K,
        'moba': _read_moba,
        'moba2': _read_moba2,
        'moba3': _read_moba3,
        }
