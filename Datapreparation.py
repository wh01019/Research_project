import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import re

def extract_first_quote(text):
    match = re.search(r'"(.*?)"', text)
    
    if match:
        extracted_text = match.group(1)
    else:
        extracted_text = ""
    
    return extracted_text

class Satellite:
    def __init__(self, orbit_path, man_path,
                 batch_size = 32,
                 time_steps = 30,
                 forecast_steps = 1,
                 val_ratio = 0.2,
                 test_ratio = 0.2,
                 test_only = False,
                 standardization='standard',
                 brouwer_only = True,
                 eda_show=True):
        """
        Initialize the SatelliteOrbit class and load data.
        :param file_path: Path to the CSV file containing orbital elements.
        """
        self.df_orbit = self.read_orbitdata(orbit_path)
        self.df_man, self.name = self.read_mandata(man_path)
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.forecast_steps = forecast_steps
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.brouwer_only = brouwer_only
        self.test_only = test_only
        self.standardization = standardization

        # prepare and split data for training
        datetime = self.df_orbit["Time"]
        if brouwer_only:
            self.raw_seq = self.df_orbit["Brouwer mean motion"].values
        else:
            self.raw_seq = self.df_orbit.drop(columns=["Time"]).values
            
        # Define X and y and split the data
        self.X, self.y, self.datetime = self._prepare_data(self.raw_seq, datetime, 
                                                           self.time_steps, self.forecast_steps,
                                                           self.batch_size, self.val_ratio, self.test_ratio, 
                                                           test_only, standardization)
        if test_only is False:
            [self.X_train, self.X_val, self.X_test] = self.X
            [self.y_train, self.y_val, self.y_test] = self.y
            [self.dtime_train, self.dtime_val, self.dtime_test] = self.datetime
        
        print(f"Loading dataset {self.name} Ready for training.")
        if eda_show:
            self.info()
            self.EDA()

    def read_orbitdata(self, file_path):
        """
        Reads a multi-line manoeuvre data file and combines each entry into a single row.
        :param file_path: Path to the data file.
        :return: A pandas DataFrame with structured data.
        """
        df = pd.read_csv(file_path)
        df = df.rename(columns={"Unnamed: 0": "Time"})
        df["Time"] = pd.to_datetime(df["Time"]).dt.date
        df.sort_values(by="Time", inplace=True)
        
        # print("Orbit Data successfully loaded!")
        # print(df.head()) 
        # print(f"Data shape: {df.shape}") 
        # df.info() 
        return df

    def read_mandata(self, file_path):
        """
        Reads a multi-line manoeuvre data file and combines each entry into a single row.
        :param file_path: Path to the data file.
        :return: A pandas DataFrame with structured data.
        """
        data = []
        pre_line = ""
        fengyun = False
    
        with open(file_path, "r") as file:
            # Get the name of the satellite
            satellite_name = file.read(5)
            file.seek(0)

            if satellite_name == "GEO-E":
                fengyun = True
                satellite_name = file_path[16:20]
                for line in file:
                    line = extract_first_quote(line)
                    data.append(line)
            else:
                # Read the lines
                for line in file:
                    line = line.strip()
                    if line.startswith(satellite_name):
                        if pre_line:
                            data.append(pre_line) # Append the previous row
                        pre_line = line # Save the new row
                    else:
                        pre_line += " " + line 
                
                if pre_line:
                    data.append(pre_line) # Add the last line

        # Convert list of concatenated entries into a DataFrame
        df = pd.DataFrame(data, columns=["RawData"])


        # Get the name of the satellite
        df['Name'] = satellite_name
        if fengyun:
            df['manoeuvre_date'] = pd.to_datetime(df["RawData"].apply(lambda x: x[:10]),format = "%Y-%m-%d")
            df['manoeuvre_date'] = df['manoeuvre_date'].dt.date
            df.drop(columns = ['RawData'], inplace=True)
        else:
            # Get the manoeuvre date
            df['year'] = df['RawData'].apply(lambda x: int(x[6:10]))
            df['beg_day'] = df['RawData'].apply(lambda x: int(x[11:14]))
            df['manoeuvre_date'] = df.apply(lambda row: datetime(row['year'], 1, 1) + timedelta(days=row['beg_day'] - 1), axis=1)
            df['manoeuvre_date'] = df['manoeuvre_date'].dt.date # Ensure same dtype as date in orbit

            # drop useless columns
            df.drop(columns=["RawData","year","beg_day"], inplace=True)

        # Sort values
        df.sort_values(by=["manoeuvre_date"], inplace=True)

        return df, satellite_name

    def info(self):
        print(f"Satellite name{self.name}.")
        print(self.df_man.info())
        if self.test_only:
            print(f"Test Set Length: {self.y.shape}.")
        else:
            print("Train Set X size: {}, y size: {}\nValidation Set X size: {}, y size: {}\nTest Set X size: {}, y size: {}"
                  .format(self.X_train.shape, self.y_train.shape,
                          self.X_val.shape, self.y_val.shape,
                          self.X_test.shape, self.y_test.shape))
        # return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def EDA(self):
        """ Visualize an orbital parameter and highlight anomalies with manoeuvre dates """
        df_sampled = self.df_orbit
    
        fig, axs = plt.subplots(len(df_sampled.columns)-1, 1, figsize=(16, 20))
        axs = axs.flatten()
        cols = list(df_sampled.columns)
        cols.remove("Time")
    
        # Loop through each subplot
        for i, col in enumerate(cols):
            if col != "Time":
                axs[i].plot(df_sampled["Time"], df_sampled[col])
                axs[i].set_ylabel(col)
        
                # Mark the manoeuvre dates on the plot if they fall within the sampled range
                for man_date in self.df_man['manoeuvre_date']:
                    # Ensure man_date is a Timestamp to compare with the index
                    if df_sampled["Time"].min() <= man_date <= df_sampled["Time"].max():  # Check if the manoeuvre date is within the range
                        axs[i].axvline(x=man_date, color='blue', linestyle='--', label='Manoeuvre Date')
        
        plt.suptitle(self.name, y=1)
        plt.tight_layout()
        plt.show()
        return fig

    def _prepare_data(self, raw_seq, raw_dtime, 
                          time_steps, forecast_steps,
                          batch_size, val_ratio, test_ratio, 
                          test_only, standardization):
            if standardization is not None:
                seq = self._standardize(raw_seq, standardization)
            else:
                seq = raw_seq
            
            if test_only:
                X, y, dtime = self._build_timeser(seq, raw_dtime, time_steps, forecast_steps)
                X = self._trim_seq(X, batch_size)
                y = self._trim_seq(y, batch_size)
                dtime = self._trim_seq(dtime, batch_size)
    
            else:
                temp_X, temp_y, temp_dtime = self._build_timeser(seq, raw_dtime, time_steps, forecast_steps)
                X_train, X_test, y_train, y_test, dtime_train, dtime_test = \
                train_test_split(temp_X, temp_y, temp_dtime, test_size=test_ratio, shuffle=False)
                X_train, X_val, y_train, y_val, dtime_train, dtime_val = \
                train_test_split(X_train, y_train, dtime_train, test_size=val_ratio, shuffle=False)
                X = [X_train, X_val, X_test]
                y = [y_train, y_val, y_test]
                dtime = [dtime_train, dtime_val, dtime_test]
                
                for i in range(len(X)):
                    X[i] = self._trim_seq(X[i], batch_size)
                    y[i] = self._trim_seq(y[i], batch_size)
                    dtime[i] = self._trim_seq(dtime[i], batch_size)
            
            return X, y, dtime
    
    def _build_timeser(self, seq, dtime, time_steps, forecast_steps):
        dim_0 = seq.shape[0] - (time_steps + forecast_steps)
        dim_1 = seq.shape[1]
        X = np.zeros((dim_0, time_steps, dim_1))

        for i in range(dim_0):
            X[i] = seq[i : i+time_steps]
        y = seq[time_steps+forecast_steps:, 0]
        dt = dtime[time_steps+forecast_steps:]

        return X, y, dt
    
    def _standardize(self, seq, method):
        if seq.ndim == 1:
            seq = seq.reshape(-1,1)
        if ('max' in method) or ('min' in method):
            scaler = MinMaxScaler()
        elif 'stand' in method:
            scaler = StandardScaler()
        scaler.fit(seq)
        seq_scaled = scaler.transform(seq)
        return seq_scaled
    
    def _trim_seq(self, seq, batch_size):
        """
        Discard elements to make the length a multiple of batch_size.
        """
        drop = seq.shape[0] % batch_size
        if drop > 0:
            return seq[:-drop]
        else:
            return seq