import pandas as pd
from datetime import datetime, timedelta
import re

def extract_first_quote(text):
    match = re.search(r'"(.*?)"', text)
    
    if match:
        extracted_text = match.group(1)
    else:
        extracted_text = ""
    
    return extracted_text


class Satellite:
    def __init__(self, orbit_path, man_path, eda_show=False):
        """
        Initialize the SatelliteOrbit class and load data.
        :param file_path: Path to the CSV file containing orbital elements.
        """
        self.df_man, self.name = self.read_mandata(man_path)
        self.df_orbit = self.read_orbitdata(orbit_path)
        print(f"Loading dataset {self.name} Ready.")
        
        if eda_show:
            from satellite_eda import SatelliteEDA
            eda = SatelliteEDA(self)
            eda.run()

    def copy_range(self, start_time, end_time):
        df_orbit_trimmed = self.df_orbit.loc[start_time:end_time].copy()
        df_man_trimmed = self.df_man.copy()
        df_man_trimmed = df_man_trimmed[
            (df_man_trimmed['manoeuvre_date'] >= start_time) & 
            (df_man_trimmed['manoeuvre_date'] <= end_time)
        ].copy()

        new_sat = Satellite.__new__(Satellite)
        new_sat.df_orbit = df_orbit_trimmed
        new_sat.df_man = df_man_trimmed
        new_sat.name = f"{self.name}_range_{start_time.date()}_{end_time.date()}"
        return new_sat


    def read_orbitdata(self, file_path):
        """
        Reads a multi-line manoeuvre data file and combines each entry into a single row.
        :param file_path: Path to the data file.
        :return: A pandas DataFrame with structured data.
        """
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "Time"

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
            df.drop(columns = ['RawData'], inplace=True)
        else:
            df['year'] = df['RawData'].apply(lambda x: int(x[6:10]))
            df['beg_day'] = df['RawData'].apply(lambda x: int(x[11:14]))
            df['manoeuvre_date'] = pd.to_datetime(
                df.apply(lambda row: datetime(row['year'], 1, 1) + timedelta(days=row['beg_day'] - 1), axis=1)
            )
            df.drop(columns=["RawData", "year", "beg_day"], inplace=True)   

        # Sort values
        df.sort_values(by=["manoeuvre_date"], inplace=True)

        return df, satellite_name