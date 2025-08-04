import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

import pandas as pd
import numpy as np

def standardize(x):
    return (x - np.mean(x)) / np.std(x)

def clip(data, start_date=None, end_date=None):
    data = data.copy()
    if isinstance(data, pd.DataFrame) and "Time" in data.columns:
        data["Time"] = pd.to_datetime(data["Time"])
        data = data.set_index("Time")
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        raise ValueError("Index must be datetime-like or DataFrame must contain 'Time' column.")
    if start_date:
        data = data.loc[pd.to_datetime(start_date):]
    if end_date:
        data = data.loc[:pd.to_datetime(end_date)]
    return data

class SatelliteEDA:
    def __init__(self, satellite):
        self.satellite = satellite
        self.df_orbit   = satellite.df_orbit
        self.df_man     = satellite.df_man
        self.name       = satellite.name

    def all_with_man(self, start_date=None, end_date=None, subplots_wrap=None):
        df            = self.df_orbit
        df_manoeuvre  = self.df_man.copy()
        df_filtered   = clip(df, start_date, end_date)
        numeric_cols  = df_filtered.columns
        df_filtered[numeric_cols] = (df_filtered[numeric_cols] - df_filtered[numeric_cols].mean()) / df_filtered[numeric_cols].std()

        df_manoeuvre["manoeuvre_date"] = pd.to_datetime(df_manoeuvre["manoeuvre_date"])
        manoeuvre_dates = df_manoeuvre["manoeuvre_date"]

        if isinstance(subplots_wrap, tuple):
            row = subplots_wrap[0]
            col = subplots_wrap[1]
            fig, axes = plt.subplots(
            row, col,
            figsize=(8, len(numeric_cols)),
            sharex=True
        )
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(
            len(numeric_cols), 1,
            figsize=(8, 2 * len(numeric_cols)),
            sharex=True
        )
            
        for ax, col in zip(axes, numeric_cols):
            ax.plot(df_filtered.index, df_filtered[col])
            ax.set_ylabel(col)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=12)

            show_label = True 
            for date in manoeuvre_dates:
                if df_filtered.index.min() <= date <= df_filtered.index.max():
                    if show_label:
                        line = ax.axvline(x=date, color="blue", linestyle="--", linewidth=1, label="Manoeuvre")
                        show_label = False
                        ax.legend(loc='upper right')
                    else:
                        ax.axvline(x=date, color="blue", linestyle="--", linewidth=1)

        
        ax = axes[-1]
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())     # picks a “nice” interval
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def col_with_man(self, colname="Brouwer mean motion", start_date=None, end_date=None):
        df = self.df_orbit[colname]
        df_filtered = clip(df, start_date, end_date)
        std_series = standardize(df_filtered)
        df_filtered = pd.DataFrame({colname: df_filtered, "stdcol": std_series})

        df_manoeuvre = self.df_man.copy()
        df_manoeuvre["manoeuvre_date"] = pd.to_datetime(df_manoeuvre["manoeuvre_date"])
        manoeuvre_dates = df_manoeuvre["manoeuvre_date"]

        _, ax = plt.subplots(figsize=(4, 4))
        ax.plot(df_filtered.index, df_filtered["stdcol"], label=f"{colname} (std)")
        ax.set_ylabel(colname)
        ax.grid(True)

        first = True
        for date in manoeuvre_dates:
            if df_filtered.index.min() <= date <= df_filtered.index.max():
                if first:
                    ax.axvline(x=date, color="blue", linestyle="--", linewidth=1, label="Manoeuvre")
                    first = False
                else:
                    ax.axvline(x=date, color="blue", linestyle="--", linewidth=1)

        ax.legend()
        ax.set_xlabel("Time")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title(f"{self.name}")
        plt.tight_layout()
        plt.show()

    def col_with_man_trimmed(self, colname="Brouwer mean motion", start_date=None, end_date=None, clip_value=2):
        """
        Plot the standardized and trimmed version of the specified column (default: 'Brouwer mean motion'),
        and annotate all known manoeuvre events.
        """
        # 1. Select the target column and clip the date range
        df = self.df_orbit[colname]
        df_filtered = clip(df, start_date, end_date)

        # 2. Standardize and trim the values
        std_series = standardize(df_filtered)
        std_series_trimmed = std_series.clip(lower=-clip_value, upper=clip_value)

        # 3. Combine into a new DataFrame
        df_filtered = pd.DataFrame({
            colname: df_filtered,
            "stdcol_trimmed": std_series_trimmed
        })

        # 4. Get manoeuvre dates
        df_manoeuvre = self.df_man.copy()
        df_manoeuvre["manoeuvre_date"] = pd.to_datetime(df_manoeuvre["manoeuvre_date"])
        manoeuvre_dates = df_manoeuvre["manoeuvre_date"]

        # 5. Plot the time series
        _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_filtered.index, df_filtered["stdcol_trimmed"], label=f"{colname} (std, trimmed)", color="tab:blue")
        ax.set_ylabel(colname)
        ax.grid(True)

        # 6. Add vertical lines for manoeuvre timestamps
        first = True
        for date in manoeuvre_dates:
            if df_filtered.index.min() <= date <= df_filtered.index.max():
                if first:
                    ax.axvline(x=date, color="blue", linestyle="--", linewidth=1, label="Manoeuvre")
                    first = False
                else:
                    ax.axvline(x=date, color="blue", linestyle="--", linewidth=1)

        ax.legend()
        ax.set_xlabel("Time")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(f"{self.name} Standardized & Trimmed {colname.title()}")
        plt.tight_layout()
        plt.show()
