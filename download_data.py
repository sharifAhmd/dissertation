import os
import csv
import json
import logging
import shutil
import pandas as pd
import urllib.request
from typing import Tuple
logging.basicConfig(level=logging.DEBUG)

missing_infos = {}

def get_year_range(start_yr: int, end_yr: int) -> list[int]:
    # to get the year's based on range
    return [yr + 1 for yr in range(start_yr - 1, end_yr)]

def make_folder(years: list[int]):
    #create folder to hold the data
    for yr in years:
        os.makedirs(f"data/{yr}", exist_ok = True)

def get_month_ranges_in_year(year: int) -> Tuple[list[tuple()], list[str]]:
    #get the month's date range from january to december in a given year
    month_numbers = []
    dates_in_month = []
    end_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq='M')
    for end_date in end_dates:
        end_date = str(end_date).strip(" ")[:10]
        splitted_end_date = end_date.split("-")
        month_numbers.append(splitted_end_date[1])
        dates_in_month.append((f"{year}{splitted_end_date[1]}01", ("").join(splitted_end_date)))
    
    return dates_in_month, month_numbers
 
def build_file_name(start: str, end: str) -> str:
    #build the file name of the csv file
    return f"g15_xrs_1m_{start}_{end}.csv"

def build_file_URL(year: int, month: str, file_name: str) -> str:
    #build the URL for file path of the csv file
    return f"https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/{year}/{month}/goes15/csv/{file_name}"

def download_file(year: int, url: str, file_name: str) -> None:
    # download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(url) as response, open(f"data/{year}/{file_name}", 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

def get_continuous_ranges(nums: list) -> list[Tuple]:
    # helper function to get continuous interval
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))
    
def get_missing_info(file_path: str) -> Tuple[list[str], list[int]]:
    # get different missing data informations
    list_missing_intervals = []
    list_missing_length = []
    
    df = pd.read_csv(file_path)
    df_missing_values = df.loc[df['A_AVG'] == -99999]
    missing_indexes = df_missing_values.index.tolist()

    intervals = get_continuous_ranges(missing_indexes)
    for interval in intervals:
        st =  df.iloc[[interval[0]]].values.tolist()[0][0]
        end = df.iloc[[interval[1]]].values.tolist()[0][0]
        list_missing_intervals.append(f"{st} to {end}")
        list_missing_length.append(interval[1] - interval[0] + 1)
    
    return list_missing_intervals, list_missing_length

def pre_process_data(year: int, month: str, file_name: str) -> None:
    """
    Extract the important rows and save them as csv
    Also get some information about missing values
    """
    data_dict = {}
    yearly_missing_dict = {}
    missing_row_counter = 0
    
    csv_filename_with_dir = f"data/{year}/{file_name}"
    starting_point_found = False
    
    starting_point = ['time_tag', 'A_QUAL_FLAG', 'A_NUM_PTS', 'A_AVG', 'B_QUAL_FLAG', 'B_NUM_PTS', 'B_AVG']
    for col in starting_point:
        data_dict[col] = []
    
    with open(csv_filename_with_dir, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if row == starting_point:
                starting_point_found = True
                continue
    
            if starting_point_found:
                for i, col in enumerate(starting_point):
                    data_dict[col].append(row[i])

    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(csv_filename_with_dir, index = False)
    
    yearly_missing_dict["number of total rows"] = df.shape[0]
    list_missing_intervals, list_missing_length = get_missing_info(csv_filename_with_dir)
    yearly_missing_dict["number of missing rows"] = sum(list_missing_length)
    yearly_missing_dict["missing rows each interval"] = list_missing_length
    yearly_missing_dict["missing intervals"] = list_missing_intervals
    missing_infos[month] = yearly_missing_dict
    
    with open(f"data/{year}_summary.json", "w") as fp:
        json.dump(missing_infos, fp)

def download_data_by_year(year: int) -> None:
    """
    Need some steps to get the data from www.ncei.noaa.gov
        -> Need date range of the year by month
        -> Need the filename
        -> Need the URL for file in NOAA website
        -> Download the data
    """
    date_range_in_month, month_numbers = get_month_ranges_in_year(year)
    for i, date_range in enumerate(date_range_in_month):
        file_name = build_file_name(date_range[0], date_range[1])
        file_url = build_file_URL(year, month_numbers[i], file_name)
        download_file(year, file_url, file_name)
        logging.info(f"Download completed of {file_name}")
        pre_process_data(year, month_numbers[i], file_name)
        logging.info(f"Pre-processing completed of {file_name}")

    
def main() -> None:
    #main function caller
    start_yr = int(input("Enter start year:"))
    end_yr = int(input("Enter end year:"))

    if start_yr > end_yr:
        logging.error("Start year can not be greater than end date")
        return
    
    years = get_year_range(start_yr, end_yr)
    make_folder(years)
    for yr in years:
        download_data_by_year(yr)
        logging.info(f"Data downloaded for year {yr}")

if __name__ == '__main__':
    main()