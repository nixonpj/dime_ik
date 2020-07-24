from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import pandas as pd
import ik_parsing as ik
import os
import zipfile
import time
import sys

list_of_banks = pd.read_csv('../bank_names.csv', header=None)
list_of_banks = list_of_banks[0].tolist()
list_of_banks = ik.preprocess_company_names(list_of_banks, bank=True)


def extract_to_df(zip_folder, list_of_textfiles):
    files = []
    titles = []
    dates = []
    courts = []
    cnr_nums = []
    case_nums = []
    petitioner = []
    respondent = []
    petitioner_advocates = []
    respondent_advocates = []
    judges = []
    banks = []
    citations = []
    count = 0
    for filename in list_of_textfiles:
        count += 1
        # if count > 5:
        #     break
        with zip_folder.open(filename) as f:
            soup = BeautifulSoup(f, 'html.parser')
            citation_banner = soup.find('div', 'doc_cite')
            if citation_banner:
                citation_banner.extract()
            text = soup.get_text().lower()
            files.append(filename)
            try:
                title, date = soup.title.string.split(' on ')
                date = datetime.strptime(date, "%d %B, %Y")
                titles.append(title)
                dates.append(date)
            except (AttributeError, ValueError):
                titles.append(np.nan)
                dates.append(np.nan)
            try:
                courts.append(soup.find("div", {"class": "docsource_main"}).text)
            except AttributeError:
                courts.append(np.nan)
            case_nums.append(ik.extract_case_num(text))
            petitioner.append(ik.extract_petitioner(text))
            respondent.append(ik.extract_respondent(text))
            pet_adv, resp_adv = ik.extract_petitioner_advocate(text)
            petitioner_advocates.append(pet_adv)
            respondent_advocates.append(resp_adv)
            cnr_nums.append(ik.extract_cnr(text))
            judges.append(ik.extract_judges(text, soup))
            banks.append(ik.extract_banks(text, list_of_banks))
            citations.append(ik.extract_citations(filename, str(soup)))
    df = pd.DataFrame({'file': files, 'cnr_num': cnr_nums, 'title': titles, 'date': dates,
                       'court': courts, 'case_number': case_nums, 'petitioner': petitioner, 'respondent': respondent,
                       'petitioner_advocate': petitioner_advocates, 'respondent_advocate': respondent_advocates,
                       'judge': judges, 'banks': banks, 'citations': citations})
    return df


head, tail = os.path.split(os.getcwd())
list_of_court_folders = os.listdir(os.path.join(head, 'IK_data'))


def extract_data(court, year_start=2000, year_end=2019):
    start_time = time.time()
    list_of_dfs = []
    with zipfile.ZipFile(os.path.join(head, 'IK_data', court + '.zip'), 'r') as zip_folder:
        for year in range(year_start, year_end + 1):
            list_of_textfiles = [textfile for textfile in zip_folder.namelist() if
                                 textfile.startswith(f"{court}/{year}/") and textfile.endswith(".txt")]
            print(f"{court} court done for year {year} at time --- {(time.time() - start_time)} seconds ---")
            sys.stdout.flush()
            df_court_year = extract_to_df(zip_folder, list_of_textfiles)
            list_of_dfs.append(df_court_year)
    res_df = pd.concat(list_of_dfs).reset_index(drop=True)
    return res_df


# def extract_data(court, year):
#     with zipfile.ZipFile(os.path.join(head, 'IK_data', court + '.zip'), 'r') as zip_folder:
#         list_of_textfiles = [textfile for textfile in zip_folder.namelist() if
#                              textfile.startswith(f"{court}/{year}/") and textfile.endswith(".txt")]
#         df = extract_to_df(zip_folder, list_of_textfiles)
#         # df.to_csv(court+'.csv')
#         return df





