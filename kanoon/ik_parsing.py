import numpy as np
import re
from textdistance import smith_waterman, needleman_wunsch, jaro_winkler, jaccard, levenshtein
from fuzzywuzzy import fuzz
from collections import defaultdict


# from bs4 import BeautifulSoup


def extract_cnr(text):
    try:
        cnr_num = re.search("""\s[a-zA-Z]{4}[0-9]{12}\s""", text, flags=re.IGNORECASE)
        if cnr_num:
            return cnr_num.group()
        else:
            return np.nan
    except AttributeError:
        return np.nan


def extract_case_num(text):
    # 'Case no.' not always written. Different courts have different formats
    try:
        case_num_0 = re.search(""".+case no\..+""", text, flags=re.IGNORECASE)  # [12][90][0-9]{2}$
        case_num_1 = re.search(""".+[\d]{1,6} """, text, flags=re.IGNORECASE)
        case_num = re.search("""case no\..+:(?P<case>.+\s*.*)\s*1:""", text)
        if case_num_0:
            return case_num_0.group().strip()
        else:
            return np.nan
    except AttributeError:
        return np.nan


#         print(filename, ae)

def extract_petitioner(text):
    try:
        if 'versus' in text:
            start = text.find('1:')
            end = text.find('versus')
            return text[start:end]
        #         petitioner = re.search("""(1:.+?)versus""", text, flags=re.DOTALL)
        #         if petitioner:
        #             print(petitioner.group())
        #             print(f"-"*40)
        #             return petitioner.group()
        else:
            return np.nan
    except AttributeError:
        return np.nan


def extract_respondent(text):
    try:
        if 'versus' in text:
            start = text.find('versus') + 6
            end = start + text[start:].find('advo')
            return text[start:end]
        #             print(start, end, '*'*40)
        #             if text[end:end+8]!='advocate':
        #                 print(text[end:end+8], text)
        else:
            return np.nan
    except AttributeError:
        return np.nan


def extract_petitioner_advocate(text):
    advocates = [np.nan, np.nan]
    try:
        if 'advocate for the petitioner' in text:
            start = text.find('advocate for the petitioner')
            mid = start + 27 + text[start + 27:].find('advocate')
            advocates[0] = text[start + 32:mid]
            if 'advocate for the respondent' in text:
                respondent_adv = re.match("""advocate for the r.*?:(?P<name>.+?)(before|linked)""", text[mid:],
                                          flags=re.DOTALL)
                end = mid + text[mid:].find('before')
                if respondent_adv:
                    advocates[1] = respondent_adv['name']
        #             print(text[start+28:mid], text[mid+27:end])
        # print('-' * 40)
        #             return (text[start+27:mid], text[mid+27:end])
        #         print(advocates[1])
        return advocates
    except AttributeError:
        return advocates


def extract_judges(text, html_soup):
    try:
        if "hon'ble" in text or "honourable" in text:
            judges = re.findall("""hon.*ble(.*\s.*(?:justice|j\.).*)""", text)
            if judges:
                return set(judge.replace(')', '')
                           .replace('\n', '')
                           .replace('\t', '')
                           .strip() for judge in judges)
        bench = html_soup.find("div", "doc_bench")
        if bench:
            return {bench.text.replace('Bench:', '').strip()}
        if 'judge' in text:
            judges = re.findall("""\((.*)\).*\s.*judge""", text)
            if judges:
                return set(judge.replace(')', '')
                           .replace('\n', '')
                           .replace('\t', '')
                           .strip() for judge in judges)
        if 'j.' in text:
            judges = re.findall("""\((.*?j\.)""", text)  # [a-zA-Z ,]
            if judges:
                return set(judge.replace(')', '')
                           .replace('\n', '')
                           .replace('\t', '')
                           .strip() for judge in judges)
        author = html_soup.find("div", "doc_author")
        if author:
            return {author.text
                        .replace(')', '')
                        .replace('\n', '')
                        .replace('\t', '')
                        .strip()}
        return np.nan
    except AttributeError:
        return np.nan


def extract_banks(text, bank_names):
    banks_set = defaultdict(int)
    try:
        if " bank" in text:
            # return True
            for bank_name in bank_names:
                if bank_name in text:
                    banks_set[bank_name] += 1
            return banks_set if banks_set else np.nan
        return np.nan
    except AttributeError:
        return np.nan


def extract_state(text):
    keywords = ['state', 'government', 'commissioner', 'national', 'india', 'indian', 'public', 'magistrate']
    try:
        return any(keyword in text for keyword in keywords)
    except AttributeError:
        return np.nan


def extract_citations(filename, text):
    try:
        citation_set = set(re.findall("""/doc/(\d*)""", text, flags=re.IGNORECASE))
        acts_set = set(re.findall("""/doc/(\d*).*(?:section|constitution|penal|act|code).*<""", text, flags=re.IGNORECASE))
        # print(citations)
        citation_set.discard('')
        # Every case cites itself for newer cases due to website structure. Comment below line if you do want that.
        citation_set.discard(filename[filename.rfind('/')+1:-4])
        citation_set = citation_set - acts_set
        # if not citations:
        #     citations = np.nan
        # if not acts:
        #     acts = np.nan
        return citation_set, acts_set
    except AttributeError:
        return np.nan


def match_business(text, list_of_company_names):
    """one wall of petitioner name text to be compared to a large list of company names to find best match"""
    # max_score, max_company_name = 0, np.nan
    max_score_jw, max_company_name_jw = 0, np.nan
    for company in list_of_company_names:
        if company in text:
            return 1, company
        score = fuzz.token_set_ratio(text, company)
        # print(text, company, score, max_score)
        if score > 0.75:
            score_jw = jaro_winkler.normalized_similarity(text, company)
            if score_jw > max_score_jw:
                max_score_jw, max_company_name_jw = score_jw, company
    print('1 done', max_score_jw, max_company_name_jw)
    return max_score_jw, max_company_name_jw


def match_business_tf(text, list_of_company_names, tf):
    """one wall of petitioner name text to be compared to a large list of company names to find best match"""
    max_overall_score, max_overall_company_name = -1, np.nan
    for company in list_of_company_names:
        max_company_score, max_company_name = -1, np.nan
        if company in text:
            return 1, company
        for company_word in company.split():
            max_word_similarity_score, max_word_similarity = -1, np.nan
            # if company_word in text:
            #     max_word_similarity_score, max_word_similarity = 1, word
            for word in text.split():
                word_similarity_score = levenshtein.normalized_similarity(word, company_word)
                # print(word_similarity_score, word, company_word, '--', max_word_similarity_score, max_word_similarity, max_company_score, max_company_name, max_overall_score, max_overall_company_name)
                if word_similarity_score > max_word_similarity_score:
                    max_word_similarity_score = word_similarity_score
                    max_word_similarity = word
            # if max_word_similarity_score > 0.8:
            # print("------", max_word_similarity, '--', tf[max_word_similarity], company_word, '--', max_word_similarity_score, max_word_similarity, max_company_score, max_company_name, max_overall_score, max_overall_company_name)
            max_company_score += (1 / tf[max_word_similarity]) * max_word_similarity_score
            max_company_name = company
        if max_company_score > max_overall_score:
            max_overall_score = max_company_score
            max_overall_company_name = max_company_name
    return max_overall_score, max_overall_company_name


def preprocess_company_names(list_of_companies, bank=False):
    list_of_companies = [
        company.lower().replace('private', '').replace('limited', '').replace('pvt', '').replace('ltd', '')
        if str(company) != 'nan' else np.nan for company in list_of_companies]
    # if bank:
    #     list_of_companies = [company.replace('bank', ' ') for company in list_of_companies]
    list_of_companies = [company.replace(',', ' ').replace('.', ' ').replace('-', ' ').replace(
        '&', ' ').replace(':', ' ').replace('(', ' ').replace(')', ' ')
                         if str(company) != 'nan' else np.nan for company in list_of_companies]
    list_of_companies = [company.strip() if str(company) != 'nan' else np.nan for company in list_of_companies]
    return list_of_companies


def tf_idf():
    pass
