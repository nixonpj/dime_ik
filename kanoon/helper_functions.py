from collections import defaultdict


def citation_count_dict(df):
    """ Iterates through each citation set in a given dataframe and builds a dictionary with a key as the case id
    and the value as the number of times it is cited across the dataframe. Also discards the case id to prevent
    self citation incidences"""
    citation_counts = defaultdict(int)
    for row in df.itertuples():
        if str(row.cases_cited) != 'nan':
            # row.citations.discard(row.case_id)
            for citation in row.cases_cited:
                citation_counts[citation] += 1
    return citation_counts

def acts_citation_count_dict(df):
    """ Iterates through each citation set in a given dataframe and builds a dictionary with a key as the case id
    and the value as the number of times it is cited across the dataframe. Also discards the case id to prevent
    self citation incidences"""
    acts_citation_counts = defaultdict(int)
    for row in df.itertuples():
        for citation in row.acts_cited:
            acts_citation_counts[citation] += 1
    return acts_citation_counts

