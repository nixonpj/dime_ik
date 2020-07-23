from collections import defaultdict


def citation_count_dict(df):
    """ Iterates through each citation set in a given dataframe and builds a dictionary with a key as the case id
    and the value as the number of times it is cited across the dataframe. Also discards the case id to prevent
    self citation incidences"""
    citation_counts = defaultdict(int)
    for row in df.itertuples():
        if str(row.citations) != 'nan':
            row.citations.discard(row.case_id)
            for citation in row.citations:
                citation_counts[citation] += 1
    return citation_counts

