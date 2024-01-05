# %%Libs
from pybliometrics.scopus import ScopusSearch
import pandas as pd
from pprint import pprint
import numpy as np
from tqdm.autonotebook import tqdm

# %% read metadata
univ_list = pd.read_csv("data/uni_metadata.csv")
print("Successfully imported the list of universities!")
univ_ids = univ_list["Affiliation_Id"]
all_subjects = pd.read_csv("data/subject_areas.csv")
subj_areas = all_subjects[all_subjects["Use"] == "Y"]["Search key"].to_list()
print("Successfully imported the list of subject areas!")
# Only the first 12 keywords are supported. Choose wisely :)
keywords = pd.read_csv("data/keywords.csv")["Keywords"][0:12].to_list()
print("Successfully imported the list of keywords!")

# %% Attempt (using library)
filtered_papers = []
paper_count = 0
for university in tqdm(univ_ids):
    uniname = (
        univ_list["Institution"].loc[univ_list["Affiliation_Id"] == university].iloc[0]
    )
    for subject in tqdm(subj_areas):
        # print(subject)
        search_query = "(AF-ID ({}) AND SUBJAREA ({}) AND ({}))".format(
            university, subject, keywords[0]
        )  # Commmented out string: AND (build) AND (life) AND (cycle)
        # carry out the scopus search
        search_results = ScopusSearch(query=search_query, subscriber=True)
        # if (len(pd.DataFrame(search_results.results)) > 0):
        search_results = pd.DataFrame(search_results.results)
        # set the correct affiliation ID
        search_results = search_results.assign(afid=university)
        search_results = search_results.assign(affilname=uniname)
        # keep count of the total number of papers
        paper_count += search_results.shape[0]

        # filter out the dataframe for the specific key topics that we want
        for i in np.arange(0, len(search_results)):
            # some abstracts are left as blank and pandas reads this is as a nonetype. Use try:except to get around this nightmare
            try:
                if any([x in search_results["authkeywords"][i] for x in keywords]):
                    filtered_papers.append(list(search_results.iloc[i].values))
            except:
                pass
# stick them all into a pandas dataframe
papers = pd.DataFrame(filtered_papers, columns=list(search_results.columns))
print("Shape of data from the search results")
print(papers.shape)
# remove dupicated rows
papers = papers.drop_duplicates(subset=["eid"], keep="first")
papers = papers.reset_index(drop=True)
print("Shape of data once duplicates are removed")
print(papers.shape)
papers.head()
# save as a CSV
papers.to_csv("Results/relevant_publications.csv", index=False)

# %%
