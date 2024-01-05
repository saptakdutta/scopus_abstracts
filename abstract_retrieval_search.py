# %%Libs
from pybliometrics.scopus import ScopusSearch
import pandas as pd
from pprint import pprint
import numpy as np
from tqdm.autonotebook import tqdm

# %% read metadata
all_subjects = pd.read_csv("data/subject_areas.csv")
subj_areas = all_subjects[all_subjects["Use"] == "Y"]["Search key"].to_list()
print("Successfully imported the list of subject areas!")
# Only the first 12 keywords are supported. Choose wisely :)
keywords = pd.read_csv("data/keywords.csv")["Keywords"][0:12].to_list()
print("Successfully imported the list of keywords!")

# %% Attempt (using library)
filtered_papers = []
paper_count = 0
for subject in tqdm(subj_areas, desc="Downloading papers for the subject areas: "):
    # print(subject)
    search_query = "(SUBJAREA ({}) AND ({}))".format(
        subject, keywords[0]
    )  # Commmented out string: AND (build) AND (life) AND (cycle)
    # carry out the scopus search
    search_results = ScopusSearch(query=search_query, subscriber=True)
    search_results = pd.DataFrame(search_results.results)
    # keep count of the total number of papers
    paper_count += search_results.shape[0]
    print("search done!")
    # filter out the dataframe for the specific key topics that we want
    for i in np.arange(0, len(search_results)):
        # some abstracts are left as blank and pandas stupidly reads this is as a nonetype. Use try:except to get around this nightmare
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
papers.to_csv("Results/relevant_publications_all.csv", index=False)

# %%
