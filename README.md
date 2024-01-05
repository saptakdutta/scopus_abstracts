# scopus_abstracts

This repository allows the user to mine relevant abstracts from the Elsevier Scopus database through an API, and then extract topics contained within the abstracts with the Latent Dirichlet Allocation text mining algorithm.

# Getting started 
In order to set up this project you will need the repository, and a virtual environment that contains all the required software dependencies.


## Installing GIT
Start by installing `GIT` on your system, which will allow us to clone the repository:
### Linux
Using apt (debian based): 
> sudo apt install git-all

Using dnf (RHEL based):

> sudo dnf install git-all

### MacOS
Use the homebrew package manager
> brew install git

### Windows
> Follow [this tutorial](https://git-scm.com/download/win) to set it up locally

Once git is installed, `cd` into the directory that you want this project to be located in and then clone this repository like so:

> git clone https://gitlabc.nrc-cnrc.gc.ca/Saptak.Dutta/scopus_abstract_mining/

You'll be prompted to enter in your gitlab username and password to clone the repo locally.
Now go ahead to the next part to set up the virtual environment

## Setting up the virtual environment
### Recommended venv setup [`conda-lock.yml` method]
The easiest and most consistent way to set up the project is through the provided `conda-lock` file. Install conda/miniconda using the [following instructions.](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
This method will set up an exact replica of the development virtual environment on the user's end regardless of operating system

Ensure that your conda install is upto date using:

> conda update conda

Now install the conda-lock package to your `base environment` like so:

> conda activate base

> conda install --channel=conda-forge --name=base conda-lock

Once conda lock has been installed,  use it to create the venv:

> conda-lock install -n scopus_abstract_mining conda-lock.yml

This should create a virtual environment called `scopus_abstract_mining` that will contain all the packages required to run this tool. The conda-lock package contains a concrete list of package versions (with checksums) to be installed. Unlike `conda env create`, the resulting environment will not change as new package versions become available, unless we explicitly update the lock file. This is a much more reliable way to create a virtual environment across multiple systems and platforms

### Alternate venv setup [`environment.yml` method] (not recommended for most users)
If you want to make changes to the repo itself and tinker around with the tool, using the environment file to create an up-to-date environment may be the better option.
Ensure that your conda install is upto date using:

> conda update conda

Use your python package manager (conda/miniconda/mamba) to cd into the root directory and run the following command:

> conda env create -f environment.yml


This should create a virtual environment called `scopus_abstract_mining` that will contain all the packages required to run this tool. I cannot guarantee that the environment will be resolved without conflicts (especially between the pip and conda packages). Some packages such as gensim and numba have been observed to create problems in the past. There may be a bit of tinkering with packages and versioning in the YML file that needs to be done to set the venv up correctly.

# Setting up the Scopus API
In order to use the scopus API, you will need an API key. If your organization is a registered Scopus institution, they will provide you with one for making requests. Go to https://dev.elsevier.com/, and request an API key. You will have to enter in your NRC credentials, which they will then verify by sending you an email. An example API key has been included with this repository

# Setting up the python API key endpoint
The first time `abstract_retreival.py` is run, a request for the API key should be presented to the user. If it isn't please run the following commands to verify the status of your API key locally.

To see what your current API key config is run:

> from pybliometrics.scopus.utils import config
>
> \#### Show keys ####\
> print(config['Authentication']['APIKey'])

If you do not see an API key registered in the metadata, you may have to manually edit the CONFIG file. To locate it run the following commands: 

> import pybliometrics
>
> \#### Show location of the config file ####\
> pybliometrics.scopus.utils.constants.CONFIG_FILE

When you locate the file, there will be a [Authentication] section which you will need to fill out with the API key like so:

> [Authentication]\
> APIKey = xxxxxxxxxxx

# Using the included batch file
The batch file allows for the abstract retreival program to be run without using a code editor. In order to use it, you must open it up and edit the second line which points to the conda/mamba distribution like so:

> C:\Users\\**username_goes_here**\AppData\Local\miniconda3\condabin\activate

Then once set up, you can simply click the batch file to open up the program.

## Pointing the batch file to the abstract retreival tool you want
By default, the batch file points to the university search abstract retreival tool. There is a second version 'abstract_retreival_search.py', which does not use university names. This makes the search much slower (~ 8-10 minutes per subject area) because the results that are returned are several orders of magnitude higher than the university filter. 

In order to point the batch file towards this alternate tool, simply right click and edit it to change line 6 to the following:

> python abstract_retreival_search.py

You should now have the version that allows you to search for ALL publications regardless of institution

# Possible API Calls
A variety of API calls can be made to the SCOPUS database. You can choose to search by author, institution or topic. For a full list of the API call objects available, please view the documentation at:

> https://pybliometrics.readthedocs.io/en/stable/classes.html

This example script is set up with an institutional search example

## Subject areas
Scopus contains 26 `subject areas` that you can use to narrow ddown your search for journals. They are: 

|  Subject Area| Search key  |
|---|---|
| Agriculture and biological sciences  | agri  |
| Arts and humanities | arts  |
| Biochemistry, genetics and molecular biology | bioc  |
| Business, management and accounting | busi  |
| Chemical engineering  | ceng  |
| Chemistry  | chem  |
| Computer science  | comp  |
| Decision sciences  | deci  |
| Earth and planetary sciences  | eart  |
| Economics, econometrics and finance  | econ  |
| Energy  | ener  |
| Engineering  | engi  |
| Environmental science  | envi  |
| Immunology and microbiology  | immu  |
| Materials science  | mate  |
| Mathematics  | math  |
| Medicine  | medi  |
| Neuroscience  | neur  |
| Nursing  | nurs  |
| Pharmacology, toxicology and pharmaceutics  | phar  |
| Physics and astronomy  | phys  |
| Psychology  | pysc  |
| Social sciences  | soci  |
| Veterinary  | vete  |
| Dentistry  | dent  |
| Health Professions  | heal  |

This table should be referenced when using your scopus search.

## Using the subject_areas.csv and keywords.csv files

In order to make the tool easier to operate, both the `subject areas` and `key topics` have been separated into two
separate csv files. For the subject areas file, simply change the 'Use' column to 'Y' when you want to include one of the 26 existing subject areas in your search. For the key words file, use upto 12 key words, and the program will read it in for use.

**Important note**
The first keyword is used as the initial search filter. Therefore, the first term is the most important and te subsequent terms will be papers related to the first term


# Removing the virtual environment
If you want to remove the project from your local computer, run the following commands to remove the created virtual environments:

> conda remove -n scopus_abstract_mining --all

This applies to both methods of virtual environment creation (i.e., `conda-lock` and `environment.yml`)


# Todo list
- [x] Add in a batch file interface so users can one click operate the tool
- [x] Separate out the subject areas and search terms into csvs so that batch 'one click' use is more friendly to the average user
- [x] Add in a way to search without universities and university ID's
- [ ] See if searching by country is something that can be done
- [ ] Continue improving the existing documentation
- [ ] Improve the topic modeling side of the tool for better user insights
