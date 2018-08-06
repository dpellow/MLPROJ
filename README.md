# MLPROJ

Terminology:

	* "dataset directory" - directory where datasets files are located (e.g. expression profiles, phenotype profiles etc.)

	* "project directory" - directory where project files are located (e.g. cloned files and folders from git project)

Directories Hierarchy:

	* env - initiation, maintenance and configuration for project and datasets environments

		* config - configuration files

			* conf.json - main config file. 
				BASE_PROFILE - make sure this field directs to datasets' base directory
		* init - creation of dataset and python environment 
			* init.sh - creation of python env
			* dependencies.txt - python libraries on which the project depends 
			* creates datasets environment, creates required folders and downloads datasets
		* omics - maintenance of dataset environment
			* sync_omics.py - utility folder for syncing between project and dataset folders 
			* list - list dataset folder last saved state
			* dictionaries- list dataset folder last saved state
			* NOTE: both  list and dictionaries directories are synced from and to BASE_PROFILE directory
		* clean.sh - cleans project folder's *.pyc files
	* filter - json files that represents filter expression on top phenotype data
	* groups - json files that represents group expression on top phenotype data		
	* tcga - fetch expression profile data from dataset directory according given filter & group expression (if any are given)
	* go - fetch GO data by their proper hierarchy
	* utils - general python helper functions
	* constant.py - project's constant fields
	* main.py - project's driver
	
In order to run the project on your machine execute the following steps:
	(1) clone project to your machine (referred "project directory" from now on)
	(2) change "BASE_PROFILE" field in conf.json file to the directories where your want to place your datasets (referred "dataset directory" from now on)
	(3) execute init.sh to create your python venv. alternatively, create one manually and install dependencies with "pip install -r dependencies.txt"
	(4) execute "python create_env.py"	
	(5) you good to go. execute main.py and whatever...:)

	  
