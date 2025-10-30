# umap-ga-cluster-split
this repository is for splitting data to train and test using UMAP and then applying genetic algorithm to select best subset features.  

How do I use it? 

1- make sure you have all the dependencies 
pip install -r requirements.txt
warning: the current requrement.txt file contains some packages that are not necessary for this project.

2- open the cluster_splitting.ipynb file and run to get the X_train, X_test, y_train, y_test csv files, and
umap_clistering_train_test.png
csv files are preprocessed, please see the details in cluster_splitting.ipynb 
warning: in current version, if a cluster has less than 5 compounds, all those compounds will be in training set 

3- run ga_feature_selection_regression.py to run genetic algorithm feature selection. 
adjust GA parameters accordingy, and choose your regression model (default is Random Forest). 