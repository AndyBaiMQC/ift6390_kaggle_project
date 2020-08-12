# ift6390_kaggle_project
IFT 6390 Kaggle Competition Project

Team: Team Default
Team Member: Yifan Bai, Maxime Daigle

Final ranking: 15/103

1. File list

	utils.py
	run_nb_lr.py
	run_bilstm_bigru.py


2. Preprequisites

The packages need to be installed includes:
	
	numpy==1.16.0
	pandas==0.23.4
	sklearn==0.20.1
	scipy==1.1.0
	keras==2.2.4
	nltk==3.2.2

There are several pretrained word embeddings needed, they are downloaded from: 
	
	https://www.kaggle.com/c/quora-insincere-questions-classification/data


3. Running Instruction

utils.py contains utility functions, which is about text preprocessing.

run_nb_lr.py is the script used to train algorithm one, to run it, use
	
	python run_nb_lr.py

run_bilstm_bigru.py is the script used to train algorithm two, to run it, use

	python run_bilstm_bigru.py
