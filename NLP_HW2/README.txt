Quinn K Wolter
NLP - HW02
2/25/2025

Submission contents:
	README.txt
	hw2_qkw3_cv.py
	hw2_qkw3_test.py
	qkw3_shared.py

How to run code:
	Unzip contents to and navigate to a directory that contains the various supporting project files.
	
	To train models and perform cross-validation (can run without args for default diplomacy_cv.csv):
	```
	python ./hw2_qkw3_cv.py [--filename diplomacy_cv.csv]
	```
	
	To generate predictions on test data (can run without args for default diplomacy_test.csv):
	```
	python ./hw2_qkw3_test.py [--filename diplomacy_test.csv]
	```

Computing environment:
	Windows 11
	Python 3.11
	Packages/imports: 
		pandas
		numpy
		torch
		torch.nn
		torch.optim
		nltk.corpus (stopwords)
		nltk.tokenize (word_tokenize)
		sklearn.feature_extraction.text (CountVectorizer, TfidfVectorizer)
		sklearn.model_selection (cross_val_score, cross_val_predict, StratifiedKFold)
		sklearn.linear_model (LogisticRegression)
		sklearn.metrics (accuracy_score, precision_recall_fscore_support, confusion_matrix)
		sklearn.feature_selection (SelectKBest, chi2)
		sklearn.pipeline (Pipeline)
		torch.utils.data (Dataset, DataLoader)
		joblib
		argparse
		re
		string
	To install all the packages at once:
	```
	pip install -r requirements.txt
	```
Additional Files: I moved some content to qkw3_shared.py to clean up the main scripts.

Resources: The course textbook and documentation for scikit-learn, PyTorch, and NLTK. StackOverflow was used to troubleshoot a few issues, and ChatGPT 4o to learn the definitions and intent of some terms like "Leaky ReLU" that I'd really never heard of before.

Interpersonal collaboration: My partner is a data scientist at Quest Diagnostics, and had lots to say about refining neural networks and the like. No code was contributed, but moreso just rambling about potential ideas to explore and look into.

Generative AI usage: Cursor IDE 4o model used for some repetitive tasks, like building out printing statements, troubleshooting a handful of PyTorch errors, handling MatPlotLib confusion matrices, etc.

No issues or problems. Thank you again for your time and effort!
Quinn K Wolter
