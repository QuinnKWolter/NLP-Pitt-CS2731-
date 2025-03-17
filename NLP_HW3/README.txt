Quinn K Wolter
NLP - HW03
3/16/2025

Submission contents:
	README.txt
	report_qkw3_hw3.pdf
	ngram_skeleton.py
	qkw3_train_ngram.py
	qkw3_hw3_char_llm_skeleton.ipynb.txt
		<- NOTE: I had to shift away from Google Colab
		<- ALSO NOTE: I could not upload a notebook, so remove ".txt" to use.
	shakespeare_input.txt
	shakespeare_sonnets.txt
	nytimes_article.txt

How to run code:
	Unzip contents to and navigate to a directory that contains the various supporting project files, then run:
	```
	python ./qkw3_train_ngram.py
	```
	The Notebook can be opened and executed in the environment of your choice.

Computing environment:
	Windows 11
	Python 3.11
	Packages/imports: 
		torch
		transformers
		datasets

Resources: The course textbook, plenty of Stack Overflow, and an article on additive/Laplace smoothing. ChatGPT 4o helped parse some errors and troubleshoot CUDA functionality on my local machine, and was also used to bundle some recommended troubleshooting functionality into a cell towards the bottom of my Notebook.

Interpersonal collaboration: None this time.

Generative AI usage: Cursor IDE 4o-mini model used for a good bit of troubleshooting CUDA assertion errors and model generation troubles.

Problems: Plenty this time. The Google Colab would run out of memory on seemingly arbitrary commands, or would function fine for two dozen minutes before suddenly running out of memory. Restarting training several times in a row without changes, the estimated time would fluctuate between 30 minutes, 90 minutes, 37 hours... I eventually opted to move things to my personal machine and set up CUDA. I also caught the flu while needing to do this assignment, so my fevered brain was not at its best.
