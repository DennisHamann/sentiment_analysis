# sentiment analysis #

Goal is to train a model which is able to extract emotions from text.


### Workflow

First checkout repository via
```shell
$ git clone https://github.com/DennisHamann/sentiment_analysis.git
```
**Never** commit directly into master branch!
For all changes we create a new branch
```shell
# create a topic branch
$ git checkout -b feature

# ( making changes ... )
$ git commit -m "done with feature"

git push -u origin HEAD
```

Then open https://github.com/DennisHamann/sentiment_analysis/branches and create a pull request in the UI. 
Do **not** merge the pull request. This is done by the owner of the repository. 


### Organization ###
For now we note our tasks in plannig/tasks.



### Technology ###
We use 
- python3.6 for data processing
- tensorflow/keras (versions tbd) for model training and evaluation
- tbd: dvc (https://dvc.org/) for versioning data set and further features, models


### Project setup ###

At first create a virtual environment (https://docs.python.org/3/library/venv.html).
Then install requirements via
```
pip3 install -r requirements.txt
```
and run tests
```python
# todo
```
### Google Drive ###

We use google drive for data storage
Account: sentiment-analysis@web.de
Passwort: iske2020
