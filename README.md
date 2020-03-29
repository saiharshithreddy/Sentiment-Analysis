# Sentiment-Analysis
## Goal
The project is aimed at solving the problem of finding the sentiment of the user reviews on yelp.com

## Installation
Libraries needed to run the code:
```python
pip3 install -r requriements.txt
```
[Download](https://www.yelp.com/dataset) the Yelp dataset:  

## Algorithms
1. Support Vector Machine
2. Naive Bayes
3. RCNN model

## Steps involved
**Data preprocessing**
1. Remove stop words
2. Lemmetization
3. Uppercase to Lowercase
4. Removing any bad characters like \n, \t, $ etc using regex.  

Check my [blog](https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8) on text preprocessing.

## Evaluation
Due to an imbalance classes, F1 score was metric was used.




