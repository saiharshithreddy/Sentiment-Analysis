# Sentiment-Analysis
## Goal
The project is aimed at solving the problem of finding the sentiment of the user reviews on yelp.com

## Installation
Libraries need to run the code: ```pandas```, ```numpy```, ```pytorch``` libraries 
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
2. Stemming/Lemmetization
3. Uppercase to Lowercase
4. Removing any bad characters like \n, \t, $ etc using regex.  
Check my [blog] on text preprocessing.

## Evaluation
Due to an imbalance classes, F1 score was metric was used.




