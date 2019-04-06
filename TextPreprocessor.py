import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

# preprocessing the raw texual data
stop_words = set(stopwords.words('english'))

"""
A class that provide text preprocessing methods
Methods include: 
word Lemmatizer
stop Words Removal
lower case
invoke batchProcessing() to invoke all of the methods
"""

class tweet_preprocessor:

    def batchProcessingJSON(self, jsonData):
        rawContent = jsonData["text"]
        step1 = self.stopWordsRemoval(rawContent)
        step2 = self.lowercase(step1)
        step3 = self.wordLemmatizer(step2)
        return step3

    def batchProcessingString(self, str):
        stem = LancasterStemmer()
        return [stem.stem(w.lower()) for w in str]

    def wordLemmatizer(self, content):
        lemmatizer = WordNetLemmatizer()
        wordList = [lemmatizer.lemmatize(word) for word in content.split()]
        return " ".join(wordList)

    def stopWordsRemoval(self, content):
        filtered = [word for word in content.split() if word not in stopwords.words('english')]
        return " ".join(filtered)

    def lowercase(self, str):
        return str.lower()