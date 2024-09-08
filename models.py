# models.py
import nltk
from nltk.corpus import stopwords

from sentiment_data import *
from utils import *
import math

import string

from collections import Counter
from collections import defaultdict
import numpy as np
import random

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        # nltk.download('stopwords')
        # self.stop_words = set(stopwords.words('english'))  # Load stopwords from nltk
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract unigram features from a sentence.

        :param sentence: List of words in the sentence.
        :param add_to_indexer: If True, add new words to the indexer. If False, don't add new words.
        :return: A Counter representing the sparse feature vector (unigram counts).
        """

        feature_vector = Counter()

        # Define a translation table to remove punctuation
        # translator = str.maketrans('', '', string.punctuation)

        # Iterate through each word in the sentence
        for word in sentence:
            # Convert to lowercase, strip whitespace, and remove punctuation
            word = word.lower().strip()

            # # Filter out stopwords and short words
            # if word in self.stop_words:
            #     continue  # Skip stopwords and short words

            #Check if we are allowed to add new features to the indexer
            if add_to_indexer:
                index = self.indexer.add_and_get_index(word, add=True)
            else:
                index = self.indexer.index_of(word)

            # If the word is in the indexer, update the feature vector
            if index !=-1:
                feature_vector[index] += 1
        
        return feature_vector

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer     
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract bigram features from a sentence.

        :param sentence: List of words in the sentence.
        :param add_to_indexer: If True, add new bigrams to the indexer. If False, don't add new bigrams.
        :return: A Counter representing the sparse feature vector (bigram counts).
        """
        feature_vector = Counter()    

        # Interate through each adjacent pair of words (bigrams) in the sentence
        for i in range(len(sentence) - 1):
            # Create the bigram
            bigram = (sentence[i].lower().strip(), sentence[i+1].lower().strip()) 

            # Create a string representation
            bigram_str = f"{bigram[0]}_{bigram[1]}"

            # Check if we are allowed to add new features to the indexer
            if add_to_indexer:
                index = self.indexer.add_and_get_index(bigram_str, add=True)
            else:
                index = self.indexer.index_of(bigram_str)

            if index != -1:
                feature_vector[index] += 1

        return feature_vector               
    
class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor using unigrams and bigrams with clipped word frequencies.
    """
    def __init__(self, indexer: Indexer, documents: List[List[str]], clip_threshold=3):
        self.indexer = indexer
        self.documents = documents
        self.clip_threshold = clip_threshold  # Maximum frequency of words or bigrams
    
    def get_indexer(self):
        return self.indexer
    
    def clean_word(self, word: str) -> str:
        """
        Clean and preprocess the word by lowercasing it and removing punctuation.
        :param word: Word to clean
        :return: Cleaned word
        """
        word = word.lower().strip().translate(str.maketrans('', '', string.punctuation))
        return word if word else None

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract unigrams and bigrams from the sentence with clipped word frequencies.
        :param sentence: List of words in the sentence.
        :param add_to_indexer: If True, add new features to the indexer. If False, don't add new features.
        :return: A Counter representing the sparse feature vector (clipped unigram and bigram counts).
        """
        feature_vector = Counter()
        term_count = defaultdict(int)
        total_terms = 0

        # Clean the sentence and get unigrams
        cleaned_sentence = [self.clean_word(word) for word in sentence if self.clean_word(word)]

        # Count unigrams and bigrams
        for i in range(len(cleaned_sentence)):
            unigram = cleaned_sentence[i]
            if unigram:
                term_count[unigram] += 1
                total_terms += 1
            
            if i < len(cleaned_sentence) - 1:
                bigram = f"{cleaned_sentence[i]}_{cleaned_sentence[i + 1]}"
                term_count[bigram] += 1
                total_terms += 1

        # Apply clipping to the term frequencies
        for term, count in term_count.items():
            tf_clipped = min(count, self.clip_threshold)  # Clip term frequency to the threshold

            # Add feature to the indexer if applicable
            if add_to_indexer:
                index = self.indexer.add_and_get_index(term, add=True)
            else:
                index = self.indexer.index_of(term)

            # If term is in the indexer, add its clipped frequency to the feature vector
            if index != -1:
                feature_vector[index] = tf_clipped

        return feature_vector

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, indexer: Indexer, feature_extractor: FeatureExtractor, learning_rate=1.0, num_epochs=100, schedule="decay"):
        self.indexer = indexer
        self.feature_extractor = feature_extractor
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.schedule = schedule  # Learning rate schedule (constant or decaying)

        self.weights = np.zeros(len(indexer))
        self.bias = 0
    
    def predict(self, sentence: List[str]) -> int:
        """
        Predict the class label for a given sentence.
        :param sentence: List of words in the sentence
        :return: Predicted label (0 or 1)
        """        

        # Extract features for the given sentence
        feature_vector = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
                                                                 
        # Compute weighted sum: z = w^T x + b                                                                 
        z = self.bias # Start with the bias term
        for index, value in feature_vector.items():
            z += self.weights[index] * value # Weighted sum of features

        # Return class prediction based on the sign of z
        return 1 if z >=0 else 0
    
    def train(self, train_exs: List[SentimentExample]):
        """
        Train the perceptron model on the training examples.
        :param train_exs: List of SentimentExample objects (training data)
        """
        # Iterate through multiple epochs
        for epoch in range(self.num_epochs):
            # Shuffle the training examples at the start of each epoch
            random.shuffle(train_exs)

            # Adjust the learning rate if using a decaying schedule
            if self.schedule == "decay":
                current_learning_rate = self.learning_rate / (1 + epoch)
            else:
                current_learning_rate = self.learning_rate  # Constant learning rate

            for ex in train_exs:
            # Extract features for the current training example
                feature_vector = self.feature_extractor.extract_features(ex.words, add_to_indexer=True)

                # Compute the prediction
                prediction = self.predict(ex.words)

                # If the prediction is incorrect, update weights and bias
                if prediction != ex.label:
                    # Update weight and bias based on the error
                    error = ex.label - prediction # Error is +1 for false negative, -1 for false positive
                    for index, value in feature_vector.items():
                        self.weights[index] += current_learning_rate * error * value
                        # Update bias
                        self.bias += current_learning_rate * error
                        
class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, indexer: Indexer, feature_extractor: FeatureExtractor, learning_rate=0.01, num_epochs=150):
        self.indexer = indexer
        self.feature_extractor = feature_extractor
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initialize weights and bias
        self.weights = np.zeros(len(indexer))
        self.bias = 0
    
    def sigmoid(self, z):
        """
        Compute the sigmoid of z.
        :param z: Weighted sum (w^T x + b)
        :return: Sigmoid of z
        """
        return 1 / (1+np.exp(-z))
    
    def predict_prob(self, feature_vector: Counter) -> float:
        """
        Predict the probability (sigmoid output) given a feature vector.
        :param feature_vector: Pre-extracted feature vector
        :return: Probability of the positive class (between 0 and 1)
        """
        # Compute weighted sum: z = w^T x + b
        z = self.bias
        for index, value in feature_vector.items():
            z += self.weights[index] * value
        
        # Return the probability from the sigmoid function
        return self.sigmoid(z)
    
    def predict(self, sentence: List[str]) -> int:
        """
        Predict the class label for a given sentence by extracting its features first.
        :param sentence: List of words in the sentence
        :return: Predicted label (0 or 1)
        """
        feature_vector = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        return 1 if self.predict_prob(feature_vector) >= 0.5 else 0
    
    def train_pre_extracted(self, pre_extracted_features: List[tuple[Counter, int]]):
        """
        Train the logistic regression model using pre-extracted features.
        :param pre_extracted_features: List of tuples containing (feature_vector, label)
        """
        for epoch in range(self.num_epochs):
            total_loss = 0
            random.shuffle(pre_extracted_features)

            for feature_vector, label in pre_extracted_features:
                # Compute the probability for the current example
                prediction_proba = self.predict_prob(feature_vector)

                # Compute the error
                error = label - prediction_proba
                
                # Update weights and bias using gradient descent
                for index, value in feature_vector.items():
                    self.weights[index] += self.learning_rate * error * value
                self.bias += self.learning_rate * error

                # Accumulate loss (binary cross-entropy)
                total_loss += -(label * np.log(prediction_proba) + (1 - label) * np.log(1 - prediction_proba))
    
def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # Ensure the Indexer is populated by extracting features for all training examples
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    # Initialize indexer and perceptron classifier
    indexer = feat_extractor.get_indexer()

    # Create a PerceptronClassifier instance
    perceptron = PerceptronClassifier(indexer, feat_extractor)

    perceptron.train(train_exs)

    return perceptron


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model using pre-extracted features.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Pre-extract features for all training examples
    pre_extracted_features = []
    for ex in train_exs:
        # Extract features and store them in a list
        feature_vector = feat_extractor.extract_features(ex.words, add_to_indexer=True)
        pre_extracted_features.append((feature_vector, ex.label))

    # Initialize indexer and logistic regression classifier
    indexer = feat_extractor.get_indexer()
    logistic_regression = LogisticRegressionClassifier(indexer, feat_extractor)

    # Train using the pre-extracted features
    logistic_regression.train_pre_extracted(pre_extracted_features)

    return logistic_regression

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        train_docs = [ex.words for ex in train_exs]
        feat_extractor = BetterFeatureExtractor(Indexer(), train_docs)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model