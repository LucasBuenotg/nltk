import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

#baixar recursos adicionais do NLTK
nltk.download('movie_reviews')

#obter avaliação de filmes e dividir em conjuntos de treinamnto e teste
neg_reviews = [(movie_reviews.words(file_id), 'negative')for file_id in 
movie_reviews.fileids('neg')]
pos_reviews = [(movie_reviews.words(file_id), 'positive')for file_id in 
movie_reviews.fileids('pos')]
reviews = neg_reviews + pos_reviews
split_index = int(len(reviews) * 0.8)
train_set = reviews[:split_index]
test_set = reviews[split_index:]

#extrair caracteristicas dos textos ( frequenci de plavras)
def extract_features(words):
    return dict([(word, True) for word in words])

# preparar os dados de treinamento e teste
train_features = [(extract_features(words), category) for (words, category) in test_set]

#treinar o classificador naive bayes
classifier = NaiveBayesClassifier.train(train_features)

#não terminei o código


