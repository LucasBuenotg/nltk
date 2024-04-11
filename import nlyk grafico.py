import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter 
import string
import matplotlib.pyplot as plt


# baixar recursos adicionais do NLTK
nltk.download('punkt')
nltk.download('stopwords')

#texto de exemplo
texto = """I must not fear. Fear is the mind-killer. Fear is the little-death that brings total obliteration. I will face my fear. I will permit it to pass over me and through me. And when it has gone past I will turn the inner eye to see its path. Where the fear has gone there will be nothing. Only I will remain."""

#tokenização 
tokens = word_tokenize(texto.lower(), language='portuguese')

#remoção de pontuação e stopwords
pontuacao = set(string.punctuation)
stop_words = set(stopwords.words('portuguese'))
tokens_filtrados = [word for word in tokens if word not in pontuacao and word not in stop_words]

#contagem de frequencia 
frequencia = Counter(tokens_filtrados)

#exibição de resultados
print("frequencia das palavras:")
for palavra, freq in frequencia.items():
    print(f"{palavra}:{freq}")

# plotagem do gráfico de barras
    plt.figure(figsize=(12,6))
    plt.bar(frequencia.keys(), frequencia.values())
    plt.xticks(rotation=45)
    plt.xlabel('palavra')
    plt.ylabel('frequencia')
    plt.title('frequencia das palavras no texto')
    plt.show()

    #plotagem do grafico circular
    plt.figure(figsize=(8,8))
    plt.pie(frequencia.values(), labels=frequencia.keys(), autopct='%1.1f%%')
    plt.title('Distribuição das palavras do texto')
    plt.show()