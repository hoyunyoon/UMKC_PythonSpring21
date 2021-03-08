# importing Natural Language Toolkit and all the required libraries
import nltk
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk, ngrams
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

# fetching the data from input.txt file
input = open('input.txt', encoding="utf8").read()

# Tokenization,POS,Stemming,Lemmatization,Trigram,NamedEntityRecognition

#Tokenization
print("\n###Tokenization### \n")

# tokenizing the input data into Sentences and Words
st_tokens = nltk.sent_tokenize(input)  # retrieve Sentence Tokens
wd_tokens = nltk.word_tokenize(input)  # retrieve Words Tokens
print("No.of Sentences: ", len(st_tokens))
print("No.of Words: ", len(wd_tokens))

###POS
print("\n### POS ### \n")

# performing Parts of Speech tagging by using pos_tag

print(nltk.pos_tag(wd_tokens))

# Stemming
print("\n###Stemming### \n")

    # PorterStemmer
pStemmer = PorterStemmer()
print("\nPorter Stemmer output: \n")
for i in st_tokens:
    print(pStemmer.stem(i), end='')

    # LancasterStemmer
lStemmer = LancasterStemmer()
print("\nLancaster Stemmer output : \n")
for i in st_tokens:
    print(lStemmer.stem(i), end='')

    # SnowballStemmer
sStemmer = SnowballStemmer('english')
print("\nSnowball Stemmer output : \n")
for i in st_tokens:
    print(sStemmer.stem(i), end='')

# Lemmatization
print("\n###========== Lemmatization ==========### \n")

lemmatizer = WordNetLemmatizer()# Applying Lemmatization using WordNetLemmatizer
print("\nLemmatization output : \n")
for i in st_tokens:
    print(lemmatizer.lemmatize(i), end=' ')

# Trigram
print("\n### Trigram ### \n")

token = nltk.word_tokenize(input)
n = 0
for s in st_tokens:
    n = n + 1
    if n < 2:
        token = nltk.word_tokenize(s)
        bigrams = list(ngrams(token, 2))
        trigrams = list(ngrams(token, 3))
        print("The text:", s, "\nword_tokenize:", token, "\nbigrams:", bigrams, "\ntrigrams", trigrams)

# Named Entity Recognition(NER)
print("\n###Named Entity Recognition ### \n")

n = 0
for s in st_tokens:
    n = n + 1
    if n < 2:
        print(ne_chunk(pos_tag(nltk.word_tokenize(s))))
