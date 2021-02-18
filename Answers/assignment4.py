import matplotlib.pyplot as plt
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.probability import FreqDist
import pandas as pd
import nltk

inp = input("Enter Paragraph : ")

txt = ''.join([c for c in inp if not c.isdigit()])

para_fdist = FreqDist()

counter = 1

for sent in sent_tokenize(txt):
    fig = plt.figure(counter,figsize = (16,8))
    ax = fig.gca()
    ax.set_xlabel('WORD')
    ax.set_ylabel('FREQUENCY OF WORD')
    sent_fdist = FreqDist()
    for word in word_tokenize(sent):
        if word not in punctuation and word not in stopwords.words('english'):
            sent_fdist[word] += 1
            para_fdist[word] += 1
    df = pd.DataFrame.from_dict(sent_fdist,columns = ['count'],orient = 'index')
    df = df.sort_values('count',ascending = False)
    df['count'][:].plot(kind = 'bar',ax = ax)
    ax.set_title(f"Frequency of words in sentence {counter}")
    plt.show()
    plt.close()
    counter += 1

fig = plt.figure(counter,figsize = (16,8))
ax = fig.gca()
ax.set_xlabel('WORD')
ax.set_ylabel('FREQUENCY OF WORD')
df = pd.DataFrame.from_dict(para_fdist,columns = ['count'],orient = 'index')
df = df.sort_values('count',ascending = False)
df['count'][:].plot(kind = 'bar',ax = ax)
ax.set_title("Frequency of words in Paragraph")
plt.show()

