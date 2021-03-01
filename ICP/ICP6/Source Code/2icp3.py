from bs4 import BeautifulSoup
import requests
import re

wiki_link = 'https://en.wikipedia.org/wiki/Deep_learning'
page_response = requests.get(wiki_link, timeout=5)

wiki_content = BeautifulSoup(page_response.content, "html.parser")

textContent = []
f = open('wiki_data.txt', 'w')
f.write( "title is" + wiki_content.title.string+ '\n')

for links in wiki_content.find_all('a',attrs={'href': re.compile("^http://")}):
    textContent.append(links.get('href'))
    f.write( repr(links.get('href')) + '\n')