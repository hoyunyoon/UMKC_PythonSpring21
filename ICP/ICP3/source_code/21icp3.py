import requests###library
from bs4 import BeautifulSoup as BS##Library
URL = 'https://en.wikipedia.org/wiki/Deep_learning'##URL
page = requests.get(URL)
soup = BS(page.content, 'html.parser')##parsing web page
print(soup.title.string)
rows = soup.find_all('a')
print(rows)
my_data_file = open('wikidata.txt', 'w')
for link in rows:
    filtered_data = link.get('href')
    print(filtered_data)
    my_data_file.write(str(filtered_data))
    my_data_file.write("\n")
my_data_file.close()