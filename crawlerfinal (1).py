############################### FROM HERE #####################################
import requests
import requests
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
import nltk
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import sqlite3
import requests
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
import sqlite3
import nltk
import re
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
google_urls=[]

def get_top_urls(topic, category=None, num_urls=5):
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": topic,
        "srprop": "title",
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    urls = []
    for result in data.get('query', {}).get('search', []):
        title = result['title']
        article_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        if not category or category.lower() in title.lower():
            urls.append(article_url)

    return urls[:num_urls]

# Global variables to store the topic and top URLs
topic = ""
top_urls = []

def extract_summary_from_wikipedia(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    summary = ' '.join(p.get_text() for p in paragraphs[:3])  # Extract the first 3 paragraphs as summary
    return summary

def get_top_google_urls(query, num_urls=5):
    base_url = "https://www.google.com/search"
    params = {
        "q": query,
        "num": num_urls
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        search_results = soup.find_all('div', class_='tF2Cxc')
        urls = [result.a['href'] for result in search_results if result.a]
        return urls
    except requests.RequestException as e:
        print("Error fetching data from Google Search:", e)
        return []
def extract_keywords_from_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Perform part-of-speech tagging
    pos_tags = nltk.pos_tag(tokens)

    # Perform named entity recognition (NER)
    named_entities = nltk.ne_chunk(pos_tags)

    # Extract named entities of type 'NE' with less than or equal to 6 words
    named_entities_list = []
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            entity = ' '.join(c[0] for c in chunk)
            if len(entity.split()) <= 6:  # Discard entities with more than 6 words
                named_entities_list.append(entity)

    return named_entities_list




# Function to extract content from URLs and perform keyword extraction
def extract_keywords_from_urls(urls):
    all_keywords = []
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join([element.get_text() for element in soup.find_all(text=True)])
            keywords = extract_keywords_from_text(text)
            all_keywords.extend(keywords)
        except Exception as e:
            print(f"Error processing URL '{url}':", e)

    # Remove duplicates while preserving order
    unique_keywords = list(set(all_keywords))
    return unique_keywords
def main():
    global topic, top_urls
    topic = input("Enter a topic: ")
    category = input("Enter a category (optional): ")
    top_urls = get_top_urls(topic, category)

    print(f"Top {len(top_urls)} URLs related to '{topic}' ({category if category else 'any category'}):")
    for i, url in enumerate(top_urls, start=1):
        print(f"{i}. {url}")
def main1(topic, wikipedia_url, google_urls):
    print("Extracting summary from the Wikipedia article...")
    wikipedia_summary = extract_summary_from_wikipedia(wikipedia_url)

    print("Generating Google search query...")
    google_query = f"{topic} {wikipedia_summary}"

    print("Fetching additional URLs from Google...")
    new_google_urls = get_top_google_urls(google_query)
    google_urls.extend(new_google_urls)

    print(f"Top {len(new_google_urls)} URLs related to the Wikipedia article:")
    for i, url in enumerate(new_google_urls, start=1):
        print(f"{i}. {url}")
def main2():
    urls = google_urls



# Extract keywords from the provided URLs
    unique_keywords = extract_keywords_from_urls(urls)

    # Print or save unique keywords
    print("Unique Keywords:", unique_keywords)

    # Write unique keywords to a file
    output_file = "unique_keywords1.txt"
    with open(output_file, 'w') as f:
        for keyword in unique_keywords:
            f.write(keyword + '\n')

    print(f"Unique keywords saved to {output_file}")
if __name__ == "__main__":
    main()
    topic = "Enter your topic here"
    wikipedia_urls = top_urls  # Assuming top_urls is a list of URLs
    for url in wikipedia_urls:
        main1(topic, url, google_urls)
    print(google_urls)
    urls = google_urls



# Extract keywords from the provided URLs
unique_keywords = extract_keywords_from_urls(urls)

# Print or save unique keywords
print("Unique Keywords:", unique_keywords)

# Write unique keywords to a file
# Write unique keywords to a file with UTF-8 encoding
output_file = "unique_keywords3.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    for keyword in unique_keywords:
        f.write(keyword + '\n')

print(f"Unique keywords saved to {output_file}")

try:
    with open("unique_keywords3.txt", "r", encoding="utf-8") as file:
        keywords = file.readlines()
except UnicodeDecodeError as e:
    print("Unicode decoding error:", e)
    # Handle the error appropriately, such as specifying a different encoding or logging the error
except FileNotFoundError as e:
    print("File not found:", e)
    # Handle the file not found error
except Exception as e:
    print("An error occurred:", e)
    # Handle other types of exceptions

# Load keywords from file


# Preprocess keywords: Remove newline characters
keywords = [keyword.strip() for keyword in keywords]

# Vectorize keywords using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(keywords)

# Cluster keywords using hierarchical clustering
num_clusters = 30  # You can adjust this based on your preference
agg_cluster = AgglomerativeClustering(n_clusters=num_clusters)
agg_cluster.fit(X.toarray())

# Create a dictionary to map clusters to keywords
cluster_keywords = {}
for i, label in enumerate(agg_cluster.labels_):
    cluster_keywords.setdefault(label, []).append(keywords[i])

# Print clusters and their keywords
for cluster, keywords in cluster_keywords.items():
    print(f"Cluster {cluster + 1}:")
    print(", ".join(keywords))
    print()

# Function to find the most significant words in each cluster
def get_cluster_names(vectorizer, cluster_keywords):
    feature_names = vectorizer.get_feature_names_out()
    cluster_names = {}

    for cluster, keywords in cluster_keywords.items():
        # Vectorize keywords in the cluster
        cluster_X = vectorizer.transform(keywords)
        # Sum TF-IDF scores for each term in the cluster
        tfidf_sum = cluster_X.sum(axis=0).A1  # Convert to 1D array
        # Get indices of the highest TF-IDF scores
        sorted_indices = np.argsort(tfidf_sum)[::-1]
        # Select top N words to represent the cluster
        top_n_words = [feature_names[index] for index in sorted_indices[:3]]
        cluster_names[cluster] = " ".join(top_n_words)

    return cluster_names

# Get cluster names
cluster_names = get_cluster_names(vectorizer, cluster_keywords)

# Print clusters with their names and keywords
conn = sqlite3.connect('clusters.db')
cursor = conn.cursor()

# Create tables for each cluster with their names
# Create tables for each cluster with their names
for cluster, keywords in cluster_keywords.items():
    cluster_name = cluster_names[cluster].replace(' ', '_').replace(':', '_')  # Replace invalid characters
    table_name = f"Cluster_{cluster_name}"
    
    # Create table with keywords and cluster name
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (Keyword TEXT)")
    
    # Insert keywords into the table
    for keyword in keywords:
        cursor.execute(f"INSERT INTO {table_name} (Keyword) VALUES (?)", (keyword,))

# Commit changes and close connection
conn.commit()
conn.close()

print("Clusters and keywords saved to SQLite database.")
def get_top_urls(topic, category=None, num_urls=2):
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": topic,
        "srprop": "title",
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    urls = []
    for result in data.get('query', {}).get('search', []):
        title = result['title']
        article_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        if not category or category.lower() in title.lower():
            urls.append(article_url)

    return urls[:num_urls]

# Connect to SQLite database
conn_clusters = sqlite3.connect('clusters.db')
cursor_clusters = conn_clusters.cursor()

# Connect to SQLite database for storing keyword URLs
conn_keyword_urls = sqlite3.connect('keyword_urls4.db')
cursor_keyword_urls = conn_keyword_urls.cursor()
def print_table_names():
    cursor_clusters.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor_clusters.fetchall()
    print("Available tables:")
    for table in tables:
        print(table[0])

# Call the function to print all table names
print_table_names()

# Prompt the user to select a table
selected_table = input("Please enter the name of the table to use: ")
# Retrieve keywords from selected cluster tables
selected_cluster_tables = [selected_table]  # Modify this with actual cluster tables
for table_name in selected_cluster_tables:
    cursor_clusters.execute(f"SELECT Keyword FROM {table_name}")
    keywords = cursor_clusters.fetchall()

    # Create table with cluster name in keyword URLs database
    cursor_keyword_urls.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (Keyword TEXT, URLs TEXT)")

    # Process each keyword and retrieve top URLs
    for keyword_tuple in keywords:
        keyword = keyword_tuple[0]  # Extract keyword from tuple
        top_urls = get_top_urls(keyword)

        # Store retrieved URLs in keyword URLs database
        url_string = "\n".join(top_urls)  # Convert list of URLs to a string with newline separators
        cursor_keyword_urls.execute(f"INSERT INTO {table_name} (Keyword, URLs) VALUES (?, ?)", (keyword, url_string))

# Commit changes and close connections
conn_keyword_urls.commit()
conn_keyword_urls.close()
conn_clusters.close()

print("Top URLs retrieved and stored in the SQLite database.")
def extract_summary_from_wikipedia(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    summary = ' '.join(p.get_text() for p in paragraphs[:3])  # Extract the first 3 paragraphs as summary
    return summary

def get_top_google_urls(query, num_urls=2):
    base_url = "https://www.google.com/search"
    params = {
        "q": query,
        "num": num_urls
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        search_results = soup.find_all('div', class_='tF2Cxc')
        urls = [result.a['href'] for result in search_results if result.a]
        return urls
    except requests.RequestException as e:
        print("Error fetching data from Google Search:", e)
        return []

def run(topic, conn):
    cursor = conn.cursor()

    # Retrieve the table name
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_name = tables[0][0]
    print(f"Using table: {table_name}")

    # Ensure GoogleURLs column exists
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    if 'GoogleURLs' not in column_names:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN GoogleURLs TEXT")
        print(f"Added 'GoogleURLs' column to {table_name} table.")

    # Retrieve Wikipedia URLs from the table
    cursor.execute(f"SELECT URLs FROM {table_name}")
    wikipedia_urls = cursor.fetchall()

    # Process each Wikipedia URL
    for url_tuple in wikipedia_urls:
        url = url_tuple[0]
        print("Extracting summary from the Wikipedia article:", url)
        wikipedia_summary = extract_summary_from_wikipedia(url)

        print("Generating Google search query...")
        google_query = f"{topic} {wikipedia_summary}"

        print("Fetching additional URLs from Google...")
        google_urls = get_top_google_urls(google_query)

        # Store Google URLs in the database
        cursor.execute(f"UPDATE {table_name} SET GoogleURLs=? WHERE URLs=?", 
                                    ("\n".join(google_urls), url))

        print(f"Top {len(google_urls)} URLs related to the Wikipedia article:")
        for i, google_url in enumerate(google_urls, start=1):
            print(f"{i}. {google_url}")

    # Commit changes
    conn.commit()


    # Connect to SQLite database
conn = sqlite3.connect('keyword_urls4.db')

    # Provide the topic
topic = "Enter your topic here"

    # Process Wikipedia URLs associated with keywords
run(topic, conn)

    # Close connection
conn.close()
print("Google URLs updated in the SQLite database.")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load SpaCy model for NER
nlp = spacy.load('en_core_web_sm')
sentiment_analyzer = SentimentIntensityAnalyzer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Extract content from URL
def extract_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text() for p in paragraphs)
        return preprocess_text(content), soup
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return "", None

# Keyword density calculation
def keyword_density(content, keyword):
    tokens = word_tokenize(content)
    keyword_tokens = word_tokenize(keyword.lower())
    count = sum(1 for token in tokens if token in keyword_tokens)
    return count / len(tokens)

# Named Entity Recognition (NER)
def named_entity_recognition(content, keywords):
    doc = nlp(content)
    entities = [ent.text.lower() for ent in doc.ents]
    keyword_entities = [word.lower() for word in keywords.split()]
    common_entities = set(entities).intersection(set(keyword_entities))
    return len(common_entities) / len(keyword_entities) if keyword_entities else 0

# Sentiment Analysis
def sentiment_analysis(content):
    sentiment = sentiment_analyzer.polarity_scores(content)
    return sentiment['compound']

# Heading Tags Analysis
def heading_tags_analysis(soup, keyword):
    headings = soup.find_all(['h1', 'h2', 'h3'])
    keyword_count = sum(keyword.lower() in heading.get_text().lower() for heading in headings)
    return keyword_count / len(headings) if headings else 0

# Multimedia Content Detection
def multimedia_content_detection(soup):
    multimedia_count = len(soup.find_all(['img', 'video', 'audio']))
    return 1 if multimedia_count > 0 else 0

# Authority on Subject
def authority_on_subject(url):
    authoritative_domains = ["wikipedia.org", "imdb.com", "indianexpress.com", "onmanorama.com"]  # Example authoritative domains
    return 1 if any(domain in url for domain in authoritative_domains) else 0

# Recency of Content
def recency_of_content(soup):
    date_tags = soup.find_all(['time', 'span', 'p'], {'class': ['date', 'time', 'published']})
    for date_tag in date_tags:
        try:
            date_text = date_tag.get_text()
            publication_date = datetime.strptime(date_text, '%B %d, %Y')  # Adjust date format as needed
            days_since_publication = (datetime.now() - publication_date).days
            return 1 / (1 + days_since_publication)
        except ValueError:
            continue
    return 0

# Citations and References
def citations_and_references(soup):
    references = len(soup.find_all('a', href=True))
    return references

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Connect to SQLite database
conn = sqlite3.connect('keyword_urls4.db')
cursor = conn.cursor()

# Drop the table if it exists and create a new one with the correct schema
cursor.execute("DROP TABLE IF EXISTS Ranked_URLs")
cursor.execute("""
    CREATE TABLE Ranked_URLs (
        Keyword TEXT,
        Wiki_URL TEXT,
        Google_URL TEXT,
        Combined_Score REAL
    )
""")

# Fetch all rows from the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
table_name = tables[0][0]
print(f"Using table: {table_name}")
cursor.execute(f"SELECT Keyword, URLs, GoogleURLs FROM {table_name}")
rows = cursor.fetchall()

data = []

for row in rows:
    keyword = row[0]
    wiki_url = row[1]
    google_urls = row[2].split('\n')

    # Extract and preprocess Wikipedia content
    wiki_content, _ = extract_content(wiki_url)
    wiki_embedding = get_bert_embedding(wiki_content)
    
    for url in google_urls:
        content, soup = extract_content(url)
        if not content or not soup:
            continue

        embedding = get_bert_embedding(content)
        
        # Calculate Cosine similarity
        similarity_score = cosine_similarity(wiki_embedding, embedding).flatten()[0]
        
        # Calculate keyword density
        density = keyword_density(content, keyword)
        
        # Calculate content length
        length = len(content.split())
        
        # Calculate readability score
        readability = textstat.flesch_reading_ease(content)
        
        # Named Entity Recognition
        ner_score = named_entity_recognition(content, keyword)
        
        # Sentiment Analysis
        sentiment_score = sentiment_analysis(content)
        
        # Heading Tags Analysis
        heading_score = heading_tags_analysis(soup, keyword)
        
        # Multimedia Content Detection
        multimedia_score = multimedia_content_detection(soup)
        
        # Authority on Subject
        authority_score = authority_on_subject(url)
        
        # Recency of Content
        recency_score = recency_of_content(soup)
        
        # Citations and References
        references_count = citations_and_references(soup)
        
        # Combine the scores (weights can be adjusted)
        combined_score = (
            (similarity_score * 0.3) +
            (density * 0.2) +
            (length * 0.1) +
            (readability * 0.1) +
            (ner_score * 0.1) +
            (sentiment_score * 0.05) +
            (heading_score * 0.05) +
            (multimedia_score * 0.05) +
            (authority_score * 0.05) +
            (recency_score * 0.05) +
            (references_count * 0.05)
        )
        
        data.append((keyword, wiki_url, url, combined_score))

# Insert data into the new table
cursor.executemany("INSERT INTO Ranked_URLs (Keyword, Wiki_URL, Google_URL, Combined_Score) VALUES (?, ?, ?, ?)", data)

# Commit changes and close the connection
conn.commit()
conn.close()

print("URLs ranked and stored in the database.")
conn = sqlite3.connect('keyword_urls4.db')
cursor = conn.cursor()

# Retrieve top 10 URLs based on combined scores
cursor.execute("""
    SELECT Keyword, Wiki_URL, Google_URL, Combined_Score
    FROM Ranked_URLs
    ORDER BY Combined_Score DESC
    LIMIT 10
""")
top_urls = cursor.fetchall()

# Display the top URLs
print("Top 10 URLs based on ranking:")
for rank, (keyword, wiki_url, google_url, combined_score) in enumerate(top_urls, start=1):
    print(f"Rank {rank}:")
    print(f"Keyword: {keyword}")
    print(f"Wikipedia URL: {wiki_url}")
    print(f"Google URL: {google_url}")
    print(f"Combined Score: {combined_score}")
    print()

# Close the connection
conn.close()
