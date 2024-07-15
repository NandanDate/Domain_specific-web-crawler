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
