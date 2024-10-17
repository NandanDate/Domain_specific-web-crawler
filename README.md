# 🌐 Domain-Specific Keyword-Centric Web Crawler

## 📘 Introduction

In today's digital landscape, the vast volume of online information poses a challenge for traditional web crawlers, which often rely on hyperlink-driven exploration. This approach can be inefficient when it comes to finding specific content. To address this, we've developed a **Domain-Specific Keyword-Centric Web Crawler** that prioritizes exact keyword searches over random hyperlink traversal. By focusing on user-defined keywords, our crawler efficiently filters through web pages, discarding irrelevant content and delivering highly targeted results. Leveraging advanced parsing and search algorithms, this approach enhances the precision and relevance of search results, streamlines information retrieval, improves recommendations, and simplifies data indexing—making it indispensable for effective web exploration in specialized domains.

## 🎯 Objective

Our objective is to design and implement a domain-specific web crawler that enhances precision in targeted search results by leveraging keyword extraction from web content.

## 🗂️ Data Set

For this project, we've chosen **Wikipedia** as our primary data source due to its vast repository of information on a wide range of topics. Wikipedia's extensive database is publicly accessible, making it ideal for our goal of providing open and unrestricted information. The availability of the Wikipedia API further allows us to efficiently retrieve and integrate data into our application. Given Wikipedia's community-driven approach and rigorous editorial standards, we can ensure that the information presented is accurate, up-to-date, and reliable.

## 🏗️ System Model

The system architecture is designed to optimize search efficiency and relevance through a structured workflow that includes user interaction, content extraction, analysis, and recommendation generation. The process begins by asking the user to specify a topic and, optionally, a category of interest. This input is used to tailor content extraction and analysis.

### Workflow:

1. **User Input**: The user specifies a topic and optional category of interest, which guides the construction of relevant URLs.
2. **Wikipedia URLs**: Based on the user's input, the system constructs 5 (default) Wikipedia URLs to serve as seed URLs, ensuring high-quality, topic-specific content.
3. **Category Filter**: A category filter is applied to ensure that the results fall within the desired scope, narrowing down the selection to the most relevant articles.
4. **Content Extraction**: The system fetches and parses content from the selected Wikipedia URLs, generating summaries by extracting the first three paragraphs from each page.
5. **Google Search Integration**: To broaden the data sources, the system also fetches the top 5 Google search results for the same topic.
6. **Text Processing**: The extracted content is tokenized, and Part-of-Speech (POS) tagging is applied. Named Entity Recognition (NER) is used to identify key entities like people, organizations, and locations.
7. **Keyword Clustering**: The identified keywords are clustered based on semantic similarity, organizing them into meaningful groups. The user can then select the cluster of most interest.
8. **Additional Content Fetching**: For each keyword in the selected cluster, an additional Wikipedia URL is constructed and stored in a SQLite database alongside its keyword. The system then retrieves content from these URLs, generates summaries, and queries Google for further relevant URLs.
9. **Custom Ranking**: A custom ranking algorithm refines the order of the URLs based on parameters like recency, sentiment, similarity, keyword density, and NER score, ensuring that the most pertinent results are prioritized.

This comprehensive approach ensures that the user receives the most relevant and comprehensive information available on their selected topic.

## ⚙️ Implementation & Results

The implementation follows a structured approach, beginning with user input and progressing through content extraction, analysis, and ranking:

1. **User Input**: The user specifies a topic and optional category, guiding the construction of relevant URLs.
2. **Wikipedia URL Construction**: Tailored Wikipedia URLs are created based on the user's input, focusing the content extraction and analysis.
3. **Category Filtering**: A filter is applied to ensure that the extracted content falls within the desired scope.
4. **URL Selection**: By default, the system uses the top 5 Wikipedia URLs, balancing the breadth and depth of the content.
5. **Content Extraction and Analysis**: The system fetches, parses, and summarizes the content, integrating additional data from Google search results and applying advanced text processing techniques.
6. **Custom Ranking and Presentation**: The top-ranked URLs are presented to the user, ensuring the most relevant and comprehensive results.

This process ensures efficient and targeted information retrieval, making the crawler a valuable tool for domain-specific web exploration.
