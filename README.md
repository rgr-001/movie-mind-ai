<!-- README.md -->

<div align="center">
  <img src="Movie Banner.jpg" alt="Movie Banner" style="width:100%; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.3);">
  <h1 style="font-family: 'Segoe UI', sans-serif; color: #d7335f; font-size: 3em; margin-top: 20px;">🎬 Movie Recommendation System</h1>
  <p style="font-size: 1.3em; color: #444;">Built using <b>TF-IDF + Cosine Similarity</b> with an interactive and data-rich approach!</p>
</div>

---

## 📂 Overview
This project provides personalized movie recommendations based on **content-based filtering** using **text similarity**. It includes:

- 🔍 TF-IDF vectorization of movie overviews
- 🎯 Cosine similarity to find most similar titles
- 📊 Beautiful visualizations of genres and words
- 🌐 Interactive, scalable and easy-to-extend pipeline

---

## 📁 Dataset
Kaggle: [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) 🎥

Files used:
- `movies_metadata.csv`
- `keywords.csv`
- `credits.csv`

---

## 🧰 Tech Stack
- Python 🐍
- Pandas
- Scikit-learn 🔬
- Seaborn & Matplotlib 📈
- WordCloud ☁️
- Jupyter Notebook

---

## 📸 Visualizations

### 🎞️ Top 10 Longest Movie Overviews
<img src="Top 10 Longest Movie Overviews" alt="Longest Overviews" width="100%">

### 📊 Genre Distribution
<img src="Genre Distribution" alt="Genre Distribution" width="100%">

### ☁️ Word Cloud of Genres
<img src="Word Cloud of Genres" alt="Genre WordCloud" width="100%">

### 🧠 Top 20 Common Words in Overviews
<img src="Top 20 Common Words in Overviews" alt="Word Frequency" width="100%">

---

## 🔄 How It Works
```python
# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Build similarity matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
```

---

## ✅ Sample Output

🎥 Recommendations for **"Inception"**:
- The Prestige
- Interstellar
- Memento
- The Matrix
- The Thirteenth Floor

---

## 📂 Project Structure
```bash
📁 Movie-Recommendation-System/
├── 📜 movie_recommender.ipynb
├── 📊 visualizations/
│   ├── genre_bar.png
│   ├── wordcloud.png
│   ├── top_overviews.png
│   └── common_words.png
├── 📄 README.md
├── 📁 dataset/
│   ├── movies_metadata.csv
│   ├── credits.csv
│   └── keywords.csv
```

---

## ✨ Features
- Clean interface for recommendations
- Explorable visuals
- Optimized for 2000+ entries 🧠
- Easy to plug into web apps or APIs

---

## 🙌 Credits
Thanks to [Kaggle Datasets](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and [Scikit-learn](https://scikit-learn.org/) ❤️

---

<div align="center">
  <img src="https://img.shields.io/badge/Project-Movie_Recommender-red?style=for-the-badge&logo=python" alt="Project Badge">
  <img src="https://img.shields.io/badge/Built%20With-TF-IDF-yellow?style=for-the-badge&logo=scikit-learn" alt="Tech Badge">
  <img src="https://img.shields.io/badge/Visualized%20With-Matplotlib-blue?style=for-the-badge&logo=seaborn" alt="Visual Badge">
</div>
