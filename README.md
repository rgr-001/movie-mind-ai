<!-- README.md -->

<div align="center">
  <img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Movie%20Banner.jpg" alt="Movie Banner" width="100%" style="border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.3);">
  <h1 style="font-family: 'Segoe UI', sans-serif; color: #d7335f; font-size: 3em; margin-top: 20px;">🎬 Movie Mind AI</h1>
  <p style="font-size: 1.3em; color: #444;">Smart Movie Recommendation System using TF-IDF, Cosine Similarity, and Visual Intelligence</p>
</div>

---

## 📂 Overview
Movie Mind AI recommends movies by analyzing their overviews using natural language processing. Key highlights include:

- 🧠 TF-IDF + Cosine Similarity for smart content matching
- 📊 Data visualizations on genres and keywords
- ☁️ Word clouds and bar charts
- 🎯 Personalized suggestions based on description similarity

---

## 📁 Dataset
📦 Source: [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Used files:
- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`

---

## 🧰 Tech Stack
- Python 🐍
- Pandas, NumPy
- Scikit-learn 🔬
- Matplotlib & Seaborn 📊
- WordCloud ☁️
- Jupyter Notebook

---

## 📸 Visualizations

### 🎞️ Top 10 Longest Movie Overviews
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Top%2010%20Longest%20Movie%20Overviews.png" alt="Longest Overviews" width="100%">

### 📊 Genre Distribution
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Genre%20Distribution.png" alt="Genre Distribution" width="100%">

### ☁️ Word Cloud of Genres
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Word%20Cloud%20of%20Genres.png" alt="Genre WordCloud" width="100%">

### 🧠 Top 20 Common Words in Overviews
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Top%2020%20Common%20Words%20in%20Overviews.png" alt="Word Frequency" width="100%">

---

## 🔄 How It Works
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Vectorize overview
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Cosine similarity
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
📁 movie-mind-ai/
├── 📜 movie_recommender.ipynb
├── 📄 README.md
├── 📊 visualizations/
│   ├── Movie Banner.jpg
│   ├── Genre Distribution.png
│   ├── Top 10 Longest Movie Overviews.png
│   ├── Top 20 Common Words in Overviews.png
│   └── Word Cloud of Genres.png
├── 📁 dataset/
│   ├── movies_metadata.csv
│   ├── credits.csv
│   └── keywords.csv
```

---

## ✨ Features
- Lightweight, fast & intelligent recommender
- Meaningful visual storytelling
- Easy to extend for web or API integrations
- Perfect for beginner to intermediate ML portfolios

---

## 🙌 Credits
Built using:
- 🧪 Scikit-learn
- 💻 Python & Jupyter
- 📈 Matplotlib, Seaborn, WordCloud
- 📂 [Kaggle Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

---

<div align="center">
  <img src="https://img.shields.io/badge/Made%20With-Python-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Model-TF--IDF%20%2B%20Cosine%20Similarity-yellowgreen?style=for-the-badge">
  <img src="https://img.shields.io/badge/Visuals-Matplotlib%20%26%20Seaborn-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-lightgrey?style=for-the-badge">
</div>
