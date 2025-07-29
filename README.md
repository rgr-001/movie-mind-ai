<!-- README.md -->

<div align="center" class="section">
  <img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Movie%20Banner.jpg" alt="Movie Banner" width="100%" class="banner">
  <h1 style="color:#d7335f; font-size: 3.5em;">🎬 Movie Mind AI</h1>
  <p style="font-size: 1.3em; color: #222;">An AI-powered Smart Movie Recommender System</p>
  <p class="highlight">✨ Built with 💛 by <strong>Rittik Gourav Raul</strong> | 🎓 OUTR, Bhubaneswar</p>
</div>

---
<div align="center">
  <img src="https://img.shields.io/badge/Made%20With-Python-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Model-TF--IDF%20%2B%20Cosine%20Similarity-yellowgreen?style=for-the-badge">
  <img src="https://img.shields.io/badge/Visuals-Matplotlib%20%26%20Seaborn-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-lightgrey?style=for-the-badge">
  <img src="https://img.shields.io/badge/Final-Project-critical?style=for-the-badge">
</div>

---

## 🎯 Overview
Movie Mind AI is a content-based recommender system that understands movie overviews and delivers the most relevant movie suggestions based on your interests.

**Key Features:**
- 🧠 Natural Language Understanding (TF-IDF + Cosine Similarity)
- 📈 Beautiful Data Visualizations
- ⚙️ Easy to use and extend
- 🎯 Great for ML beginners and intermediates

---

## 📁 Dataset Used
Source: [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Used Files:
- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`

---

## 🧰 Technologies

| 📌 Component | 🔧 Tools |
|-------------|----------|
| Language | Python 🐍 |
| Libraries | Pandas, Numpy, Scikit-learn 🔬 |
| Visualization | Matplotlib, Seaborn, WordCloud 📊 |
| Notebook | Jupyter 📒 |

---

## 🖼️ Visual Insights

### 🎞️ Top 10 Longest Movie Overviews
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Top%2010%20Longest%20Movie%20Overviews.png" alt="Longest Overviews" width="100%">

### 📊 Genre Distribution
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Genre%20Distribution.png" alt="Genre Distribution" width="100%">

### ☁️ Word Cloud of Genres
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Word%20Cloud%20of%20Genres.png" alt="Genre WordCloud" width="100%">

### 🧠 Top 20 Common Words in Overviews
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Top%2020%20Common%20Words%20in%20Overviews.png" alt="Word Frequency" width="100%">

---

## 🧠 How It Works
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['overview'])

# Cosine Similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_movies(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:6]]
    return df['title'].iloc[movie_indices]
```

---

## 📊 Sample Output
```bash
🎥 Recommendations for "Inception":
1. The Prestige
2. Interstellar
3. Memento
4. The Matrix
5. The Thirteenth Floor
```

---

## 📂 Folder Structure
```
📁 movie-mind-ai/
├── 📜 movie_recommender.ipynb
├── 📄 README.md
├── 📊 visualizations
│   ├── Movie Banner.jpg
│   ├── Genre Distribution.png
│   ├── Top 10 Longest Movie Overviews.png
│   ├── Top 20 Common Words in Overviews.png
│   └── Word Cloud of Genres.png
├── 📁 dataset
│   ├── movies_metadata.csv
│   ├── credits.csv
│   └── keywords.csv
```

---

## 🧑‍💻 Author
📌 Developed by **Rittik Gourav Raul**  
🎓 B.Tech, OUTR Bhubaneswar  
📁 GitHub: [rgr-001](https://github.com/rgr-001)

---

## 🏷️ Tags & Badges
<div align="center">
  <img src="https://img.shields.io/badge/Made%20With-Python-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Model-TF--IDF%20%2B%20Cosine%20Similarity-yellowgreen?style=for-the-badge">
  <img src="https://img.shields.io/badge/Visuals-Matplotlib%20%26%20Seaborn-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-lightgrey?style=for-the-badge">
</div>

---

## 🌟 Show Some Love
If you found this project useful, please ⭐ it and share it. Contributions, suggestions, and feedback are always welcome!
