<!-- README.md -->

<div align="center">
  <img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Movie%20Banner.jpg" alt="Movie Banner" width="100%" style="border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.3);">
  <h1 style="font-family: 'Segoe UI', sans-serif; color: #d7335f; font-size: 3em; margin-top: 20px; text-shadow: 2px 2px 4px #000000;">🎬 Movie Mind AI</h1>
  <p style="font-size: 1.3em; color: #555; font-style: italic;">Smart Movie Recommendation System using NLP + Visual Intelligence</p>
</div>

<hr style="border: 1px solid #e0e0e0; margin: 30px 0;">

## 📘 Overview
Movie Mind AI is an intelligent movie recommendation system powered by **TF-IDF**, **Cosine Similarity**, and **NLP**. Dive into the world of movies with:

- 🧠 Text-based recommendation system
- 🧾 Analysis of movie overviews
- 📊 Genre and keyword visualizations
- 🌐 Easy-to-understand insights for movie lovers

---

## 📂 Dataset Source
- 📦 **[Kaggle: The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)**

Files used:
- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`

---

## 🛠️ Tech Stack
| Tool | Description |
|------|-------------|
| Python 🐍 | Core programming language |
| Scikit-learn 🔬 | ML and similarity modeling |
| Pandas & NumPy | Data analysis |
| Matplotlib & Seaborn 📈 | Data visualization |
| WordCloud ☁️ | Word cloud generation |

---

## 🖼️ Visual Explorations

### 🎞️ Top 10 Longest Movie Overviews
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Top%2010%20Longest%20Movie%20Overviews.png" alt="Longest Overviews" width="100%">

### 📊 Genre Distribution
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Genre%20Distribution.png" alt="Genre Distribution" width="100%">

### ☁️ Word Cloud of Genres
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Word%20Cloud%20of%20Genres.png" alt="Genre WordCloud" width="100%">

### 🧠 Top 20 Common Words in Overviews
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Top%2020%20Common%20Words%20in%20Overviews.png" alt="Word Frequency" width="100%">

---

## ⚙️ How It Works
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

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
```bash
🎥 Recommendations for "Inception":
1. The Prestige
2. Interstellar
3. Memento
4. The Matrix
5. The Thirteenth Floor
```

---

## 📁 Project Structure
```bash
📦 movie-mind-ai/
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
- 💡 Intelligent content-based recommendation
- 📊 Beautiful genre & keyword insights
- 🌐 Easy for deployment with Flask or Streamlit
- 🎓 Great for students building ML/NLP portfolios

---

## 🙏 Credits
Made possible by:
- 📚 Scikit-learn + TF-IDF
- 📂 [Kaggle Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- 🧪 Jupyter, Pandas, WordCloud

---

<div align="center">
  <img src="https://img.shields.io/badge/Made%20With-Python-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Model-TF--IDF%20%2B%20Cosine%20Similarity-yellowgreen?style=for-the-badge">
  <img src="https://img.shields.io/badge/Visuals-Matplotlib%20%26%20Seaborn-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-lightgrey?style=for-the-badge">
</div>

---

<style>
  h1, h2, h3, p, ul, li {
    font-family: 'Segoe UI', sans-serif;
  }
  code {
    background-color: #f0f0f0;
    padding: 4px 6px;
    border-radius: 4px;
    font-size: 95%;
  }
</style>
