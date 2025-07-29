<!-- README.md -->


    font-family: 'Segoe UI', sans-serif;
  }
</style>

<div align="center">
  <img src="https://raw.githubusercontent.com/your-username/your-repo-name/main/Movie%20Banner.jpg" alt="Movie Banner" style="width:100%; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.3);">
  <h1 style="color: #d7335f; font-size: 3em; margin-top: 20px;">🎬 Advanced Movie Recommendation System</h1>
  <p style="font-size: 1.3em; color: #444;">A smart and interactive system built with <b>TF-IDF + Cosine Similarity</b> to suggest movies like magic! ✨</p>
</div>

---

## 📚 Project Highlights
- 📽️ Recommends movies based on content similarity
- 🧠 NLP powered by TF-IDF Vectorizer and Cosine Similarity
- 🎨 Stunning visualizations: Word Clouds, Bar Charts, Insights
- 🚀 Optimized for speed with 2000+ entries
- 💡 Educational and Production-ready notebook

---

## 📁 Dataset
Source: [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Files used:
- `movies_metadata.csv`
- `keywords.csv`
- `credits.csv`

---

## 🧰 Technologies Used
- `Python` 🐍
- `Pandas`, `NumPy`
- `Scikit-learn`, `WordCloud`
- `Matplotlib`, `Seaborn`
- `Jupyter Notebook`

---

## 🔍 How It Works
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# TF-IDF on movie overviews
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommend movies
indices = pd.Series(df.index, index=df['title'])
def recommend_movies(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
```

---

## 📈 Visual Explorations

### 🎞️ Top 10 Longest Movie Overviews
<img src="https://raw.githubusercontent.com/your-username/your-repo-name/main/Top%2010%20Longest%20Movie%20Overviews" alt="Longest Overviews" width="100%">

### 🎭 Genre Distribution
<img src="https://raw.githubusercontent.com/your-username/your-repo-name/main/Genre%20Distribution" alt="Genre Distribution" width="100%">

### ☁️ Word Cloud of Genres
<img src="https://raw.githubusercontent.com/your-username/your-repo-name/main/Word%20Cloud%20of%20Genres" alt="Word Cloud" width="100%">

### 🔠 Top 20 Common Words in Overviews
<img src="https://raw.githubusercontent.com/your-username/your-repo-name/main/Top%2020%20Common%20Words%20in%20Overviews" alt="Common Words" width="100%">

---

## 🎯 Sample Output
📽️ Recommendations for **"Inception"**:
```
1. Interstellar
2. The Prestige
3. Memento
4. The Matrix
5. The Thirteenth Floor
```

---

## 📂 Directory Structure
```bash
📁 Movie-Recommendation-System/
├── 📘 movie_recommender.ipynb
├── 📊 visualizations/
│   ├── Genre Distribution
│   ├── Top 10 Longest Movie Overviews
│   ├── Word Cloud of Genres
│   └── Top 20 Common Words in Overviews
├── 📄 README.md
├── 📁 dataset/
│   ├── movies_metadata.csv
│   ├── credits.csv
│   └── keywords.csv
```

---

## 🧠 Tags
`#TF-IDF` `#MovieRecommendation` `#ContentFiltering` `#NLP` `#DataScience` `#Jupyter` `#Visualization` `#Python` `#MachineLearning`

---

## 🙌 Credits
- Dataset by [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- Libraries: `Scikit-learn`, `Matplotlib`, `Seaborn`, `WordCloud`

---

<div align="center">
  <img src="https://img.shields.io/badge/Project-Movie_Recommender-red?style=for-the-badge&logo=python" alt="Badge">
  <img src="https://img.shields.io/badge/Technique-TF--IDF-yellow?style=for-the-badge&logo=scikit-learn" alt="Badge">
  <img src="https://img.shields.io/badge/Visualized%20With-Matplotlib-blue?style=for-the-badge&logo=seaborn" alt="Badge">
</div>
