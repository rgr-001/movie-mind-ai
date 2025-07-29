<!-- README.md -->

<div align="center">
  <img src="https://github.com/your-username/your-repo-name/blob/main/Movie%20Banner.png?raw=true" alt="Movie Banner" width="100%" style="border-radius: 12px; box-shadow: 0 5px 20px rgba(0,0,0,0.2);">
  <h1 style="color:#d63384; font-size:3rem; font-family:'Segoe UI', sans-serif;">🍿 Movie Recommendation System</h1>
  <p style="font-size: 1.2rem; color: #555;">A visually rich and intelligent system to suggest movies you might love! 🚀</p>
</div>

---

## 🔍 Overview
🎥 Content-based filtering using TF-IDF and Cosine Similarity.
✨ Includes:
- Intelligent matching of similar movies
- Data visualization with genre analysis & word insights
- Simple, fast & accurate recommendations

---

## 📦 Dataset
Kaggle: [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- `movies_metadata.csv`
- `keywords.csv`
- `credits.csv`

---

## 💻 Tech Stack
| Tool | Description |
|------|-------------|
| Python 🐍 | Core Language |
| Scikit-learn ⚙️ | TF-IDF, Cosine Similarity |
| Pandas | Data Handling 🧾 |
| Matplotlib & Seaborn 📊 | Visuals & Insights |
| WordCloud ☁️ | Genre & Word Visuals |
| Jupyter Notebook 📓 | Implementation |

---

## 🌟 Visualizations

### 🎞️ Top 10 Longest Movie Overviews
<img src="https://github.com/your-username/your-repo-name/blob/main/Top%2010%20Longest%20Movie%20Overviews.png?raw=true" alt="Longest Overviews" width="100%">

### 📊 Genre Distribution
<img src="https://github.com/your-username/your-repo-name/blob/main/Genre%20Distribution.png?raw=true" alt="Genre Distribution" width="100%">

### ☁️ Word Cloud of Genres
<img src="https://github.com/your-username/your-repo-name/blob/main/Word%20Cloud%20of%20Genres.png?raw=true" alt="Genre WordCloud" width="100%">

### 🧠 Top 20 Common Words in Overviews
<img src="https://github.com/your-username/your-repo-name/blob/main/Top%2020%20Common%20Words%20in%20Overviews.png?raw=true" alt="Top Words" width="100%">

---

## 🧠 How It Works
```python
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation Function
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
Recommendations for 'Inception':
1. The Prestige
2. Interstellar
3. Memento
4. The Matrix
5. The Thirteenth Floor
```

---

## 📁 Project Structure
```
Movie-Recommendation-System/
├── README.md
├── movie_recommender.ipynb
├── dataset/
│   ├── movies_metadata.csv
│   ├── credits.csv
│   └── keywords.csv
├── Genre Distribution.png
├── Top 10 Longest Movie Overviews.png
├── Word Cloud of Genres.png
├── Top 20 Common Words in Overviews.png
└── Movie Banner.png
```

---

## 🚀 Features
- 🎬 Content-based recommendation engine
- 📈 Visual genre & overview analysis
- 🧠 Optimized for 2,000+ movie entries
- 📦 Easy to use & extend

---

## 🙌 Credits
Thanks to [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) & the open-source community ❤️

---

<div align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Framework-ScikitLearn-orange?style=for-the-badge&logo=scikit-learn">
  <img src="https://img.shields.io/badge/Notebook-Jupyter-yellow?style=for-the-badge&logo=jupyter">
</div>

---

> 📫 _Made with ❤️ by 
**Rittik Gourav Raul**
**OUTR,BBSR**
