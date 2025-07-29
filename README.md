<!-- README.md -->

<div align="center">
  <img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Movie%20Banner.jpg" alt="Movie Banner" width="100%" style="border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.3);">
  <h1 style="font-family: 'Segoe UI', sans-serif; color: #d7335f; font-size: 3.5em; margin-top: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🎬 Movie Mind AI</h1>
  <p style="font-size: 1.4em; color: #555; font-weight: 500;">Your Intelligent Movie Recommendation System powered by Machine Learning & NLP</p>
</div>

---

## 💡 Project Overview

Movie Mind AI uses smart Natural Language Processing techniques like **TF-IDF** and **Cosine Similarity** to recommend similar movies based on their descriptions. It also includes beautiful visualizations and genre insights to help you explore movie trends!

---

## 🧠 Tech Stack

| Tool           | Use Case                          |
|----------------|-----------------------------------|
| Python 🐍      | Main Programming Language         |
| Pandas & NumPy | Data manipulation                 |
| Scikit-learn 🔬| ML models and vectorization       |
| Matplotlib 📊  | Data visualization                |
| WordCloud ☁️   | Genre and keyword clouds          |
| Jupyter 📓     | Notebook for development          |

---

## 📦 Dataset Source

[Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

**Used files:**
- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`

---

## ✨ Key Features

- 🎯 Content-based movie recommender using TF-IDF
- 📑 Cosine similarity for overview matching
- 🎨 Beautiful visual insights: genre charts, word clouds, frequency bars
- ⚙️ Easy to customize and scale
- 👤 Built with ❤️ by **Rittik Gourav Raul** from **OUTR, BBSR**

---

## 🖼️ Visualizations

### 🎞️ Top 10 Longest Movie Overviews
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Top%2010%20Longest%20Movie%20Overviews.png" alt="Longest Overviews" width="100%">

### 📊 Genre Distribution
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Genre%20Distribution.png" alt="Genre Distribution" width="100%">

### ☁️ Word Cloud of Genres
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Word%20Cloud%20of%20Genres.png" alt="Genre WordCloud" width="100%">

### 📚 Top 20 Common Words in Overviews
<img src="https://raw.githubusercontent.com/rgr-001/movie-mind-ai/main/Top%2020%20Common%20Words%20in%20Overviews.png" alt="Word Frequency" width="100%">

---

## 🔁 How It Works

```python
# Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute Cosine Similarity
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

## 🎥 Sample Output

```bash
Recommendations for "Inception":
1. Interstellar
2. The Prestige
3. The Matrix
4. Memento
5. The Thirteenth Floor
```

---

## 📁 Project Structure

```bash
📁 movie-mind-ai/
├── 📜 movie_recommender.ipynb
├── 📄 README.md
├── 📊 Visualizations/
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

## 🏷️ Tags

`#MovieRecommendation` `#MachineLearning` `#NLP` `#Python` `#TFIDF` `#CosineSimilarity` `#WordCloud` `#DataViz`

---

## 🙌 Author

Made with 💻 by **Rittik Gourav Raul**
🎓 B.Tech Student, OUTR, Bhubaneswar
🔗 [GitHub](https://github.com/rgr-001)

---

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-yellow?style=for-the-badge">
  <img src="https://img.shields.io/badge/Notebook-Jupyter-orange?style=for-the-badge&logo=jupyter">
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%26%20Cosine%20Sim-lightgreen?style=for-the-badge">
</div>
