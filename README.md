---
<!-- Banner Image -->
<p align="center">
  <img src="images/banner.jpg" alt="Movie Recommendation Banner" width="100%">
</p>

<h1 align="center">🎬 Movie Recommendation System 📽️</h1>

<p align="center">
  <b>🔍 Discover the best movies tailored to your taste using Natural Language Processing (NLP) & Content-Based Filtering! 🎯</b>
</p>

---

## 🧠 Project Overview

This project leverages **TF-IDF**, **cosine similarity**, and **movie metadata** to provide content-based movie recommendations. It also includes advanced data visualizations to better understand the dataset, such as:

- 📊 Genre Distributions
- ☁️ Word Clouds
- 🔠 Top Words in Overviews
- 🔥 Longest Movie Overviews

> Built with ❤️ using Python, Pandas, Matplotlib, Seaborn, and Scikit-learn.

---

## 📁 Dataset Used

> 🔗 [TMDb 5000 Movie Dataset - Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`

We extracted a **subset of 2000 movies** to ensure faster computation.

---

## ⚙️ Technologies Used

| Technology       | Description                             |
|------------------|-----------------------------------------|
| 🐍 Python         | Programming Language                    |
| 🧾 Pandas         | Data Manipulation                       |
| 📊 Matplotlib     | Data Visualization                      |
| 🌊 Seaborn        | Statistical Plots                       |
| 💬 Scikit-learn   | TF-IDF Vectorizer & Cosine Similarity  |
| ☁️ WordCloud      | Text Visualizations                     |

---

## 🔍 How It Works

1. **Data Cleaning** - Removing nulls & formatting genres.
2. **Text Vectorization** - Using TF-IDF on movie overviews.
3. **Similarity Calculation** - Cosine similarity to find movie neighbors.
4. **Recommendation Function** - Input movie returns similar titles.

---

## 📌 Example Visualizations

### 📈 Genre Distribution
<img src="images/genre_bar.png" width="60%">

### 📘 Top 10 Longest Movie Overviews
<img src="images/longest_overviews.png" width="60%">

### ☁️ Word Cloud of Genres
<img src="images/wordcloud_genres.png" width="60%">

### 🔠 Top 20 Common Words in Overviews
<img src="images/top_words.png" width="60%">

---

## 💡 Sample Output

```
🎬 Top 5 Recommendations for 'Inception':
1. The Manchurian Candidate
2. Mulholland Falls
3. Heat
4. Desperado
5. Dingo
```

---

## 🧩 Project Structure

```
├── README.md
├── movie_recommender.ipynb
├── movies_metadata.csv
├── images/
│   ├── banner.jpg
│   ├── genre_bar.png
│   ├── longest_overviews.png
│   ├── top_words.png
│   └── wordcloud_genres.png
```

---

## 🚀 How to Run

1. Clone the repo or download the files
2. Open `movie_recommender.ipynb` in **Jupyter Notebook** or **Google Colab**
3. Run all cells to see:
   - Visualizations
   - Recommendation engine in action

---

## 🤝 Contributions

Feel free to fork the repo, open issues, or suggest improvements!

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ for movie lovers and data science enthusiasts.
</p>


<p align="center">Made with ❤️ by <strong>Rittik Gourav Raul from OUTR,BBSR</strong></p>
