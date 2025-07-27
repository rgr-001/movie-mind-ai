# 🎥 Movie Recommendation System using NLP & TF-IDF

## 📌 Project Overview  
This project builds a **Content-Based Movie Recommendation System** using **Natural Language Processing (NLP)** techniques. It analyzes the `overview` of each movie using **TF-IDF Vectorization** and measures similarity using **cosine similarity** to suggest movies similar to a user's choice.

---

## 🛠️ Tools & Technologies  
- 🐍 Python  
- 🧠 Scikit-learn (`TfidfVectorizer`, `linear_kernel`)  
- 📊 Pandas & Matplotlib  
- ☁️ WordCloud  
- 🧼 Jupyter Notebook  
- 🗂️ Dataset: `movies_metadata.csv` from Kaggle

---

## 📁 Dataset Summary  
- Total movies: ~45,000  
- Used features: `title`, `overview`  
- Preprocessing:  
  - Removed `NaN` overviews  
  - Lowercased movie titles  
  - Cleaned unnecessary columns  

---

## 🔍 Project Workflow  
1. **📥 Load Data**: Imported metadata CSV using `pandas`.  
2. **🧹 Data Cleaning**: Dropped rows with missing overviews.  
3. **🧠 Feature Extraction**: Applied `TfidfVectorizer` on movie overviews.  
4. **📏 Similarity Calculation**: Used `linear_kernel` to compute cosine similarity between movie vectors.  
5. **🎯 Recommendation Function**: Created a function that returns top 5 similar movies based on input title.  
6. **📊 Visualization**:  
   - Bar chart for top 10 genres  
   - Word cloud of all overviews  

---

## 🧪 Sample Output  
**Input:** `"The Dark Knight"`  
**Recommendations:**  
- Batman Begins  
- The Prestige  
- The Dark Knight Rises  
- Inception  
- Man of Steel

---

## 📈 Additional Features  
- 📌 Word Cloud of Overview Texts  
- 📊 Genre Frequency Distribution  
- 🔎 Case-insensitive Title Matching  
- 🚨 Graceful Error Handling (`Movie not found` message)

---

## ✅ Status: **Completed**  
> Project tested on subset of 2,000 movies for efficiency on limited systems.  
> Final notebook runs without errors and includes all visuals and functions.