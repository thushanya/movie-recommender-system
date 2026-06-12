# 🎬 Movie Recommendation System (Hybrid ML Project)

A machine learning-based movie recommendation system built using **MovieLens 100K dataset**.  
The system combines **Collaborative Filtering (SVD)** and **Content-Based Filtering (TF-IDF + Cosine Similarity)** and provides an interactive **Streamlit web application**.

---

## 🚀 Live Features

- 🎯 Content-Based Recommendations (based on movie genres)
- 🤝 Collaborative Filtering (SVD Matrix Factorization)
- 🔥 Hybrid Recommendation System (combines both approaches)
- ⭐ Trending Movies (based on ratings + popularity)
- 📊 Top Rated Movies dashboard
- 🎛 Interactive Streamlit UI
- ⚡ Cached ML model for fast performance

---

## 🧠 Machine Learning Techniques Used

### 1. Collaborative Filtering
- Algorithm: **Singular Value Decomposition (SVD)**
- Library: `scikit-surprise`
- Predicts user ratings based on similar users

### 2. Content-Based Filtering
- Technique: TF-IDF Vectorization
- Similarity: Cosine Similarity
- Uses movie genres to find similar movies

### 3. Hybrid System
- Weighted combination of CF + CBF
- Produces more accurate recommendations

---

## 📂 Dataset

**MovieLens 100K Dataset**
- Source: GroupLens
- Contains:
  - 100,000+ ratings
  - 1,000+ users
  - 1,600+ movies

---

## 🏗 Project Structure
movie-recommender-system/
│
├── data/
│ ├── movies.csv
│ ├── ratings.csv
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_collaborative_filtering.ipynb
│ ├── 03_content_based.ipynb
│
├── src/
│ ├── utils.py
│ ├── collaborative.py
│ ├── content_based.py
│ ├── hybrid.py
│
├── app.py
├── convert_data.py
├── requirements.txt
├── test_models.py
└── README.md


---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/movie-recommender-system.git
cd movie-recommender-system

2. Create Virtual Environment
python -m venv venv

Activate:
Windows
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run Streamlit App
streamlit run app.py


📊 System Workflow

User selects a movie
        ↓
Content-Based Filtering (similar genres)
        ↓
Collaborative Filtering (user behavior)
        ↓
Hybrid Model combines both
        ↓
Final ranked recommendations



🔥 Key Highlights

Hybrid recommendation improves accuracy
Cached ML models improve performance
Clean and responsive UI
Real-world recommendation system architecture



🧪 Example Output

Input:

Toy Story (1995)

Output:

Toy Story 2
Aladdin
Bug’s Life
Lion King
Monsters Inc.



👨‍💻 Technologies Used

Python
Pandas
Scikit-learn
Surprise (SVD)
Streamlit
NumPy



📜 License

This project is for educational purposes.

⭐ Author

Built as a Machine Learning academic project demonstrating:

Recommendation Systems
Hybrid ML Models
Interactive Web Deployment