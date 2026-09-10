````markdown
# 📧 Spam Email Detection Chatbot (NLP)

A full-stack **Spam Email/SMS Detection** project using **NLP and Machine Learning**. Users can enter a message through a React chatbot interface, and the application predicts whether it is **SPAM** or **NOT SPAM**.

## 🛠️ Technologies

- Python
- Flask
- React.js
- JavaScript
- Scikit-learn
- Pandas
- TF-IDF
- LinearSVC
- Joblib

## 🔄 Workflow

```text
User Message
     ↓
React Frontend
     ↓
Flask API
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
LinearSVC Model
     ↓
SPAM / NOT SPAM
````

## 📂 Project Structure

```text
spam/
├── Backend/
│   ├── app.py
│   ├── train_model.py
│   ├── spam.csv
│   ├── spam_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── requirements.txt
├── Frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
└── Spam Detection(NLP).ipynb
```

## ▶️ Run the Project

### Backend

```bash
cd Backend
pip install -r requirements.txt
python train_model.py
python app.py
```

Backend runs on:

```text
http://127.0.0.1:5000
```

### Frontend

```bash
cd Frontend
npm install
npm run dev
```

Frontend runs on the Vite URL shown in the terminal.

## 🔗 API

```text
GET /api/predict?message=Congratulations%20you%20won%20a%20free%20prize
```

Example result:

```json
{
  "prediction": "SPAM",
  "label": 1,
  "status": "success"
}
```

## 🔐 Security

Do not upload `.env` or API keys to GitHub.

```gitignore
.env
Backend/.env
node_modules/
__pycache__/
```

## 👩‍💻 Author

**Kopparthi Lakshmi Sukadha**

```
```
