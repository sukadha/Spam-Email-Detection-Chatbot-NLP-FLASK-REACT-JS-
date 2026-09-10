from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import re
import joblib
import os

from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# OPENAI CLIENT
# ============================================================

if OPENAI_API_KEY:

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

else:

    client = None


# ============================================================
# LOAD ML MODEL
# ============================================================

try:

    model = joblib.load(
        "spam_model.pkl"
    )

    print(
        "Spam model loaded successfully."
    )

except Exception as e:

    print(
        "Error loading spam model:",
        e
    )

    model = None


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

try:

    tfidf_vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    print(
        "TF-IDF vectorizer loaded successfully."
    )

except Exception as e:

    print(
        "Error loading vectorizer:",
        e
    )

    tfidf_vectorizer = None


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

try:

    # Load original spam.csv
    df = pd.read_csv(
        "spam.csv",
        encoding="latin-1"
    )

    print(
        "Original dataset loaded successfully."
    )


    # ========================================================
    # CHECK v1 AND v2 COLUMNS
    # ========================================================

    if "v1" not in df.columns or "v2" not in df.columns:

        raise ValueError(
            "spam.csv must contain v1 and v2 columns."
        )


    # ========================================================
    # KEEP ORIGINAL v1 AND v2
    # ========================================================

    df = df[
        ["v1", "v2"]
    ].copy()


    # ========================================================
    # REMOVE MISSING VALUES
    # ========================================================

    df["v1"] = df["v1"].fillna("")

    df["v2"] = df["v2"].fillna("")


    # ========================================================
    # REMOVE EMPTY MESSAGES
    # ========================================================

    df = df[
        df["v2"].astype(str).str.strip() != ""
    ].copy()


    # ========================================================
    # REMOVE INVALID LABELS
    # ========================================================

    df = df[
        df["v1"].isin(
            ["ham", "spam"]
        )
    ].copy()


    # ========================================================
    # CREATE ML LABEL
    #
    # ham  = 0
    # spam = 1
    # ========================================================

    df["label"] = df["v1"].map({
        "ham": 0,
        "spam": 1
    })


    # ========================================================
    # CREATE MESSAGE COLUMN
    # v2 -> message
    # ========================================================

    df["message"] = df["v2"]


    # ========================================================
    # CREATE CLEAN TEXT
    # ========================================================

    df["clean_text"] = df["v2"].apply(
        clean_text
    )


    # ========================================================
    # REMOVE EMPTY CLEAN TEXT
    # ========================================================

    df = df[
        df["clean_text"].str.strip() != ""
    ].copy()


    # ========================================================
    # RESET INDEX
    # ========================================================

    df.reset_index(
        drop=True,
        inplace=True
    )


    print(
        "Dataset processed successfully."
    )

    print(
        "Total records:",
        len(df)
    )

    print(
        "Dataset columns:",
        list(df.columns)
    )


except Exception as e:

    print(
        "Error loading dataset:",
        e
    )

    df = pd.DataFrame()


# ============================================================
# CREATE DATASET SEARCH MATRIX
# ============================================================

if (
    not df.empty
    and tfidf_vectorizer is not None
):

    try:

        # Make sure clean_text has no NaN
        df["clean_text"] = (
            df["clean_text"]
            .fillna("")
            .astype(str)
        )


        # Create TF-IDF matrix
        dataset_matrix = (
            tfidf_vectorizer.transform(
                df["clean_text"].tolist()
            )
        )


        print(
            "Dataset search matrix created successfully."
        )

        print(
            "Search records:",
            len(df)
        )


    except Exception as e:

        print(
            "Error creating dataset search matrix:",
            e
        )

        dataset_matrix = None

else:

    dataset_matrix = None


# ============================================================
# HOME API
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "status": "success",

        "message":
        "Spam Email Detection Backend is running",

        "dataset":
        "spam.csv",

        "columns": [
            "v1",
            "v2",
            "label",
            "message",
            "clean_text"
        ],

        "apis": {

            "health":
            "GET /api/spam",

            "stats":
            "GET /api/stats",

            "examples":
            "GET /api/examples",

            "predict_get":
            "GET /api/predict?message=your message",

            "predict_post":
            "POST /api/predict",

            "search_get":
            "GET /api/search?query=your query",

            "search_post":
            "POST /api/search",

            "chat_get":
            "GET /api/chat?message=your question",

            "chat_post":
            "POST /api/chat"

        }

    })


# ============================================================
# HEALTH API
# ============================================================

@app.route(
    "/api/spam",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "success",

        "message":
        "Spam Detection Backend is running",

        "model_loaded":
        model is not None,

        "vectorizer_loaded":
        tfidf_vectorizer is not None,

        "dataset_loaded":
        not df.empty,

        "dataset_file":
        "spam.csv",

        "dataset_columns": [
            "v1",
            "v2"
        ],

        "total_records":
        int(len(df))

    })


# ============================================================
# PREDICTION API
# GET + POST
# ============================================================

@app.route(
    "/api/predict",
    methods=["GET", "POST"]
)
def predict():

    try:

        # ====================================================
        # CHECK MODEL
        # ====================================================

        if model is None:

            return jsonify({

                "status": "error",

                "message":
                "ML model is not loaded."

            }), 500


        # ====================================================
        # CHECK VECTORIZER
        # ====================================================

        if tfidf_vectorizer is None:

            return jsonify({

                "status": "error",

                "message":
                "TF-IDF vectorizer is not loaded."

            }), 500


        # ====================================================
        # GET MESSAGE
        # GET  -> ?message=
        # POST -> JSON
        # ====================================================

        if request.method == "GET":

            message = request.args.get(
                "message",
                ""
            )

        else:

            data = request.get_json(
                silent=True
            )

            if not data:

                return jsonify({

                    "status": "error",

                    "message":
                    "JSON request body is required."

                }), 400

            message = data.get(
                "message",
                ""
            )


        # ====================================================
        # MAKE SURE MESSAGE IS STRING
        # ====================================================

        message = str(message)


        if not message.strip():

            return jsonify({

                "status": "error",

                "message":
                "Email message is required."

            }), 400


        # ====================================================
        # CLEAN TEXT
        # ====================================================

        cleaned_message = clean_text(
            message
        )


        # ====================================================
        # TF-IDF
        # ====================================================

        message_vector = (
            tfidf_vectorizer.transform(
                [cleaned_message]
            )
        )


        # ====================================================
        # PREDICT
        # ====================================================

        prediction = model.predict(
            message_vector
        )[0]


        # ====================================================
        # DECISION SCORE
        # ====================================================

        decision_score = (
            model.decision_function(
                message_vector
            )[0]
        )


        # ====================================================
        # RESULT
        # ====================================================

        if prediction == 1:

            result = "SPAM"

            label = 1

        else:

            result = "NOT SPAM"

            label = 0


        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return jsonify({

            "status":
            "success",

            "prediction":
            result,

            "label":
            label,

            "v1_prediction":
            "spam"
            if label == 1
            else "ham",

            "v2_message":
            message,

            "clean_text":
            cleaned_message,

            "decision_score":
            round(
                float(decision_score),
                4
            )

        })


    except Exception as e:

        return jsonify({

            "status":
            "error",

            "message":
            str(e)

        }), 500


# ============================================================
# DATASET STATISTICS API
# ============================================================

@app.route(
    "/api/stats",
    methods=["GET"]
)
def statistics():

    if df.empty:

        return jsonify({

            "status":
            "error",

            "message":
            "Dataset not loaded."

        }), 500


    # ========================================================
    # TOTAL
    # ========================================================

    total_messages = len(df)


    # ========================================================
    # SPAM COUNT
    # ========================================================

    spam_count = (
        df["v1"] == "spam"
    ).sum()


    # ========================================================
    # HAM COUNT
    # ========================================================

    ham_count = (
        df["v1"] == "ham"
    ).sum()


    # ========================================================
    # PERCENTAGES
    # ========================================================

    spam_percentage = (
        spam_count /
        total_messages
    ) * 100


    ham_percentage = (
        ham_count /
        total_messages
    ) * 100


    # ========================================================
    # RETURN
    # ========================================================

    return jsonify({

        "status":
        "success",

        "total_messages":
        int(total_messages),

        "spam_messages":
        int(spam_count),

        "ham_messages":
        int(ham_count),

        "spam_percentage":
        round(
            float(spam_percentage),
            2
        ),

        "ham_percentage":
        round(
            float(ham_percentage),
            2
        )

    })


# ============================================================
# EXAMPLES API
# ============================================================

@app.route(
    "/api/examples",
    methods=["GET"]
)
def examples():

    if df.empty:

        return jsonify({

            "status":
            "error",

            "message":
            "Dataset not loaded."

        }), 500


    # ========================================================
    # GET TYPE
    # ========================================================

    message_type = request.args.get(
        "type",
        "spam"
    ).lower()


    # ========================================================
    # GET LIMIT
    # ========================================================

    try:

        limit = int(
            request.args.get(
                "limit",
                5
            )
        )

    except:

        limit = 5


    # ========================================================
    # LIMIT PROTECTION
    # ========================================================

    if limit < 1:

        limit = 5

    if limit > 100:

        limit = 100


    # ========================================================
    # FILTER DATA
    # ========================================================

    if message_type == "spam":

        filtered = df[
            df["v1"] == "spam"
        ]

    elif message_type == "ham":

        filtered = df[
            df["v1"] == "ham"
        ]

    else:

        return jsonify({

            "status":
            "error",

            "message":
            "type must be spam or ham."

        }), 400


    # ========================================================
    # CREATE RESULTS
    # ========================================================

    results = []


    for _, row in filtered.head(
        limit
    ).iterrows():

        results.append({

            "v1":
            row["v1"],

            "v2":
            row["v2"],

            "label":
            "spam"
            if row["label"] == 1
            else "ham",

            "message":
            row["message"],

            "clean_text":
            row["clean_text"]

        })


    # ========================================================
    # RETURN
    # ========================================================

    return jsonify({

        "status":
        "success",

        "count":
        len(results),

        "examples":
        results

    })


# ============================================================
# DATASET SEARCH API
# GET + POST
# ============================================================

@app.route(
    "/api/search",
    methods=["GET", "POST"]
)
def search_dataset():

    try:

        # ====================================================
        # CHECK SEARCH MATRIX
        # ====================================================

        if dataset_matrix is None:

            return jsonify({

                "status":
                "error",

                "message":
                "Dataset search is unavailable."

            }), 500


        # ====================================================
        # GET QUERY
        # GET  -> ?query=
        # POST -> JSON
        # ====================================================

        if request.method == "GET":

            query = request.args.get(
                "query",
                ""
            )

        else:

            data = request.get_json(
                silent=True
            )

            if not data:

                return jsonify({

                    "status":
                    "error",

                    "message":
                    "JSON request body is required."

                }), 400

            query = data.get(
                "query",
                ""
            )


        # ====================================================
        # MAKE SURE QUERY IS STRING
        # ====================================================

        query = str(query)


        if not query.strip():

            return jsonify({

                "status":
                "error",

                "message":
                "Search query is required."

            }), 400


        # ====================================================
        # CLEAN QUERY
        # ====================================================

        cleaned_query = clean_text(
            query
        )


        # ====================================================
        # VECTORIZE QUERY
        # ====================================================

        query_vector = (
            tfidf_vectorizer.transform(
                [cleaned_query]
            )
        )


        # ====================================================
        # COSINE SIMILARITY
        # ====================================================

        similarities = (
            cosine_similarity(
                query_vector,
                dataset_matrix
            )[0]
        )


        # ====================================================
        # TOP 10
        # ====================================================

        top_indices = (
            similarities.argsort()[
                -10:
            ][::-1]
        )


        # ====================================================
        # RESULTS
        # ====================================================

        results = []


        for index in top_indices:

            score = similarities[index]


            if score > 0:

                results.append({

                    "v1":
                    df.iloc[index]["v1"],

                    "v2":
                    df.iloc[index]["v2"],

                    "label":
                    "spam"
                    if df.iloc[index]["label"] == 1
                    else "ham",

                    "message":
                    df.iloc[index]["message"],

                    "clean_text":
                    df.iloc[index]["clean_text"],

                    "similarity":
                    round(
                        float(score),
                        4
                    )

                })


        # ====================================================
        # RETURN
        # ====================================================

        return jsonify({

            "status":
            "success",

            "query":
            query,

            "results":
            results

        })


    except Exception as e:

        return jsonify({

            "status":
            "error",

            "message":
            str(e)

        }), 500


# ============================================================
# CHATBOT API
# GET + POST
# ============================================================

@app.route(
    "/api/chat",
    methods=["GET", "POST"]
)
def chatbot():

    try:

        # ====================================================
        # GET QUESTION
        # GET  -> ?message=
        # POST -> JSON
        # ====================================================

        if request.method == "GET":

            question = request.args.get(
                "message",
                ""
            )

        else:

            data = request.get_json(
                silent=True
            )

            if not data:

                return jsonify({

                    "status":
                    "error",

                    "message":
                    "JSON request body is required."

                }), 400

            question = data.get(
                "message",
                ""
            )


        # ====================================================
        # MAKE SURE QUESTION IS STRING
        # ====================================================

        question = str(question)


        if not question.strip():

            return jsonify({

                "status":
                "error",

                "message":
                "Chat message is required."

            }), 400


        # ====================================================
        # CHECK DATASET
        # ====================================================

        if dataset_matrix is None:

            return jsonify({

                "status":
                "error",

                "message":
                "Dataset search is unavailable."

            }), 500


        # ====================================================
        # SEARCH DATASET
        # ====================================================

        cleaned_question = clean_text(
            question
        )


        query_vector = (
            tfidf_vectorizer.transform(
                [cleaned_question]
            )
        )


        similarities = (
            cosine_similarity(
                query_vector,
                dataset_matrix
            )[0]
        )


        # ====================================================
        # TOP 8 RECORDS
        # ====================================================

        top_indices = (
            similarities.argsort()[
                -8:
            ][::-1]
        )


        # ====================================================
        # RETRIEVED RESULTS
        # ====================================================

        retrieved_results = []


        for index in top_indices:

            score = similarities[index]


            if score > 0:

                retrieved_results.append({

                    "v1":
                    df.iloc[index]["v1"],

                    "v2":
                    df.iloc[index]["v2"],

                    "label":
                    "spam"
                    if df.iloc[index]["label"] == 1
                    else "ham",

                    "message":
                    df.iloc[index]["message"],

                    "similarity":
                    round(
                        float(score),
                        4
                    )

                })


        # ====================================================
        # DATASET STATISTICS
        # ====================================================

        total = len(df)


        spam_count = (
            df["v1"] == "spam"
        ).sum()


        ham_count = (
            df["v1"] == "ham"
        ).sum()


        spam_percentage = (
            spam_count /
            total
        ) * 100


        ham_percentage = (
            ham_count /
            total
        ) * 100


        # ====================================================
        # OPENAI CLIENT CHECK
        # ====================================================

        if client is None:

            return jsonify({

                "status":
                "error",

                "message":
                "OpenAI API key is not configured."

            }), 500


        # ====================================================
        # BUILD DATASET CONTEXT
        # ====================================================

        context = ""


        for i, result in enumerate(
            retrieved_results,
            start=1
        ):

            context += f"""

Record {i}

v1 Label:
{result['v1']}

v2 Message:
{result['v2']}

ML Label:
{result['label']}

Similarity:
{result['similarity']}

"""


        # ====================================================
        # STATISTICS CONTEXT
        # ====================================================

        statistics_context = f"""

Total messages:
{total}

Spam messages:
{spam_count}

Ham messages:
{ham_count}

Spam percentage:
{spam_percentage:.2f}%

Ham percentage:
{ham_percentage:.2f}%

"""


        # ====================================================
        # SYSTEM INSTRUCTIONS
        # ====================================================

        instructions = """

You are a Spam Email Dataset Chatbot.

You MUST answer using ONLY the dataset information
provided in the input.

Rules:

1. Do not use outside knowledge.

2. Do not invent information.

3. Do not guess.

4. If the dataset does not contain enough information,
say:

"I could not find enough information in the dataset
to answer that."

5. When giving examples, use only messages retrieved
from the dataset.

6. When discussing spam or ham, respect the v1 labels
present in the dataset.

7. For statistics, use the supplied dataset statistics.

8. Keep the answer concise and clear.

9. Never claim that a message exists in the dataset
unless it is present in the retrieved records.

10. v1 is the original dataset label.

11. v2 is the original dataset message.

"""


        # ====================================================
        # USER PROMPT
        # ====================================================

        prompt = f"""

User Question:

{question}


Dataset Statistics:

{statistics_context}


Retrieved Dataset Records:

{context}


Answer the question using ONLY the dataset
statistics and retrieved records.

"""


        # ====================================================
        # OPENAI RESPONSE
        # ====================================================

        response = client.responses.create(

            model="gpt-5.6-luna",

            instructions=instructions,

            input=prompt

        )


        answer = response.output_text


        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return jsonify({

            "status":
            "success",

            "question":
            question,

            "answer":
            answer,

            "retrieved_records":
            retrieved_results

        })


    except Exception as e:

        return jsonify({

            "status":
            "error",

            "message":
            str(e)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=" * 60
    )

    print(
        "SPAM EMAIL DETECTION BACKEND"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "Server: http://127.0.0.1:5000"
    )

    print()

    print(
        "Available APIs:"
    )

    print(
        "GET  /"
    )

    print(
        "GET  /api/spam"
    )

    print(
        "GET  /api/stats"
    )

    print(
        "GET  /api/examples"
    )

    print(
        "GET  /api/predict?message=..."
    )

    print(
        "POST /api/predict"
    )

    print(
        "GET  /api/search?query=..."
    )

    print(
        "POST /api/search"
    )

    print(
        "GET  /api/chat?message=..."
    )

    print(
        "POST /api/chat"
    )

    print()


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )