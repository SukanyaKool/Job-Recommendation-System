import os
import re

import numpy as np
import pandas as pd
import streamlit as st

import kagglehub

from google import genai
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Job Recommendation System", page_icon="💼", layout="wide")


def load_css():
    """Load and apply CSS styling."""
    with open("styles.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()
st.title("💼 Job Recommendation System")
st.markdown("""
# 🎯 Career Navigator

### Intelligent Job Recommendation & Career Guidance System

Leverage Artificial Intelligence to discover opportunities aligned
with your skills, experience, and future industry trends.
""")


# =====================================================
# GEMINI CONFIG
# =====================================================
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GEMINI_AVAILABLE = False

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")
        GEMINI_AVAILABLE = True
        st.write("Gemini Available:", GEMINI_AVAILABLE)
        st.write("Key Loaded:", GOOGLE_API_KEY is not None)
    except Exception:
        GEMINI_AVAILABLE = False


# =====================================================
# LOAD DATASET
# =====================================================
@st.cache_resource
def load_dataset():
    path = kagglehub.dataset_download("asaniczka/1-3m-linkedin-jobs-and-skills-2024")

    jobs = pd.read_csv(os.path.join(path, "linkedin_job_postings.csv"))
    jobs.fillna("", inplace=True)

    return jobs


jobs = load_dataset()


# =====================================================
# PREPARE DATA
# =====================================================
def prepare_data(df):
    """Prepare and clean dataset by mapping column names."""
    # Company column mapping
    company_cols = ["company_name", "company", "companyName"]
    found = False

    for col in company_cols:
        if col in df.columns:
            df["company_name"] = df[col]
            found = True
            break

    if not found:
        df["company_name"] = "Unknown Company"

    # Location column mapping
    if "job_location" not in df.columns:
        df["job_location"] = "Unknown Location"

    # Job title mapping
    if "job_title" not in df.columns:
        df["job_title"] = "Unknown Job"

    # Description mapping
    if "description" in df.columns:
        desc_col = "description"

    elif "job_description" in df.columns:
        desc_col = "job_description"

    else:
        df["description"] = ""
        desc_col = "description"

    df["combined"] = (
        df["job_title"].astype(str)
        + " "
        + df["company_name"].astype(str)
        + " "
        + df["job_location"].astype(str)
        + " "
        + df[desc_col].astype(str)
    )

    return df

# If dataset has expected columns, prepare combined text
jobs = prepare_data(jobs)


# =====================================================
# TF-IDF
# =====================================================
@st.cache_resource
def build_tfidf():
    tfidf = TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(jobs["combined"])
    return tfidf, tfidf_matrix


tfidf, tfidf_matrix = build_tfidf()


# =====================================================
# RETRIEVAL
# =====================================================
def retrieve_jobs(user_input, top_n):
    user_vector = tfidf.transform([user_input])

    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)[0]

    top_indices = np.argpartition(similarity_scores, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(similarity_scores[top_indices])[::-1]]

    return jobs.iloc[top_indices], similarity_scores[top_indices]


# =====================================================
# XAI
# =====================================================
def explain_skills(user_input, job_text):
    """Extract skills from user input that match job text."""
    user_skills = [
        skill.strip().lower()
        for skill in re.split(r",|;|\s+", user_input)
        if skill.strip()
    ]

    job_text = str(job_text).lower()

    matched = [
        skill
        for skill in user_skills
        if skill in job_text
    ]

    return matched


def explain_keywords(user_input):
    """Extract keywords from user input."""
    keywords = [
        keyword.strip().lower()
        for keyword in re.split(r",|;|\s+", user_input)
        if keyword.strip()
    ]
    return keywords


# =====================================================
# EXPERIENCE
# =====================================================
def get_job_level(job_title):
    title = str(job_title).lower()

    if any(x in title for x in ["intern", "junior", "associate"]):
        return 1
    elif any(x in title for x in ["senior", "sr"]):
        return 3
    elif any(x in title for x in ["lead", "manager", "architect", "director"]):
        return 5

    return 2


def get_experience_match_score(user_exp, job_title):
    job_level = get_job_level(job_title)

    if user_exp <= 2:
        user_level = 1
    elif user_exp <= 5:
        user_level = 2
    elif user_exp <= 8:
        user_level = 3
    else:
        user_level = 5

    diff = abs(user_level - job_level)
    return max(0, 100 - diff * 25)


# =====================================================
# DEMAND SCORE
# =====================================================
future_skill_score = {
    "python": 90,
    "machine learning": 92,
    "deep learning": 94,
    "artificial intelligence": 95,
    "aws": 88,
    "cloud computing": 90,
    "azure": 86,
    "cybersecurity": 94,
    "splunk": 82,
    "devops": 91,
    "llm": 98,
    "genai": 99,
}


def demand_score(user_input):
    text = user_input.lower()
    scores = []

    for skill, value in future_skill_score.items():
        if skill in text:
            scores.append(value)

    return sum(scores) / len(scores) if scores else 50


def calculate_weighted_score(similarity, exp_score, future_score):
    return 0.50 * similarity + 0.25 * exp_score + 0.25 * future_score


# =====================================================
# LIME
# =====================================================
explainer = LimeTextExplainer(class_names=["Not Relevant", "Relevant"])

sample_matrix = tfidf_matrix[:5000]


def lime_predict(texts):
    similarities = []
    for text in texts:
        vector = tfidf.transform([text])
        similarity = cosine_similarity(vector, sample_matrix).max()
        similarities.append([1 - similarity, similarity])

    return np.array(similarities)


# =====================================================
# RAG
# =====================================================
@st.cache_data(ttl=3600)
def generate_explanation(
    user_input,
    experience,
    weighted_score,
    job_row
):

    if not GEMINI_AVAILABLE:

        return (
            f"This role aligns with your skills in {user_input}. "
            f"With {experience} years of experience, the position "
            f"matches your profile and has strong growth potential."
        )

    prompt = f"""
    User Skills: {user_input}
    Experience: {experience}
    Job Title: {job_row['job_title']}
    Company: {job_row['company_name']}
    Weighted Score: {weighted_score:.2f}

    Explain why this job is suitable in 5 lines.
    """

    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Career Insight unavailable: {e}"



# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Search Parameters")
skills = st.sidebar.text_area("Skills")
experience = st.sidebar.number_input("Years of Experience", 0, 30, 2)
top_n = st.sidebar.slider("Recommendations", 1, 20, 5)


# =====================================================
# BUTTON
# =====================================================
if st.sidebar.button("Get Recommendations"):
    results, scores = retrieve_jobs(skills, top_n)
    st.write(f"Jobs Retrieved: {len(results)}")

    recommendations = []

    for i, (_, row) in enumerate(results.iterrows()):

        similarity_score = scores[i] * 100

        exp_score = get_experience_match_score(
            experience,
            row["job_title"]
        )

        future_score = demand_score(
            skills
        )

        weighted_score = calculate_weighted_score(
            similarity_score,
            exp_score,
            future_score
        )

        recommendations.append({
            "row": row,
            "weighted_score": weighted_score,
            "similarity_score": similarity_score,
            "experience_score": exp_score,
            "future_score": future_score
        })

    recommendations.sort(
        key=lambda x: x["weighted_score"],
        reverse=True
    )

    # DISPLAY ALL RECOMMENDATIONS
    for rec in recommendations:

        row = rec["row"]

        st.markdown(
            f"""
<div class="job-card">

<div class="job-title">
💼 {row['job_title']}
</div>

<div>
🏢 <b>Company:</b> {row['company_name']}
</div>

<div>
📍 <b>Location:</b> {row['job_location']}
</div>

<br>

<div class="job-score">
⭐ Match Score: {rec['weighted_score']:.2f}/100
</div>

</div>
""",
            unsafe_allow_html=True
        )

        with st.expander("🔍 Recommendation Insights"):

            matched_skills = explain_skills(
                skills,
                row["combined"]
            )

            keywords = explain_keywords(
                skills
            )

            st.write(
                "🎯 Matching Skills:",
                matched_skills
            )

            st.write(
                "🔑 Important Keywords:",
                keywords
            )

            st.write(
                f"📊 Similarity Score: {rec['similarity_score']:.2f}"
            )

            st.write(
                f"👨‍💼 Experience Match: {rec['experience_score']:.2f}"
            )

            st.write(
                f"📈 Future Demand Score: {rec['future_score']:.2f}"
            )

    # TOP RECOMMENDATION FOR RAG
    if len(recommendations) > 0:
        top_job = recommendations[0]["row"]
        try:
            rag_output = generate_explanation(
                skills,
                experience,
                recommendations[0]["weighted_score"],
                top_job,
            )
            st.markdown("## 🚀 AI Career Advisor")
            st.info(rag_output)
        except Exception:
            st.warning("AI Career Advisor temporarily unavailable.")
