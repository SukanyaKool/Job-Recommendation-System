import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import kagglehub
import google.generativeai as genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lime.lime_text import LimeTextExplainer

# =====================================================

# PAGE CONFIG

# =====================================================

st.set_page_config(
page_title=" Job Recommendation System",
page_icon="💼",
layout="wide"
)

st.title("💼 Job Recommendation System")
st.caption("TF-IDF + Weighted Ranking + XAI + LIME + RAG")

# =====================================================

# GEMINI CONFIG

# =====================================================

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(
api_key=GOOGLE_API_KEY
)

model_gemini = genai.GenerativeModel(
"models/gemini-2.5-flash"
)

# =====================================================

# LOAD DATASET

# =====================================================

@st.cache_resource
def load_dataset():

path = kagglehub.dataset_download(
    "asaniczka/1-3m-linkedin-jobs-and-skills-2024"
)

jobs = pd.read_csv(
    os.path.join(
        path,
        "linkedin_job_postings.csv"
    )
)

jobs.fillna("", inplace=True)

return jobs

jobs = load_dataset()

# =====================================================

# PREPARE DATA

# =====================================================

def prepare_data(df):

for col in [
    "company_name",
    "company",
    "companyName"
]:

    if col in df.columns:

        df["company_name"] = df[col]
        break

for col in [
    "job_location",
    "location_name",
    "formatted_location"
]:

    if col in df.columns:

        df["job_location"] = df[col]
        break

for col in [
    "description",
    "job_description"
]:

    if col in df.columns:

        df["description"] = df[col]
        break

df["combined"] = (

    df["job_title"].astype(str)

    + " "

    + df["company_name"].astype(str)

    + " "

    + df["job_location"].astype(str)

    + " "

    + df["description"].astype(str)

)

return df

jobs = prepare_data(jobs)

# =====================================================

# TF-IDF

# =====================================================

@st.cache_resource
def build_tfidf():

tfidf = TfidfVectorizer(

    stop_words="english",

    max_features=30000,

    ngram_range=(1,2),

    min_df=2

)

tfidf_matrix = tfidf.fit_transform(
    jobs["combined"]
)

return tfidf, tfidf_matrix

tfidf, tfidf_matrix = build_tfidf()

# =====================================================

# RETRIEVAL

# =====================================================

def retrieve_jobs(
user_input,
top_n
):


user_vector = tfidf.transform(
    [user_input]
)

similarity_scores = cosine_similarity(
    user_vector,
    tfidf_matrix
)[0]

top_indices = np.argpartition(
    similarity_scores,
    -top_n
)[-top_n:]

top_indices = top_indices[
    np.argsort(
        similarity_scores[top_indices]
    )[::-1]
]

return (

    jobs.iloc[top_indices],

    similarity_scores[top_indices]

)


# =====================================================

# XAI

# =====================================================

def explain_skills(
user_input,
job_text
):


user_skills = [

    skill.strip().lower()

    for skill in re.split(
        r",|;",
        user_input
    )

    if skill.strip()

]

job_text = str(
    job_text
).lower()

matched = []

for skill in user_skills:

    pattern = r"\b" + re.escape(skill) + r"\b"

    if re.search(
        pattern,
        job_text
    ):

        matched.append(
            skill
        )

return matched


def explain_keywords(
user_input
):


vector = tfidf.transform(
    [user_input]
)

feature_names = np.array(
    tfidf.get_feature_names_out()
)

tfidf_scores = vector.toarray()[0]

top_indices = tfidf_scores.argsort()[-5:][::-1]

return list(
    feature_names[top_indices]
)


# =====================================================

# EXPERIENCE

# =====================================================

def get_job_level(job_title):


title = str(job_title).lower()

if any(x in title for x in [
    "intern",
    "junior",
    "associate"
]):
    return 1

elif any(x in title for x in [
    "senior",
    "sr"
]):
    return 3

elif any(x in title for x in [
    "lead",
    "manager",
    "architect",
    "director"
]):
    return 5

return 2


def get_experience_match_score(
user_exp,
job_title
):


job_level = get_job_level(
    job_title
)

if user_exp <= 2:
    user_level = 1

elif user_exp <= 5:
    user_level = 2

elif user_exp <= 8:
    user_level = 3

else:
    user_level = 5

diff = abs(
    user_level - job_level
)

return max(
    0,
    100 - diff * 25
)


# =====================================================

# DEMAND SCORE

# =====================================================

future_skill_score = {


"python":90,
"machine learning":92,
"deep learning":94,
"artificial intelligence":95,
"aws":88,
"cloud computing":90,
"azure":86,
"cybersecurity":94,
"splunk":82,
"devops":91,
"llm":98,
"genai":99


}

def demand_score(user_input):


text = user_input.lower()

scores = []

for skill, value in future_skill_score.items():

    if skill in text:

        scores.append(value)

return (
    sum(scores) / len(scores)
    if scores
    else 50
)


def calculate_weighted_score(
similarity,
exp_score,
future_score
):


return (

    0.50 * similarity

    +

    0.25 * exp_score

    +

    0.25 * future_score

)


# =====================================================

# LIME

# =====================================================

explainer = LimeTextExplainer(
class_names=[
"Not Relevant",
"Relevant"
]
)

sample_matrix = tfidf_matrix[:5000]

def lime_predict(texts):


similarities = []

for text in texts:

    vector = tfidf.transform(
        [text]
    )

    similarity = cosine_similarity(
        vector,
        sample_matrix
    ).max()

    similarities.append(
        [
            1-similarity,
            similarity
        ]
    )

return np.array(
    similarities
)


# =====================================================

# RAG

# =====================================================

def generate_explanation(
user_input,
experience,
weighted_score,
job_row
):


prompt = f"""


User Skills: {user_input}

Experience: {experience}

Job Title: {job_row['job_title']}

Company: {job_row['company_name']}

Weighted Score: {weighted_score:.2f}

Explain why this job is suitable in 5 lines.
"""

```
response = model_gemini.generate_content(
    prompt
)

return response.text


# =====================================================

# SIDEBAR

# =====================================================

st.sidebar.header(
"Search Parameters"
)

skills = st.sidebar.text_area(
"Skills"
)

experience = st.sidebar.number_input(
"Years of Experience",
0,
30,
2
)

top_n = st.sidebar.slider(
"Recommendations",
1,
20,
5
)

# =====================================================

# BUTTON

# =====================================================

if st.sidebar.button(
"Get Recommendations"
):


results, scores = retrieve_jobs(
    skills,
    top_n
)

recommendations = []

for i, (_, row) in enumerate(
    results.iterrows()
):

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

        "weighted_score": weighted_score

    })

recommendations.sort(

    key=lambda x:
    x["weighted_score"],

    reverse=True

)

for rec in recommendations:

    row = rec["row"]

    st.markdown(
        f"""


### {row['job_title']}

**Company:** {row['company_name']}

**Location:** {row['job_location']}

**Weighted Score:** {rec['weighted_score']:.2f}/100
"""
)


    with st.expander(
        "Explain Recommendation"
    ):

        st.write(
            "Matching Skills:",
            explain_skills(
                skills,
                row["combined"]
            )
        )

        st.write(
            "Keywords:",
            explain_keywords(
                skills
            )
        )

top_job = recommendations[0]["row"]

rag_output = generate_explanation(

    skills,

    experience,

    recommendations[0]["weighted_score"],

    top_job

)

st.subheader(
    "AI Career Insight"
)

st.success(
    rag_output
)
