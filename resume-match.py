import os
import streamlit as st
import fitz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load API key
st.sidebar.header("⚡ OpenAI API Key")
st.sidebar.markdown(
    "Get an API key [here](https://platform.openai.com/account/api-keys) if you don't have one.")
api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key", type="password")

if not user_api_key:
    st.warning("Please enter your OpenAI API key to use the app.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Streamlit UI setup
st.set_page_config(page_title="📝 AI Resume Matcher", layout="wide")
st.title("📝 AI Resume vs Job Description Matcher")
st.markdown("""
Upload your **resume (PDF)** and paste the **job description** below.  
The AI will summarize the job requirements, highlight your resume's skills/experience, and provide recommendations.
""")

# Extract text from PDF
@st.cache_data
def extract_text(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

# Summarize job description with langchain summary chain
def summarize_job(job_text):
    template = """
Summarize the following job description in 3-5 bullet points highlighting required skills, experience, and qualifications.

Job Description:
{job_text}
"""
    prompt = PromptTemplate(input_variables=["job_text"], template=template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=api_key)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(job_text=job_text)

# Summarize resume
def summarize_resume(resume_text):
    template = """
Summarize the following resume in 3-5 bullet points highlighting the candidate's skills, experience, and qualifications.

Resume:
{resume_text}
"""
    prompt = PromptTemplate(input_variables=["resume_text"], template=template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=api_key)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(resume_text=resume_text)

# Generate example tailored resume
def generate_ex(resume_text, job_text):
    prompt = f"""
You are a professional career coach and resume expert. 
Generate a single-page, clean, professional resume based on the candidate's existing resume 
and tailored to the following job description. 

Guidelines:
- Keep it to 1 page maximum.
- Use clear sections: Contact, Summary, Skills, Experience, Education.
- Tailor skills and experience to the job description.
- Use bullet points, active verbs, and concise formatting.
- Avoid explanations only output the resume text.
- Ensure readability and professional appearance.
- make it such that it is ATS friendly and would match perfectly with the job description.
- If you have to make up experience or skills to further tailor it, do so vaguely without name dropping anything (ex: did x at y company accomplishing z).
- remove any irrelevant or outdated information.
- Make it so that when passed through an AI resume filtering system used by real companies, it would get a 90+% match with the job description.

Candidate Resume:
{resume_text}

Job Description:
{job_text}
"""
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "You are an expert resume writer."},
            {"role": "user", "content": prompt}
        ],
        reasoning = {"effort": "medium"},
        text={"verbosity": "medium"},
    )
    return response.output_text.strip()

# Generate short recommendation summary
def generate_rec(resume_text, job_text):
    prompt = f"""
A candidate has the following resume:

{resume_text[:1000]}

The job description is:

{job_text[:1000]}

Write a simple, actionable summary highlighting strengths and suggestions for improving alignment to this job.
"""
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=[
            {"role": "system", "content": "You are a helpful career coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.output[0].content[0].text.strip()

# Find learning resources
def find_jobs(resume_text):
    prompt = f"""
A candidate has the following resume:
{resume_text}
Find 3 relevant job opportunities online that match this resume. 
If theyre still in school, include internships. If done with school, do not include internships. 
If they are a senior level candidate, include senior roles.
They should be recent postings from reputable sources. 
Format as a numbered list with links if possible.
"""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        tools=[{"type": "web_search"}],
    )
    return response.output_text.strip()

# File upload & job description input
resume_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])
job_description_input = st.text_area("📝 Paste Job Description Text", height=250)

# Run if both inputs are present
if resume_file and job_description_input.strip():
    resume_text = extract_text(resume_file)
    job_text = job_description_input.strip()

    # Overall similarity
    with st.spinner("Calculating overall resume-job similarity..."):
        embed_resume = np.array(client.embeddings.create(
            model="text-embedding-3-small", input=resume_text).data[0].embedding)
        embed_job = np.array(client.embeddings.create(
            model="text-embedding-3-small", input=job_text).data[0].embedding)
        similarity = cosine_similarity([embed_resume], [embed_job])[0][0]
        similarity_pct = similarity * 100

    st.subheader("📊 Overall Match Score")
    st.write(f"**{similarity_pct:.2f}%** match with job description")
    st.progress(int(similarity_pct))
    if similarity_pct > 65:
        st.success("🎯 Excellent match! Very strong alignment.")
    elif similarity_pct > 45:
        st.info("✅ Good match! Your resume fits the job fairly well.")
    elif similarity_pct > 25:
        st.warning("⚠️ Partial match. Consider tailoring your resume more.")
    else:
        st.error("❌ Weak match. Consider major edits.")

    # Summaries & recommendations
    with st.spinner("Summarizing job description..."):
        job_summary = summarize_job(job_text)
    with st.spinner("Summarizing your resume..."):
        resume_summary = summarize_resume(resume_text)
    with st.spinner("Generating AI recommendations..."):
        recommendations = generate_rec(resume_text, job_text)

    # Display comparison
    st.subheader("📊 Job vs Resume Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💼 Job Requirements")
        st.markdown(f"📝 {job_summary}")
    with col2:
        st.markdown("### 📄 Your Resume")
        st.markdown(f"✅ {resume_summary}")

    # AI recommendations
    st.subheader("📝 AI Recommendations")
    st.info(f"💡 {recommendations}")

    # Example tailored resume
    if st.button("🖋️ Generate Example Tailored Resume"):
        with st.spinner("Generating a tailored example resume..."):
            example_resume = generate_ex(resume_text, job_text)
            st.subheader("📄 Example Tailored Resume")
            st.text_area("Sample Resume", example_resume, height=400)
            st.download_button("📥 Download Example Resume", example_resume, "example_resume.txt")

    # Useful resources
    if st.button("🌐 Find Jobs that Match Resume"):
        with st.spinner("Searching for jobs..."):
            resources = find_jobs(resume_text)
            st.subheader("🔗 Job Oppurtunities Based on Resume")
            st.markdown(resources)

else:
    st.info("📥 Upload your resume and paste a job description to get started.")
