import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define knowledge base
job_listings = [
    {"id": 1, "title": "Software Engineer", "company": "TechCorp", "location": "Remote", 
     "description": "Software engineering role focused on frontend development.", "tags": ["tech", "coding", "frontend"]},
    {"id": 2, "title": "Data Analyst", "company": "DataInsights", "location": "Bangalore", 
     "description": "Analyze business data and create dashboards.", "tags": ["data", "analytics", "sql"]},
    {"id": 3, "title": "Product Manager", "company": "InnovateCo", "location": "Hybrid", 
     "description": "Lead product development initiatives for women-focused tech products.", "tags": ["product", "leadership"]},
    {"id": 4, "title": "Marketing Specialist", "company": "GrowthMedia", "location": "Mumbai", 
     "description": "Create marketing campaigns for women empowerment initiatives.", "tags": ["marketing", "social media"]}
]

events = [
    {"id": 1, "title": "Women in Tech Conference", "date": "2025-06-15", "location": "Virtual", 
     "description": "Annual conference showcasing women leaders in technology."},
    {"id": 2, "title": "Resume Building Workshop", "date": "2025-06-20", "location": "Bangalore", 
     "description": "Learn how to create effective resumes for tech industry roles."},
    {"id": 3, "title": "Networking Mixer", "date": "2025-06-25", "location": "Delhi", 
     "description": "Connect with women professionals across industries."}
]

mentorship_programs = [
    {"id": 1, "title": "Tech Leadership Program", "duration": "3 months", 
     "description": "Mentorship for women aspiring to leadership roles in tech."},
    {"id": 2, "title": "Career Comeback Program", "duration": "2 months", 
     "description": "Support for women returning to work after a career break."},
    {"id": 3, "title": "Entrepreneurship Guidance", "duration": "6 months", 
     "description": "Mentorship for women starting their own businesses."}
]

faqs = [
    {"question": "How do I update my profile?", 
     "answer": "Log in to your JobsForHer account, click on your profile picture, select 'Edit Profile', and update your information."},
    {"question": "How can I apply for jobs?", 
     "answer": "Browse job listings, click on a job you're interested in, and click the 'Apply' button. You'll need to complete your profile first."},
    {"question": "What is the mentorship program?", 
     "answer": "Our mentorship programs connect you with experienced professionals who can guide your career growth in specific areas."},
    {"question": "How do I sign up for events?", 
     "answer": "Browse our events section, select an event, and click 'Register'. You'll receive a confirmation email with details."}
]

company_info = """
JobsForHer Foundation is dedicated to empowering women in their professional careers through job opportunities, 
mentorship, networking events, and skill development. Our platform connects women with employers committed to 
diversity and inclusion, providing resources to help women advance in their careers or return to the workforce 
after a break. We offer job listings across various industries, mentorship programs, career advice, and 
community events designed specifically for women professionals.
"""

# Create text chunks for embedding
all_texts = []
all_metadata = []

# Process job listings
for job in job_listings:
    text = f"Job Title: {job['title']}\nCompany: {job['company']}\nLocation: {job['location']}\nDescription: {job['description']}"
    all_texts.append(text)
    all_metadata.append({"type": "job", "id": job["id"]})

# Process events
for event in events:
    text = f"Event: {event['title']}\nDate: {event['date']}\nLocation: {event['location']}\nDescription: {event['description']}"
    all_texts.append(text)
    all_metadata.append({"type": "event", "id": event["id"]})

# Process mentorship programs
for program in mentorship_programs:
    text = f"Program: {program['title']}\nDuration: {program['duration']}\nDescription: {program['description']}"
    all_texts.append(text)
    all_metadata.append({"type": "mentorship", "id": program["id"]})

# Process FAQs
for faq in faqs:
    text = f"Q: {faq['question']}\nA: {faq['answer']}"
    all_texts.append(text)
    all_metadata.append({"type": "faq", "id": faqs.index(faq)})

# Add company info
all_texts.append(company_info)
all_metadata.append({"type": "info", "id": 0})

# Build the RAG system
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # This model works well on HF Spaces

# Create embeddings
print("Creating embeddings...")
embeddings = embedding_model.encode(all_texts)

# Build FAISS index for efficient similarity search
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# Function for direct answers to common questions
def get_direct_answer(query):
    if "update profile" in query.lower() or "edit profile" in query.lower():
        return "To update your profile, log in to your JobsForHer account, click on your profile picture in the top right, select 'Edit Profile', and update your information as needed."
    
    if "apply for job" in query.lower() or "job application" in query.lower():
        return "To apply for jobs, browse our job listings, click on a job you're interested in, and click the 'Apply' button. Make sure your profile is complete with your resume and relevant experience."
    
    if "sign up" in query.lower() or "register" in query.lower() or "create account" in query.lower():
        return "To create a JobsForHer account, visit our homepage and click 'Sign Up'. Enter your email address and create a password, then complete your profile with your professional details."
    
    return None

# Function to generate responses
def generate_response(query, chat_history=None):
    if chat_history is None:
        chat_history = []
    
    # First check if we have a direct answer
    direct_answer = get_direct_answer(query)
    if direct_answer:
        return direct_answer
    
    # Create embedding for the query
    query_embedding = embedding_model.encode([query])[0]
    
    # Search for similar content in our knowledge base
    k = 3  # Number of results to retrieve
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    
    # Retrieve the most similar texts
    retrieved_texts = [all_texts[idx] for idx in indices[0]]
    
    # Prepare context
    context = "\n\n".join(retrieved_texts)
    
    # For Hugging Face Spaces, we'll use a simpler approach without the LLM
    # This will ensure it works without API keys
    
    # Check what type of information was retrieved
    response_parts = []
    
    # Check if any job listings were retrieved
    jobs_found = [meta for i, meta in enumerate(all_metadata) if i in indices[0] and meta["type"] == "job"]
    if jobs_found:
        response_parts.append("I found some job opportunities that might interest you:")
        for job_meta in jobs_found:
            job = job_listings[job_meta["id"]-1]
            response_parts.append(f"- {job['title']} at {job['company']} ({job['location']}): {job['description']}")
    
    # Check if any events were retrieved
    events_found = [meta for i, meta in enumerate(all_metadata) if i in indices[0] and meta["type"] == "event"]
    if events_found:
        response_parts.append("Here are some upcoming events:")
        for event_meta in events_found:
            event = events[event_meta["id"]-1]
            response_parts.append(f"- {event['title']} on {event['date']} in {event['location']}: {event['description']}")
    
    # Check if any mentorship programs were retrieved
    programs_found = [meta for i, meta in enumerate(all_metadata) if i in indices[0] and meta["type"] == "mentorship"]
    if programs_found:
        response_parts.append("Here are mentorship programs you might be interested in:")
        for program_meta in programs_found:
            program = mentorship_programs[program_meta["id"]-1]
            response_parts.append(f"- {program['title']} ({program['duration']}): {program['description']}")
    
    # Check if any FAQs were retrieved
    faqs_found = [meta for i, meta in enumerate(all_metadata) if i in indices[0] and meta["type"] == "faq"]
    if faqs_found:
        for faq_meta in faqs_found:
            faq = faqs[faq_meta["id"]]
            response_parts.append(f"Q: {faq['question']}\nA: {faq['answer']}")
    
    # Check if company info was retrieved
    if any(meta["type"] == "info" for i, meta in enumerate(all_metadata) if i in indices[0]):
        response_parts.append("About JobsForHer Foundation:")
        response_parts.append(company_info.strip())
    
    # If we found relevant information, return it
    if response_parts:
        return "\n\n".join(response_parts)
    
    # If we didn't find anything relevant, give a general response
    return "I'm Asha, your virtual assistant for JobsForHer. I can help you with finding job opportunities, mentorship programs, upcoming events, and more. Could you provide more details about what you're looking for?"

# Create Gradio interface
def respond(message, history):
    bot_message = generate_response(message, history)
    return bot_message

# Create welcome message
welcome_message = """
# Welcome to Asha AI 👋

I'm your virtual assistant for JobsForHer Foundation. I can help you with:

* Finding job opportunities
* Learning about mentorship programs
* Discovering upcoming events
* Answering questions about profile setup
* Providing career guidance for women professionals

How can I assist you today?
"""

# Create and launch the interface
demo = gr.ChatInterface(
    respond,
    title="Asha AI - JobsForHer Assistant",
    description=welcome_message,
    theme="soft",
    examples=[
        "Tell me about JobsForHer",
        "What job opportunities are available?",
        "Are there any upcoming events?",
        "How do I update my profile?",
        "Tell me about mentorship programs"
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
