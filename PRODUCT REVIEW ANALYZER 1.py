#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load the dataset
df = pd.read_csv("Reviews.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

# Data Preprocessing
df = df[['Score', 'Summary', 'Text']]
df = df.dropna()
df = df.drop_duplicates(subset=['Text'])
df.reset_index(drop=True, inplace=True)

# Define positive and negative words for sentiment analysis
positive_words = ["good", "excellent", 'amazing', "love", "great", "best", "wonderful", "like", "happy", "fantastic"]
negative_words = ["bad", "terrible", "awful", "worst", "hate", "poor", "horrible", "sad", "boring"]

# Sentiment Analysis function
def get_sentiment(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    pos_count = sum(1 for word in words if word in positive_words)  # Fix variable name
    neg_count = sum(1 for word in words if word in negative_words)  # Fix variable name
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis to the 'Text' column
df["Sentiment"] = df["Text"].apply(get_sentiment)
print(df[["Text", "Sentiment"]].head())

# Install the necessary libraries (uncomment if not installed)
# !pip install faiss-cpu sentence-transformers
# !pip install --upgrade typing_extensions transformers torch sentence-transformers

# Initialize SentenceTransformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example reviews
reviews = [
    "The battery lasts for two days on a full charge.",
    "This phone has an amazing camera, but the battery drains fast.",
    "I love the display, but the battery is average."
]

# Generate embeddings for reviews
embeddings = model.encode(reviews, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
index.add(embeddings)

# Define the RAG query function
def rag_query(user_query, top_k=2):
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve top-k most relevant reviews
    retrieved_reviews = [reviews[i] for i in indices[0]]
    context = " ".join(retrieved_reviews)
    
    # Generate a response using GPT-2
    qa_pipeline = pipeline("text-generation", model="gpt2")
    prompt = f"Based on these reviews, {user_query}:\n{context}"
    response = qa_pipeline(prompt)
    
    return response[0]["generated_text"]

# Example query
query = "What do people say about battery life?"
answer = rag_query(query)
print(answer)

# Generating a summary using GPT-2
text_generator = pipeline("text-generation", model="gpt2")

retrieved_reviews = [
    "The battery life is amazing and lasts all day!",
    "The battery drains fast when playing games."
]

# Create a text prompt for GPT-2
prompt = f"Based on user reviews, summarize how the battery performance is:\n\n{retrieved_reviews}"

# Generate response
summary = text_generator(prompt, max_length=50, do_sample=True)[0]["generated_text"]
print("\n🔹 Generated Summary:")
print(summary)

# Function to ask a review and get AI-generated summary
def ask_review(query):
    """Retrieve relevant reviews and return only the AI-generated summary."""
    
    # Convert query to embedding
    query_embedding = model.encode([query])
    
    # Retrieve top 2 relevant reviews
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), 2)
    
    # Fetch relevant reviews
    retrieved_reviews = [reviews[i] for i in indices[0]]
    
    # Generate summary using GPT-2
    prompt = f"Based on user reviews, summarize the answer:\n\n{retrieved_reviews}"
    summary = text_generator(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    
    # Only show the final answer to the user
    print(summary)

# Example Usage:
ask_review("What is the feedback on the camera quality?")


# In[ ]:





# In[2]:


import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline


# In[3]:


df = pd.read_csv("Reviews.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())


# In[4]:


df = df[['Score', 'Summary', 'Text']]
df = df.dropna()
df = df.drop_duplicates(subset=['Text'])
df.reset_index(drop=True, inplace=True)


# In[5]:


positive_words = ["good", "excellent", 'amazing', "love", "great", "best", "wonderful", "like", "happy", "fantastic"]
negative_words = ["bad", "terrible", "awful", "worst", "hate", "poor", "horrible", "sad", "boring"]


# In[6]:


def get_sentiment(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    pos_count = sum(1 for word in words if word in positive_words)  # Fix variable name
    neg_count = sum(1 for word in words if word in negative_words)  # Fix variable name
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"


# In[7]:


df["Sentiment"] = df["Text"].apply(get_sentiment)
print(df[["Text", "Sentiment"]].head())


# In[8]:


model = SentenceTransformer("all-MiniLM-L6-v2")


# In[9]:


reviews = [
    "The battery lasts for two days on a full charge.",
    "This phone has an amazing camera, but the battery drains fast.",
    "I love the display, but the battery is average."
]


# In[10]:


embeddings = model.encode(reviews, convert_to_numpy=True)


# In[11]:


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
index.add(embeddings)


# In[12]:


def rag_query(user_query, top_k=2):
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve top-k most relevant reviews
    retrieved_reviews = [reviews[i] for i in indices[0]]
    context = " ".join(retrieved_reviews)
    
    # Generate a response using GPT-2
    qa_pipeline = pipeline("text-generation", model="gpt2")
    prompt = f"Based on these reviews, {user_query}:\n{context}"
    response = qa_pipeline(prompt)
    
    return response[0]["generated_text"]


# In[13]:


query = "What do people say about battery life?"
answer = rag_query(query)
print(answer)


# In[14]:


text_generator = pipeline("text-generation", model="gpt2")


# In[15]:


retrieved_reviews = [
    "The battery life is amazing and lasts all day!",
    "The battery drains fast when playing games."
]


# In[16]:


prompt = f"Based on user reviews, summarize how the battery performance is:\n\n{retrieved_reviews}"


# In[17]:


summary = text_generator(prompt, max_length=50, do_sample=True)[0]["generated_text"]
print("\n🔹 Generated Summary:")
print(summary)


# In[18]:


def ask_review(query):
    """Retrieve relevant reviews and return only the AI-generated summary."""
    
    # Convert query to embedding
    query_embedding = model.encode([query])
    
    # Retrieve top 2 relevant reviews
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), 2)
    
    # Fetch relevant reviews
    retrieved_reviews = [reviews[i] for i in indices[0]]
    
    # Generate summary using GPT-2
    prompt = f"Based on user reviews, summarize the answer:\n\n{retrieved_reviews}"
    summary = text_generator(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    
    # Only show the final answer to the user
    print(summary)

# Example Usage:
ask_review("What is the feedback on the camera quality?")


# In[19]:


ask_review("What is the feedback on the camera quality?")


# In[20]:


def interactive_review_bot():
    print("🔹 Welcome to the Product Review Analyzer 🔹")
    print("Ask about a product (e.g., 'How is the battery life?') or type 'exit' to quit.\n")
    
    while True:
        query = input("🔹 Enter your query: ").strip()
        
        if query.lower() == "exit":
            print("👋 Exiting... Thank you for using the Review Analyzer!")
            break
        
        print("\n⏳ Fetching reviews and generating summary...\n")
        
        # Get AI-generated summary
        summary = ask_review(query)
        
        print("\n🔹 **Summary of User Reviews:**")
        print(summary)
        print("\n" + "="*50 + "\n")

# Run the interactive interface
interactive_review_bot()


# In[ ]:


def get_sentiment(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    pos_count = sum(1 for word in words if word in positive_words)  # Corrected variable name
    neg_count = sum(1 for word in words if word in negative_words)  # Corrected variable name
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"


# In[ ]:


retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = retrieval_model.encode(reviews, convert_to_numpy=True)


# In[ ]:


from transformers import pipeline
model = SentenceTransformer("all-MiniLM-L6-v2")
retrieval_model = model  # Corrected the variable usage

# Initialize GPT-2 text-generation model
text_generator = pipeline("text-generation", model="gpt2")

reviews = [
    "The battery lasts for two days on a full charge.",
    "This phone has an amazing camera, but the battery drains fast.",
    "I love the display, but the battery is average.",
]

# Ensure embeddings are created after the model is initialized
embeddings = retrieval_model.encode(reviews, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Continue with the rest of the code...


# In[ ]:


def interactive_review_bot():
    print("🔹 Welcome to the Product Review Analyzer 🔹")
    print("Ask about a product (e.g., 'How is the battery life?') or type 'exit' to quit.\n")
    
    while True:
        query = input("🔹 Enter your query: ").strip()
        
        if query.lower() == "exit":
            print("👋 Exiting... Thank you for using the Review Analyzer!")
            break
        
        print("\n⏳ Fetching reviews and generating summary...\n")
        
        # Get AI-generated summary
        summary = ask_review(query)
        
        print("\n🔹 **Summary of User Reviews:**")
        print(summary)
        print("\n" + "="*50 + "\n")

# Run the interactive interface
interactive_review_bot()


# In[21]:


interactive_review_bot()


# In[22]:


def ask_review(query):
    """Retrieve relevant reviews and return only the AI-generated summary."""
    
    # Convert query to embedding
    query_embedding = model.encode([query])

    # Retrieve top 2 relevant reviews
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), 2)

    # Fetch relevant reviews
    retrieved_reviews = [reviews[i] for i in indices[0]]

    # Create a text prompt to pass to the GPT-2 model for summarization
    prompt = f"Based on user reviews for Dell products, summarize the answer to the query: {query}\n\nReviews:\n{retrieved_reviews}"

    # Generate summary using GPT-2
    summary = text_generator(prompt, max_length=50, do_sample=True)[0]["generated_text"]

    # Return the final summary
    return summary.strip()

# Example query
query = "Tell me about the display and battery life of Dell products."
summary = ask_review(query)
print("\n🔹 **Summary of User Reviews:**")
print(summary)


# In[23]:


def ask_review(query):
    """Retrieve relevant reviews and return only the AI-generated summary."""
    
    # Convert query to embedding
    query_embedding = model.encode([query])

    # Retrieve top 2 relevant reviews
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), 2)

    # Fetch relevant reviews
    retrieved_reviews = [reviews[i] for i in indices[0]]

    # Create a text prompt to pass to the GPT-2 model for summarization
    prompt = f"Based on user reviews for Dell products, summarize the answer to the query: {query}\n\nReviews:\n{retrieved_reviews}"

    # Generate summary using GPT-2
    summary = text_generator(prompt, max_new_tokens=50, do_sample=True)[0]["generated_text"]

    # Return the final summary
    return summary.strip()

# Example query
query = "Tell me about the display and battery life of Dell products."
summary = ask_review(query)
print("\n🔹 **Summary of User Reviews:**")
print(summary)


# In[24]:


interactive_review_bot()


# In[26]:


interactive_review_bot()


# In[27]:


reviews = [
    "The LG AC cools the room quickly and efficiently.",
    "The LG AC has excellent energy efficiency and low noise.",
    "The LG AC is a bit noisy but works great for cooling.",
    "I love the cooling performance of the LG AC, but it's a bit expensive."
]

def ask_review(query):
    """Retrieve relevant reviews and return only the AI-generated summary."""
    
    # Convert query to embedding
    query_embedding = model.encode([query])

    # Retrieve top 2 relevant reviews
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), 2)

    # Fetch relevant reviews
    retrieved_reviews = [reviews[i] for i in indices[0]]

    # Create a text prompt to pass to the GPT-2 model for summarization
    prompt = f"Based on user reviews for LG ACs, summarize the answer to the query: {query}\n\nReviews:\n{retrieved_reviews}"

    # Generate summary using GPT-2
    summary = text_generator(prompt, max_new_tokens=50, do_sample=True)[0]["generated_text"]

    # Return the final summary
    return summary.strip()

# Example query
query = "Tell me about the LG AC."
summary = ask_review(query)
print("\n🔹 **Summary of User Reviews:**")
print(summary)


# In[ ]:


interactive_review_bot()


# In[ ]:




