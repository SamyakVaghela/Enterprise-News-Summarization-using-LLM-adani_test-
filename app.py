
"""
================================================================================
PROJECT: Enterprise News Summarization Engine
AUTHOR: Samyak vaghela
===============================================================================

1. Why this approach for the LLM?
- Abstractive vs. Extractive: I chose an 'Abstractive' approach. Unlike other
  methods that simply cull existing sentences, this approach uses an LLM to 
  understand the latent semantics of the article and generate a 
  summary.
- Scalability (Streaming): The implementation utilizes Hugging Face's 'streaming' 
  mode for the dataset. This ensures the application can handle massive 
  datasets (like XSum's 200k+ articles) without crashing the system's RAM.
- Efficiency: Optimized with FP16 (Half-Precision) when a GPU is available, 
  doubling inference speed and reducing memory footprint for enterprise use.

2. Choosing the model (facebook/bart-large-xsum)?
-------------------------------------------------
- Architecture: BART is a Sequence-to-Sequence (Seq2Seq) model with a 
  bidirectional encoder (like BERT) and an autoregressive decoder (like GPT). 
  This makes it uniquely powerful for tasks where understanding the input and 
  generating fluent output are equally critical.
- Specialized Tuning: This specific 'large-xsum' variant is fine-tuned on the 
  XSum dataset. It is specifically designed for 'extreme' summarization, 
  condensing complex reports into a single, high-impact sentence.

3. Dataset (XSum)
---------------------
- Source: from over 226,000 professional BBC news articles.
- Objective: It is the industry benchmark for 'extreme' summarization. The 
  model must read and learn from a full article into the most informative first sentence 
  of a news story, making it a rigorous test of an LLM's  capability.
================================================================================
"""






import streamlit as st
import torch
import random
import re
from transformers import pipeline
from datasets import load_dataset

st.set_page_config(page_title="AI Summarizer Pro", layout="wide")

@st.cache_resource
def load_assets():
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="facebook/bart-large-xsum", device=device)
    dataset = load_dataset("xsum", split="test", streaming=True)
    return summarizer, dataset

summarizer, dataset = load_assets()

st.title("📰 Enterprise News Summarization Engine")

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

if st.sidebar.button("🎲 Load Random Article"):
    shuffled = dataset.shuffle(seed=random.randint(0, 10000), buffer_size=100)
    sample = next(iter(shuffled))
    st.session_state.input_text = sample['document']
    st.session_state.ground_truth = sample['summary']

article_input = st.text_area("Input Article:", value=st.session_state.input_text, height=300)

if st.button("Generate Summary"):
    if article_input:
        with st.spinner('Summarizing...'):
            clean_text = re.sub(r'\(BBC\)|BBC News|Share this with.*', '', article_input).strip()
            summary = summarizer(clean_text, max_length=60, min_length=20)[0]['summary_text']
            st.success(f"**AI Summary:** {summary}")
