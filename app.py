import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt 
import altair as alt 
import pypdf
import faiss 
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

# --- CONFIG ---
st.set_page_config(page_title="NarrativeNexus Pro", layout="wide", page_icon="🔴")

# --- UI STYLING ---
try:
    import ui
    ui.load_css()           
    ui.inject_aos()         
    ui.set_background()     
except Exception:
    pass 

# --- HELPER: LOAD LOTTIE ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# --- CACHED MODEL LOADING (CRITICAL FOR PERFORMANCE) ---

@st.cache_resource
def load_rag_models():
    """Loads embedding model and T5 for RAG"""
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    return embedder, generator

@st.cache_resource
def load_roberta_model():
    """Loads RoBERTa for Sentiment"""
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

# --- DATA LOADER ---
def load_data(uploaded_file):
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    return pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='warn')
                except:
                    # Fallback for comma errors
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode("utf-8", errors='replace')
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    if lines and 'review_text' in lines[0].lower(): lines = lines[1:]
                    return pd.DataFrame({"review_text": lines})

            elif uploaded_file.name.endswith('.pdf'):
                pdf_reader = pypdf.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                lines = [line for line in text.split('\n') if len(line) > 20]
                return pd.DataFrame({"review_text": lines})

            elif uploaded_file.name.endswith('.txt'):
                content = uploaded_file.read().decode("utf-8")
                lines = content.split('\n')
                return pd.DataFrame({"review_text": [line for line in lines if line.strip() != ""]})
                
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None

# --- AI ENGINES ---

def get_roberta_sentiment(text):
    """Deep Learning Sentiment Analysis"""
    tokenizer, model = load_roberta_model()
    encoded_input = tokenizer(str(text)[:512], return_tensors='pt')
    output = model(**encoded_input)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores) # [Neg, Neu, Pos]
    
    labels = ['Negative 😞', 'Neutral 😐', 'Positive 😀']
    ranking = np.argsort(scores)[::-1]
    top_label = labels[ranking[0]]
    top_score = scores[ranking[0]]
    
    if top_label == 'Negative 😞': top_score = -top_score
    return top_score, top_label

def run_bertopic(docs):
    """Advanced Topic Modeling"""
    topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", min_topic_size=5)
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topic_model.get_topic_info()

class RAGEngine:
    """Chat with Data Logic"""
    def __init__(self, text_data):
        self.texts = text_data
        self.embedder, self.generator = load_rag_models()
        self.index = None
        self._build_index()
        
    def _build_index(self):
        embeddings = self.embedder.encode(self.texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
    def query(self, user_question):
        q_embed = self.embedder.encode([user_question])
        distances, indices = self.index.search(q_embed, 5) # Retrieve Top 5
        retrieved_docs = [self.texts[i] for i in indices[0]]
        context_str = "\n".join(retrieved_docs)
        prompt = f"question: {user_question} context: {context_str}"
        response = self.generator(prompt, max_length=150, do_sample=False)
        return response[0]['generated_text'], retrieved_docs

# --- MAIN APP LAYOUT ---

# Session State for Chat
if 'rag_engine' not in st.session_state: st.session_state.rag_engine = None
if 'messages' not in st.session_state: st.session_state.messages = []

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_m9n89kpl.json")

with st.sidebar:
    if lottie_ai: st_lottie(lottie_ai, height=200, key="ai_bot")
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Upload Data", "Advanced Topics", "AI Sentiment", "Chat with Data"],
        icons=["cloud-upload", "diagram-3-fill", "heart-pulse-fill", "chat-dots-fill"],
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#ff3131"}}
    )
    st.info("ℹ️ Pro Features Enabled: RoBERTa & BERTopic")

st.title("NarrativeNexus ⚡ Pro")

# 1. UPLOAD
if selected == "Upload Data":
    st.info("📂 Upload CSV, PDF, or TXT.")
    uploaded_file = st.file_uploader("", type=['txt', 'csv', 'pdf']) 
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['df'] = df
            
            # Init RAG on upload
            with st.spinner("Indexing data for Chat..."):
                text_col = df.columns[0]
                st.session_state.rag_engine = RAGEngine(df[text_col].astype(str).tolist())
            
            st.success("Data Loaded & Indexed!")
            st.dataframe(df.head())

# 2. ADVANCED TOPICS (BERTopic)
elif selected == "Advanced Topics":
    if 'df' in st.session_state:
        df = st.session_state['df']
        text_col = st.selectbox("Select Text Column:", df.columns)
        
        if st.button("Run BERTopic"):
            with st.spinner("Clustering topics..."):
                docs = df[text_col].astype(str).tolist()
                topic_model, topic_info = run_bertopic(docs)
                
                st.subheader("Topic Clusters")
                fig = topic_model.visualize_topics()
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Keywords")
                st.dataframe(topic_info[['Topic', 'Count', 'Name']])
    else:
        st.warning("Upload data first.")

# 3. AI SENTIMENT (RoBERTa)
elif selected == "AI Sentiment":
    if 'df' in st.session_state:
        df = st.session_state['df']
        text_col = st.selectbox("Select Text Column:", df.columns)
        
        if st.button("Run RoBERTa Analysis"):
            progress_bar = st.progress(0)
            results = []
            total = len(df)
            
            for i, text in enumerate(df[text_col]):
                score, label = get_roberta_sentiment(text)
                results.append({'score': score, 'label': label})
                if i % 10 == 0: progress_bar.progress((i + 1) / total)
            progress_bar.empty()
            
            res_df = pd.DataFrame(results)
            df = pd.concat([df.reset_index(drop=True), res_df], axis=1)
            
            # Export
            st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "sentiment.csv")
            
            # Visualization
            st.subheader("📊 Distribution")
            chart = alt.Chart(df['label'].value_counts().reset_index(name='count').rename(columns={'index':'label'})).mark_bar().encode(
                x='label', y='count', color=alt.Color('label', scale=alt.Scale(range=['#ff3131', '#808080', '#00ff9d']))
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Upload data first.")

# 4. CHAT WITH DATA (RAG)
elif selected == "Chat with Data":
    st.subheader("🤖 Ask your data")
    if 'rag_engine' in st.session_state and st.session_state.rag_engine:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, sources = st.session_state.rag_engine.query(prompt)
                    st.markdown(response)
                    with st.expander("Sources"):
                        for s in sources: st.text(f"- {s[:100]}...")
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Upload data first.")