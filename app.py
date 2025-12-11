import streamlit as st
import pandas as pd
import requests
import nltk
import matplotlib.pyplot as plt 
import altair as alt 
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

# --- IMPORT UI MODULE ---
try:
    import ui
    ui.load_css()           
    ui.inject_aos()         
    ui.set_background()     
except Exception:
    pass 

# --- CONFIG ---
st.set_page_config(page_title="NarrativeNexus", layout="wide", page_icon="🔴")

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="google/flan-t5-small")

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# NLTK Setup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# --- DATA PROCESSING ---
def load_data(uploaded_file):
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                # FORCE UTF-8 ENCODING TO PRESERVE EMOJIS
                return pd.read_csv(uploaded_file, encoding='utf-8', encoding_errors='replace')
            elif uploaded_file.name.endswith('.txt'):
                return pd.DataFrame({"text": [uploaded_file.read().decode("utf-8")]})
        except UnicodeDecodeError:
            # Fallback if utf-8 fails
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding='latin1')
            except Exception as e:
                st.error(f"Error reading file: {e}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None

def preprocess_text(text):
    # This is for TOPIC MODELING only (removes emojis/punctuation)
    text = str(text).lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words])

# --- ANALYSIS ALGORITHMS ---
def run_topic_modeling(text_data, n_topics=3):
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(text_data)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    return lda, vectorizer

# --- UPDATED SENTIMENT FUNCTION ---
def get_sentiment(text):
    """
    Uses VADER with CUSTOM UPDATES for slang and emojis.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    # === MANUALLY TEACHING THE MODEL ===
    new_words = {
        '🔥': 4.0,        
        'lit': 3.0,
        'goat': 3.5,
        'mid': -2.0,
        'trash': -3.5,
        'meh': -1.5,
        '🤢': -3.5,
        '😡': -3.5,
        '❤️': 3.5,
        '🙂': 2.0,
        'w': 3.0,         # 'W' (Win)
        'l': -3.0         # 'L' (Loss)
    }
    analyzer.lexicon.update(new_words)
    
    score = analyzer.polarity_scores(str(text))
    return score['compound']

# --- EMOJI LABEL FUNCTION ---

def get_sentiment_label(score):
    
    if score >= 0.25:
        return "Positive 😀"
    elif score <= -0.25:
        return "Negative 😞"
    else:
        return "Neutral 😐"

# --- MAIN APP LAYOUT ---
lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_m9n89kpl.json")

with st.sidebar:
    if lottie_ai:
        st_lottie(lottie_ai, height=200, key="ai_bot")
    st.markdown("---")
    
    # --- DEBUG SANDBOX ---
    st.markdown("### 🧪 Live Sentiment Test")
    test_input = st.text_input("Type a sentence (e.g., 'So 🔥'):")
    if test_input:
        score = get_sentiment(test_input)
        label = get_sentiment_label(score)
        st.markdown(f"**Score:** {score:.2f}")
        st.markdown(f"**Result:** {label}")
        if score == 0:
            st.warning("⚠️ If score is 0, the model didn't see the emoji.")
    
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Instructions", "Upload Data", "Topic Modeling", "Sentiment Analysis", "Reports"],
        icons=["info-circle-fill", "cloud-upload", "cpu-fill", "heart-pulse-fill", "file-earmark-text"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "transparent"},
            "icon": {"color": "#ff3131", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#300000"},
            "nav-link-selected": {"background-color": "#ff3131", "color": "#ffffff"},
        }
    )

st.title("NarrativeNexus ⚡")
st.markdown("### The AI-Powered Dynamic Text Analysis Platform")

# ==========================================
# 1. INSTRUCTIONS TAB
# ==========================================
if selected == "Instructions":
    st.markdown("""<div data-aos="fade-right">""", unsafe_allow_html=True)
    
    st.markdown("## 📚 Platform Documentation")
    st.write("Welcome to NarrativeNexus. This platform uses advanced Natural Language Processing (NLP) techniques to analyze text data.")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="padding:20px; border:1px solid #ff3131; border-radius:10px; height:100%;">
            <h3 style="color:#ff3131;">🧠 Summarization Engine</h3>
            <p><strong>Model Used:</strong> <code>Google Flan-T5 Small</code></p>
            <p><strong>Provider:</strong> HuggingFace Transformers</p>
            <p><strong>How it works:</strong> An abstractive summarization model that understands context and generates new sentences to summarize the input text.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div style="padding:20px; border:1px solid #ff3131; border-radius:10px; height:100%;">
            <h3 style="color:#ff3131;">🔍 Topic Modeling</h3>
            <p><strong>Algorithm:</strong> Latent Dirichlet Allocation (LDA)</p>
            <p><strong>Library:</strong> Scikit-Learn</p>
            <p><strong>How it works:</strong> A statistical model that groups words that frequently appear together into hidden "Topics".</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
        <div style="padding:20px; border:1px solid #ff3131; border-radius:10px; height:100%;">
            <h3 style="color:#ff3131;">❤️ Sentiment & Emojis</h3>
            <p><strong>Algorithm:</strong> VADER with Custom Lexicon</p>
            <p><strong>Enhanced Logic:</strong> We manually taught the model that:</p>
            <ul style="color:#e0e0e0;">
                <li>🔥 / "Lit" = Positive</li>
                <li>"Mid" / "Meh" = Negative/Neutral</li>
                <li>"Goat" = Strong Positive</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with c4:
        st.markdown("""
        <div style="padding:20px; border:1px solid #ff3131; border-radius:10px; height:100%;">
            <h3 style="color:#ff3131;">⚙️ Preprocessing Pipeline</h3>
            <p>Before Topic Modeling, text undergoes cleaning:</p>
            <ol style="color:#e0e0e0;">
                <li><strong>Lowercasing:</strong> "Hello" → "hello"</li>
                <li><strong>Noise Removal:</strong> Special characters and punctuation removed.</li>
                <li><strong>Stopword Removal:</strong> Common words (the, is, at) are stripped.</li>
                <li><strong>Lemmatization:</strong> Words converted to root form (e.g., "running" → "run").</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 How to Use")
    st.info("1. Go to **Upload Data** and drop your CSV file.\n2. Navigate to **Topic Modeling** to find hidden themes.\n3. Check **Sentiment Analysis** for emoji-based breakdowns.\n4. Use **Reports** to visualize word clouds.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 2. UPLOAD DATA TAB
# ==========================================
elif selected == "Upload Data":
    st.markdown("""<div data-aos="fade-up" data-aos-duration="1000">""", unsafe_allow_html=True)
    st.markdown("""
    <div style="border: 1px solid #ff3131; padding: 20px; border-radius: 10px; background-color: rgba(255, 49, 49, 0.05);">
        <h3 style="color: #ffffff;">📂 Data Upload</h3>
        <p style="color: #e0e0e0;">Upload your dataset below. Supports CSV and TXT.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['txt', 'csv']) 
    if uploaded_file:
        st.session_state['df'] = load_data(uploaded_file)
        st.success("Data uploaded successfully!")
        st.dataframe(st.session_state['df'].head())
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 3. TOPIC MODELING TAB
# ==========================================
elif selected == "Topic Modeling":
    if 'df' in st.session_state:
        df = st.session_state['df']
        text_col = st.selectbox("Select text column to analyze:", df.columns.tolist())
        df[text_col] = df[text_col].astype(str)
        df['cleaned_text'] = df[text_col].apply(preprocess_text)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🔍 Topic Extraction")
            num_topics = st.slider("Number of Topics", 2, 10, 3)
            if st.button("Run LDA Analysis"):
                try:
                    lda_model, vectorizer = run_topic_modeling(df['cleaned_text'], num_topics)
                    feature_names = vectorizer.get_feature_names_out()
                    for idx, topic in enumerate(lda_model.components_):
                        keywords = ", ".join([feature_names[i] for i in topic.argsort()[-10:]])
                        st.markdown(f"""
                        <div style="padding:15px; border-left: 4px solid #ff3131; background:rgba(255,255,255,0.05); margin-bottom:10px; border-radius:0 5px 5px 0;">
                            <strong style="color:#ff3131; font-size:1.1em;">Topic {idx+1}:</strong> 
                            <br><span style="color:#e0e0e0">{keywords}</span>
                        </div>
                        """, unsafe_allow_html=True)
                except ValueError:
                    st.error("Not enough clean text to generate topics.")
        with col2:
            st.subheader("🧠 Summarization")
            if st.button("Generate Summary"):
                with st.spinner("AI is thinking..."):
                    try:
                        summarizer = load_summarizer()
                        combined_text = " ".join(df[text_col].astype(str).tolist())[:2000] 
                        summary_result = summarizer(combined_text, max_length=130, min_length=30, do_sample=False)
                        st.markdown(f"""
                        <div style="padding:20px; background-color:rgba(255, 49, 49, 0.1); border: 1px solid #ff3131; border-radius: 10px;">
                            <h4 style="color:#ff3131; margin:0;">AI Insight:</h4>
                            <p style="color:white; font-size:1em; margin-top:10px;">{summary_result[0]['summary_text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.warning("Please upload data first.")

# ==========================================
# 4. SENTIMENT ANALYSIS TAB
# ==========================================
elif selected == "Sentiment Analysis":
    if 'df' in st.session_state:
        df = st.session_state['df']
        text_col = st.selectbox("Select text column for sentiment:", df.columns.tolist(), key="sent_col")
        
        # Calculate Scores
        df['sentiment_score'] = df[text_col].astype(str).apply(get_sentiment)
        
        # Apply Emoji Labels
        df['category'] = df['sentiment_score'].apply(get_sentiment_label)
        
        st.markdown("---")
        
        # --- GRAPH SECTION ---
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("📊 Sentiment Distribution")
            chart_data = df['category'].value_counts().reset_index()
            chart_data.columns = ['category', 'count']
            
            domain = ['Negative 😞', 'Neutral 😐', 'Positive 😀']
            range_ = ['#ff3131', '#808080', '#00ff9d'] 
            
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('category', axis=alt.Axis(labelColor='white', titleColor='white')),
                y=alt.Y('count', axis=alt.Axis(labelColor='white', titleColor='white')),
                color=alt.Color('category', scale=alt.Scale(domain=domain, range=range_), legend=None),
                tooltip=['category', 'count']
            ).properties(height=300).configure_view(strokeWidth=0).configure_axis(grid=False)
            
            st.altair_chart(chart, use_container_width=True)

        with c2:
            st.subheader("Metrics")
            total = len(df)
            pos_pct = (len(df[df['category']=='Positive 😀']) / total) * 100
            neg_pct = (len(df[df['category']=='Negative 😞']) / total) * 100
            
            st.metric("Total Reviews", total)
            st.metric("Positive 😀 %", f"{pos_pct:.1f}%")
            st.metric("Negative 😞 %", f"{neg_pct:.1f}%")

        st.markdown("---")
        
        # --- REVIEW INSPECTOR ---
        st.subheader("🕵️ Review Inspector")
        st.write("Click a category below to see specific reviews:")
        
        filter_choice = st.radio("Show me:", ["All", "Positive 😀", "Negative 😞", "Neutral 😐"], horizontal=True)
        
        if filter_choice == "All":
            filtered_df = df
        else:
            filtered_df = df[df['category'] == filter_choice]
        
        st.markdown(f"**Showing {len(filtered_df)} {filter_choice if filter_choice != 'All' else ''} reviews:**")
        st.dataframe(filtered_df[[text_col, 'sentiment_score', 'category']], use_container_width=True, height=400)
    else:
        st.warning("Please upload data first.")

# ==========================================
# 5. REPORTS TAB
# ==========================================
elif selected == "Reports":
    st.subheader("☁️ Word Cloud Generation")
    if 'df' in st.session_state:
        df = st.session_state['df']
        if 'cleaned_text' not in df.columns:
            text_col = df.columns[0]
            df['cleaned_text'] = df[text_col].astype(str).apply(preprocess_text)
            
        all_text = " ".join(df['cleaned_text'])
        if len(all_text) > 0:
            st.markdown("generating visualization...")
            
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='black', 
                colormap='inferno', 
                contour_color='#ff3131', 
                contour_width=1, 
                max_words=100
            ).generate(all_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('black')
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            
            st.markdown('<div style="border: 2px solid #ff3131; border-radius: 10px; overflow: hidden;">', unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Not enough text data for Word Cloud.")
    else:
        st.warning("No data available.")

st.markdown("<script>AOS.init();</script>", unsafe_allow_html=True)