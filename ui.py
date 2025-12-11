import streamlit as st
import base64
import os

# --- 1. CSS LOADER ---
def load_css(file_name="style.css"):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# --- 2. AOS ANIMATIONS ---
def inject_aos():
    st.markdown("""
        <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
        <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
        <script>
            document.addEventListener('DOMContentLoaded', function() { AOS.init(); }, false);
        </script>
    """, unsafe_allow_html=True)

# --- 3. BACKGROUND IMAGE SETTER ---
def set_background():
    """
    Sets 'banner.jpg' as the fixed background image for the entire app.
    Adds a dark overlay so text remains readable.
    """
    image_file = "banner.jpeg"
    
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)), url("data:image/jpg;base64,{bin_str}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """, unsafe_allow_html=True)
    else:
        st.error(f"⚠️ Image not found! looking for '{image_file}' in {os.getcwd()}")