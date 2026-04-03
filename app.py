import streamlit as st
import pickle
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load model
df = pickle.load(open("C:/recommendation_system/products.pkl","rb"))
tfidf = pickle.load(open("C:/recommendation_system/tfidf.pkl","rb"))
tfidf_matrix = pickle.load(open("C:/recommendation_system/tfidf_matrix.pkl","rb"))

st.set_page_config(layout="wide")
st.title("🛍️ AI Fashion Intelligence System")

# ================= IMAGE =================
def get_image(product_id):
    path = f"images/{product_id}.jpg"
    if os.path.exists(path):
        return path
    return "https://via.placeholder.com/150"

# ================= USER INPUT =================
st.sidebar.header("🎯 User Preferences")

gender = st.sidebar.selectbox("Gender", df['gender'].unique())
category = st.sidebar.selectbox("Category", df['masterCategory'].unique())
color = st.sidebar.selectbox("Color", df['baseColour'].unique())
season = st.sidebar.selectbox("Season", df['season'].unique())

user_input = [gender, category, color, season]

# ================= RECOMMEND =================
def recommend(user_input):

    user_text = " ".join(user_input)
    user_vec = tfidf.transform([user_text])

    similarity = cosine_similarity(user_vec, tfidf_matrix)
    scores = list(enumerate(similarity[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    seen = set()
    unique_indices = []

    for i, score in scores:
        article = df.iloc[i]['articleType']

        if article not in seen:
            seen.add(article)
            unique_indices.append(i)

        if len(unique_indices) == 8:
            break

    return df.iloc[unique_indices]

# ================= BUSINESS INSIGHTS =================

def business_insights():

    st.subheader("📊 Business Insights")

    col1, col2, col3 = st.columns(3)

    # Top categories
    with col1:
        top_cat = df['masterCategory'].value_counts().head(3)
        st.write("🔥 Top Categories")
        st.write(top_cat)

    # Popular colors
    with col2:
        colors = df['baseColour'].value_counts().head(3)
        st.write("🎨 Popular Colors")
        st.write(colors)

    # Seasonal trend
    with col3:
        season_trend = df['season'].value_counts()
        st.write("🌦️ Season Trend")
        st.write(season_trend)

# ================= CART LOGIC =================

def cart_strategy(results):

    st.subheader("💰 Sales Strategy")

    st.write("✔ Customers who buy this may also buy:")

    combos = results['masterCategory'].value_counts()

    for item in combos.index[:3]:
        st.write(f"👉 {item} + Accessories")

# ================= STYLING =================

def style_recommendation(main_product):

    category = main_product['masterCategory']

    style_map = {
        'Apparel': ['Footwear', 'Accessories'],
        'Footwear': ['Apparel'],
        'Accessories': ['Apparel']
    }

    related = style_map.get(category, [])
    styled = df[df['masterCategory'].isin(related)]

    return styled.sample(min(4, len(styled)))

# ================= MAIN BUTTON =================

if st.sidebar.button("🔍 Generate Intelligence"):

    results = recommend(user_input)

    # ===== PRODUCTS =====
    st.subheader("🎯 Recommended Products")

    cols = st.columns(4)

    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i % 4]:

            st.image(get_image(row['id']), width= "stretch")

            st.markdown(f"**{row['articleType']}**")
            st.write(f"{row['baseColour']} | {row['season']}")
            st.caption("✔ Based on your preference")

    # ===== STYLING =====
    st.subheader("👗 Complete the Look")

    style_items = style_recommendation(results.iloc[0])

    cols = st.columns(4)

    for i, (_, row) in enumerate(style_items.iterrows()):
        with cols[i]:

            st.image(get_image(row['id']), width="stretch")
            st.write(row['articleType'])

    # ===== BUSINESS INSIGHTS =====
    business_insights()

    # ===== CART STRATEGY =====
    cart_strategy(results)