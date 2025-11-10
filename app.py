# import streamlit as st
# import numpy as np
# import re
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression


# # Load data
# news_df = pd.read_csv(r'C:\Users\harman kaur makkad\Desktop\FNDAI\Fake-News-Detection-Machine-Learning-Scam-Detection-NLP\train.csv')
# news_df = news_df.fillna(' ')
# news_df['content'] = news_df['author'] + ' ' + news_df['title']
# X = news_df.drop('label', axis=1)
# y = news_df['label']

# # Define stemming function
# ps = PorterStemmer()
# def stemming(content):
#     stemmed_content = re.sub('[^a-zA-Z]',' ',content)
#     stemmed_content = stemmed_content.lower()
#     stemmed_content = stemmed_content.split()
#     stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
#     stemmed_content = ' '.join(stemmed_content)
#     return stemmed_content

# # Apply stemming function to content column
# news_df['content'] = news_df['content'].apply(stemming)

# # Vectorize data
# X = news_df['content'].values
# y = news_df['label'].values
# vector = TfidfVectorizer()
# vector.fit(X)
# X = vector.transform(X)

# # Split data into train and test sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# # Fit logistic regression model
# model = LogisticRegression()
# model.fit(X_train,Y_train)


# # website
# st.title('Fake News Detector')
# input_text = st.text_input('Enter news Article')

# def prediction(input_text):
#     input_data = vector.transform([input_text])
#     prediction = model.predict(input_data)
#     return prediction[0]

# if input_text:
#     pred = prediction(input_text)
#     if pred == 1:
#         st.write('The News is Fake')
#     else:
#         st.write('The News Is Real')

# import streamlit as st
# import pandas as pd

# st.title("📰 Fake News Detector - Debug Mode")

# try:
#     news_df = pd.read_csv(r'C:\Users\harman kaur makkad\Desktop\FNDAI\Fake-News-Detection-Machine-Learning-Scam-Detection-NLP\train.csv')
#     st.success("✅ CSV Loaded Successfully!")
#     st.write(news_df.head())
# except Exception as e:
#     st.error(f"❌ Error loading CSV: {e}")

#CHATGPT
import os
import streamlit as st
import joblib
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')

# --- Custom CSS for background and text ---
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://www.azoai.com/images/news/ImageForNews_2250_17031242031341781.jpg");
        background-size: cover;
        background-position: center;
    }
    h1 {
        color: red;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }
    div.stButton > button {
        background-color: black;
        color: yellow;
        border-radius: 10px;
        height: 3em;
        width: 10em;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: red;
        color: white;
    }
    textarea {
        border: 2px solid #1f4e79;
        border-radius: 10px;
        background-color: red;
    }
    </style>
""", unsafe_allow_html=True)

# --- UI Title ---
st.title("📰 Fake News Detector")
st.write("This app predicts whether a news article is **Real** or **Fake**.")

# --- Step 1: Check Current Directory (Debug Info) ---
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

# --- Step 2: Load Dataset (Relative Path Fix) ---
try:
    st.info("📂 Loading dataset...")

    # ✅ Use relative path — works on Vercel & locally
    dataset_path = os.path.join(os.path.dirname(__file__), "train.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"train.csv not found in: {os.getcwd()}")

    news_df = pd.read_csv(dataset_path)
    news_df = news_df.fillna(' ')
    news_df['content'] = news_df['author'] + ' ' + news_df['title']

    st.success("✅ Dataset loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# --- Step 3: Preprocessing ---
st.info("⚙️ Processing text data...")

ps = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

try:
    news_df['content'] = news_df['content'].apply(stemming)
    st.success("✅ Text preprocessing done!")
except Exception as e:
    st.error(f"❌ Error during preprocessing: {e}")
    st.stop()

# --- Step 4: TF-IDF and Model Training ---
try:
    st.info("📊 Training the model (please wait)...")
    X = news_df['content'].values
    y = news_df['label'].values
    vector = TfidfVectorizer()
    X = vector.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    st.success("✅ Model trained successfully!")
except Exception as e:
    st.error(f"❌ Model training failed: {e}")
    st.stop()

# --- Step 5: Prediction Interface ---
st.subheader("🔍 Check News Authenticity")

input_text = st.text_area("Enter a news article or headline here:")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        input_data = vector.transform([input_text])
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("🚨 The News is **Fake**.")
        else:
            st.success("✅ The News is **Real**.")


