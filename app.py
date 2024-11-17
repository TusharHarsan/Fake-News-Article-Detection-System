import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import time
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme and animated background
st.markdown("""
    <style>
        /* Modern Dark theme colors */
        :root {
            --bg-primary: #0a0a0a;
            --text-primary: #ffffff;
            --accent-primary: #00ff95;
            --accent-secondary: #ff3366;
            --gradient-1: #1a1a1a;
            --gradient-2: #0a0a0a;
        }
        
        /* Particle animation */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }
        
        .particle {
            position: absolute;
            width: 2px;
            height: 2px;
            background-color: rgba(255, 255, 255, 0.5);
            border-radius: 50%;
            animation: float 8s infinite ease-in-out;
        }
        
        @keyframes float {
            0%, 100% {
                transform: translateY(0) translateX(0);
                opacity: 0;
            }
            50% {
                transform: translateY(-100vh) translateX(100vw);
                opacity: 1;
            }
        }
        
        /* Modern UI Elements */
        .stApp {
            background: linear-gradient(135deg, var(--gradient-1), var(--gradient-2));
            color: var(--text-primary);
        }
        
        .title-container {
            text-align: center;
            padding: 2rem;
            background: rgba(0,0,0,0.4);
            border-radius: 20px;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
        }
        
        .stTitle {
            color: var(--accent-primary);
            font-size: 3rem;
            text-shadow: 0 0 20px rgba(0,255,149,0.5);
            font-weight: 700;
            letter-spacing: 2px;
        }
        
        .stats-container {
            background: rgba(255,255,255,0.05);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s ease;
        }
        
        .stats-container:hover {
            transform: translateY(-5px);
        }
        
        .result-box {
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
            font-size: 1.2rem;
            animation: fade-in 0.5s ease-out;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
        }
        
        .fake-news {
            background: rgba(255,51,102,0.1);
            border: 2px solid var(--accent-secondary);
        }
        
        .real-news {
            background: rgba(0,255,149,0.1);
            border: 2px solid var(--accent-primary);
        }
        
        /* Modern Button Styling */
        .stButton>button {
            background: linear-gradient(45deg, var(--accent-primary), var(--accent-secondary));
            color: var(--text-primary);
            border: none;
            padding: 0.8rem 2.5rem;
            border-radius: 50px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,255,149,0.3);
        }
        
        /* Sidebar Enhancement */
        .css-1d391kg {
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(10px);
        }
        
        /* Text Area Enhancement */
        .stTextArea>div>div>textarea {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            color: white;
            padding: 1rem;
        }
        
        .stTextArea>div>div>textarea:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 15px rgba(0,255,149,0.3);
        }
        
        /* Loading Animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .loading-spinner {
            text-align: center;
            padding: 2rem;
            color: var(--accent-primary);
            animation: pulse 1.5s infinite ease-in-out;
        }
    </style>
    
    <!-- Particle Background -->
    <div class="particles">
        <script>
            function createParticle() {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + 'vw';
                particle.style.animationDelay = Math.random() * 5 + 's';
                document.querySelector('.particles').appendChild(particle);
                setTimeout(() => particle.remove(), 8000);
            }
            
            setInterval(createParticle, 200);
        </script>
    </div>
""", unsafe_allow_html=True)

# Download NLTK data
nltk.download('stopwords')
stemmer = SnowballStemmer("english")

def get_text_stats(text):
    words = len(text.split())
    sentences = len(text.split('.'))
    chars = len(text)
    return words, sentences, chars

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [stemmer.stem(word) for word in con if not word in stopwords.words('english')]
    return ' '.join(con)

@st.cache_resource
def load_all_models():
    try:
        vector_load = pickle.load(open('vector.pkl', 'rb'))
        models = {
            'Passive Aggressive Classifier': pickle.load(open('passive_aggressive_classifier_model.pkl', 'rb')),
            'Random Forest': pickle.load(open('random_forest_model.pkl', 'rb')),
            'Support Vector Machine': pickle.load(open('support_vector_machine_model.pkl', 'rb')),
            'Naive Bayes': pickle.load(open('naive_bayes_model.pkl', 'rb'))
        }
        return vector_load, models
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run train_models.py first. Error: {e}")
        st.stop()

def fake_news_detect(news, model, vectorizer):
    # Preprocess the text
    news = stemming(news)
    # Transform using the loaded vectorizer
    vector_form = vectorizer.transform([news])
    # Predict using the selected model
    prediction = model.predict(vector_form)
    return prediction[0]

def get_ensemble_prediction(news_text, models, vectorizer):
    predictions = {}
    for model_name, model in models.items():
        pred = fake_news_detect(news_text, model, vectorizer)
        predictions[model_name] = pred
    
    # Get majority vote
    fake_count = sum(1 for pred in predictions.values() if pred == 'FAKE')
    real_count = len(predictions) - fake_count
    final_prediction = 'FAKE' if fake_count > real_count else 'REAL'
    
    return predictions, final_prediction, fake_count, real_count

def main():
    # Load models and vectorizer
    vector_load, models = load_all_models()

    # Sidebar for model selection
    st.sidebar.title("🤖 Model Selection")
    model_choice = st.sidebar.radio(
        "Choose your model:",
        list(models.keys())
    )

    # Model descriptions
    model_descriptions = {
        'Passive Aggressive Classifier': "Best for online learning scenarios with 92.11% accuracy",
        'Random Forest': "Ensemble learning model with 89.42% accuracy",
        'Support Vector Machine': "Linear SVM with 93.05% accuracy",
        'Naive Bayes': "Probabilistic classifier with 85.24% accuracy"
    }

    st.sidebar.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-top: 20px;'>
            <h4>Model Info</h4>
            <p style='color: rgba(255,255,255,0.8);'>
                {model_descriptions[model_choice]}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main content
    st.markdown("""
        <div class="title-container">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔍</div>
            <h1 class="stTitle">Fake News Detective</h1>
            <p style="color: rgba(255,255,255,0.8); margin-top: 1rem;">
                Advanced AI-powered news verification system
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,6,1])
    
    with col2:
        st.subheader("📝 Enter News Content")
        news_text = st.text_area(
            "Paste the news article here...",
            height=200,
            help="Copy and paste the news article you want to verify"
        )
        
        if news_text:
            words, sentences, chars = get_text_stats(news_text)
            st.markdown(f"""
                <div class="stats-container">
                    <h4>📊 Text Statistics</h4>
                    <p>Words: {words} | Sentences: {sentences} | Characters: {chars}</p>
                </div>
            """, unsafe_allow_html=True)
        
        predict_button = st.button("🔍 Analyze News")
        
        if predict_button:
            if news_text:
                with st.spinner(''):
                    st.markdown("""
                        <div class="loading-spinner">
                            🔄 Analyzing with {} model...
                        </div>
                    """.format(model_choice), unsafe_allow_html=True)
                    
                    # Get prediction
                    prediction = fake_news_detect(news_text, models[model_choice], vector_load)
                    time.sleep(1)  # For effect
                
                if prediction == 'FAKE':
                    st.markdown("""
                        <div class='result-box fake-news'>
                            ❌ This appears to be Fake News!
                            <br><small>Predicted by {} model</small>
                        </div>
                    """.format(model_choice), unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class='result-box real-news'>
                            ✅ This appears to be Real News!
                            <br><small>Predicted by {} model</small>
                        </div>
                    """.format(model_choice), unsafe_allow_html=True)
            else:
                st.error('⚠️ Please enter some news content to verify')

        # Add ensemble prediction button
        ensemble_button = st.button("🎯 Analyze with All Models")

        if ensemble_button:
            if news_text:
                with st.spinner('Analyzing with all models...'):
                    predictions, final_prediction, fake_count, real_count = get_ensemble_prediction(
                        news_text, models, vector_load
                    )
                    
                    # Create results visualization
                    fig = go.Figure()
                    
                    # Add bar for each model prediction
                    for model_name, pred in predictions.items():
                        fig.add_trace(go.Bar(
                            name=model_name,
                            x=[model_name],
                            y=[1],
                            marker_color='#ff3366' if pred == 'FAKE' else '#00ff95'
                        ))
                    
                    fig.update_layout(
                        title={
                            'text': "Model Predictions",
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': {'size': 24, 'color': 'white'}
                        },
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(t=60, b=40, l=40, r=40),
                        font=dict(color='white', size=14),
                        xaxis=dict(
                            showgrid=False,
                            showline=True,
                            linecolor='rgba(255,255,255,0.2)'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.1)',
                            showline=True,
                            linecolor='rgba(255,255,255,0.2)'
                        )
                    )
                    
                    # Add hover effects
                    fig.update_traces(
                        hoverinfo='name+y',
                        hoverlabel=dict(
                            bgcolor='rgba(0,0,0,0.8)',
                            font_size=14,
                            font_family="Arial"
                        )
                    )
                    
                    # Show final prediction
                    if final_prediction == 'FAKE':
                        st.markdown("""
                            <div class='result-box fake-news'>
                                ❌ Majority Verdict: This appears to be Fake News!
                                <br><small>{} out of {} models predict this is FAKE</small>
                            </div>
                        """.format(fake_count, len(predictions)), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class='result-box real-news'>
                                ✅ Majority Verdict: This appears to be Real News!
                                <br><small>{} out of {} models predict this is REAL</small>
                            </div>
                        """.format(real_count, len(predictions)), unsafe_allow_html=True)
                    
                    # Display the visualization
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show individual model predictions
                    st.markdown("<h4>Individual Model Predictions:</h4>", unsafe_allow_html=True)
                    for model_name, pred in predictions.items():
                        color = '#ff3366' if pred == 'FAKE' else '#00ff95'
                        st.markdown(f"""
                            <div style='padding: 10px; margin: 5px 0; border-radius: 5px; background: rgba(255,255,255,0.1);'>
                                <span style='color: {color}'>{'❌' if pred == 'FAKE' else '✅'}</span>
                                <strong>{model_name}:</strong> {pred}
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.error('⚠️ Please enter some news content to verify')

if __name__ == '__main__':
    main()