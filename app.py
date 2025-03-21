import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Load the trained model
with open('model/best_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract components from the model_data dictionary
model = model_data['model']
scaler = model_data.get('scaler')
feature_names = model_data.get('feature_names', [])

# Analyze text characteristics
def analyze_text(text):
    length = len(text)
    words = len(text.split())
    has_question = 'No question mark' if '?' not in text else 'Question mark detected'
    has_exclamation = 'No exclamation mark' if '!' not in text else 'Exclamation mark detected'
    uppercase_ratio = f'{sum(1 for char in text if char.isupper()) / max(1, length) * 100:.1f}% uppercase letters'
    
    return {
        'length': length,
        'word_count': words,
        'has_question_mark': has_question,
        'has_exclamation': has_exclamation,
        'uppercase_ratio': uppercase_ratio,
        'sentiment': get_sentiment_analysis(text)
    }

# Dummy sentiment analysis function (replace with actual implementation)
def get_sentiment_analysis(text):
    return {
        'positive': 0.0,
        'negative': 34.8,
        'neutral': 65.2,
        'overall': -0.4939,
        'explanation': 'Negative sentiment detected in words: mad'
    }

# Plot engagement patterns
def plot_engagement_patterns(date):
    """
    Create plots similar to the example but for Streamlit display.
    Uses dummy data since we don't have the actual videos_df.
    """
    # Get input date info
    input_day = date.strftime('%A')
    input_month = date.strftime('%B')
    
    # Dummy data (replace with actual data in production)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_rates = [0.034, 0.028, 0.031, 0.029, 0.038, 0.042, 0.036]
    
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    month_rates = [0.033, 0.029, 0.032, 0.035, 0.037, 0.039, 
                  0.041, 0.038, 0.036, 0.034, 0.031, 0.030]
    
    # Create figure for Streamlit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot engagement rate by day
    bars1 = ax1.bar(days, day_rates, color='skyblue')
    ax1.set_title('Engagement Rate by Upload Day\nwith Your Upload Day Highlighted', pad=20)
    ax1.set_xlabel('Day of the Week')
    ax1.set_ylabel('Engagement Rate')
    ax1.set_xticklabels(days, rotation=45)
    
    # Add vertical line for input day
    day_index = days.index(input_day)
    ax1.axvline(x=day_index, color='red', linestyle='--', alpha=0.7)
    
    # Add annotation for input day
    max_height = max(day_rates) * 1.1
    ax1.text(day_index, max_height, 'Your Upload Day', 
            rotation=0, ha='center', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Plot engagement rate by month
    bars2 = ax2.bar(months, month_rates, color='orange')
    ax2.set_title('Engagement Rate by Upload Month\nwith Your Upload Month Highlighted', pad=20)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Engagement Rate')
    ax2.set_xticklabels(months, rotation=45)
    
    # Add vertical line for input month
    month_index = months.index(input_month)
    ax2.axvline(x=month_index, color='red', linestyle='--', alpha=0.7)
    
    # Add annotation for input month
    max_height = max(month_rates) * 1.1
    ax2.text(month_index, max_height, 'Your Upload Month', 
            rotation=0, ha='center', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    
    # Return both the figure and the metrics
    return {
        'figure': fig,
        'day_rate': day_rates[day_index],
        'month_rate': month_rates[month_index],
        'day_average': sum(day_rates) / len(day_rates),
        'month_average': sum(month_rates) / len(month_rates)
    }

# Function to get simple prediction
def get_simple_prediction(title, date, keyword):
    # Prepare features
    features = {}
    
    # Add title features
    features['title_length'] = len(title)
    features['has_question'] = 1 if '?' in title else 0
    features['has_exclamation'] = 1 if '!' in title else 0
    
    # Add date features
    features['year'] = date.year
    features['month'] = date.month
    features['day'] = date.day
    features['day_of_week'] = date.weekday()
    
    # Add keyword/category features (simplified)
    features['keyword_length'] = len(keyword)
    
    # Create input array in the correct order based on feature_names
    input_array = []
    for feature in feature_names:
        if feature in features:
            input_array.append(features[feature])
        else:
            # Use 0 for missing features
            input_array.append(0)
    
    # Scale features if scaler is available
    if scaler:
        input_scaled = scaler.transform([input_array])
        prediction = model.predict(input_scaled)[0]
    else:
        prediction = model.predict([input_array])[0]
    
    # Convert prediction to percentage if needed
    prediction_percent = prediction * 100 if prediction < 1 else prediction
    
    # Analyze title
    title_analysis = analyze_text(title)
    
    # Get engagement patterns
    engagement_patterns = plot_engagement_patterns(date)
    
    # Return complete result
    return {
        'predicted_engagement': prediction_percent,
        'title_analysis': title_analysis,
        'engagement_patterns': engagement_patterns,
        'date': date,
        'keyword': keyword
    }

def main():
    st.set_page_config(
    page_title="YTE", layout="wide", initial_sidebar_state="auto"
    )
    st.title('YouTube Engagement Rate Predictor 📈')
    st.write('Enter video details to predict its engagement rate and analyze its characteristics.')
        
    # Display model information
    # st.subheader("Model Information")
    # st.write(f"Best model: {model_data.get('best_model_name', 'Unknown')}")
    # if feature_names:
    #     st.write(f"Features used: {', '.join(feature_names)}")
        
    main_col = st.columns(2)
    with main_col[0]:
        with st.container(border=True):
        # User input
            title = st.text_input('Enter video title:')
            
            # Date input
            col1, col2, col3 = st.columns(3)
            with col1:
                year = st.number_input('Year', min_value=2005, max_value=2025, value=2025)
            with col2:
                month = st.number_input('Month', min_value=1, max_value=12, value=3)
            with col3:
                day = st.number_input('Day', min_value=1, max_value=31, value=21)
                
            keyword = st.text_input('Enter keyword/category:')
            btn = st.button('Analyze')
    
    with main_col[1]:
        with st.container(border=True):
            if btn and title and keyword:
                try:
                    # Create date object
                    date = datetime(int(year), int(month), int(day))
                    
                    # Get prediction and analysis
                    result = get_simple_prediction(title, date, keyword)
                    
                    # Display results
                    st.header('Analysis Results 📌')
                    st.write(f"**Predicted Engagement Rate:** {result['predicted_engagement']:.2f}%")
                    
                    # Create tabs and put content inside each tab
                    tabs = st.tabs(['Title Analysis', 'Sentiment Analysis', 'Timing Analysis'])
                    
                    # Title Analysis Tab
                    with tabs[0]:
                        st.subheader('Title Analysis 📕:')
                        st.write(f"**Length:** {result['title_analysis']['length']} characters")
                        st.write(f"**Words:** {result['title_analysis']['word_count']}")
                        st.write(f"**Questions:** {result['title_analysis']['has_question_mark']}")
                        st.write(f"**Exclamations:** {result['title_analysis']['has_exclamation']}")
                        st.write(f"**Uppercase:** {result['title_analysis']['uppercase_ratio']}")

                    # Sentiment Analysis Tab
                    with tabs[1]:
                        st.subheader('Sentiment Analysis 🙂:')
                        sentiment = result['title_analysis']['sentiment']
                        st.write(f"**Positive:** {sentiment['positive']}%")
                        st.write(f"**Negative:** {sentiment['negative']}%")
                        st.write(f"**Neutral:** {sentiment['neutral']}%")
                        st.write(f"**Overall Score:** {sentiment['overall']}")
                        st.write(f"**Summary:** {sentiment['explanation']}")
                    
                    # Timing Analysis Tab
                    with tabs[2]:
                        st.subheader('Timing Analysis 🕒:')
                        patterns = result['engagement_patterns']
                        st.write(f"**Average engagement rate for {date.strftime('%A')}s:** {patterns['day_rate']:.4f}")
                        st.write(f"**Average engagement rate for {date.strftime('%B')}:** {patterns['month_rate']:.4f}")
                        st.write(f"**Overall daily average:** {patterns['day_average']:.4f}")
                        st.write(f"**Overall monthly average:** {patterns['month_average']:.4f}")
                        
                        # Display the engagement patterns visualization
                        st.subheader('Engagement Patterns Visualization:')
                        st.pyplot(patterns['figure'])
                    
                    # Feature Importance Tab
                    # with tabs[3]:
                    #     # Display feature importance if available
                    #     if 'feature_importance' in model_data and model_data['feature_importance'] is not None:
                    #         st.subheader('Feature Importance:')
                    #         importance_df = pd.DataFrame({
                    #             'Feature': feature_names,
                    #             'Importance': model_data['feature_importance']
                    #         }).sort_values('Importance', ascending=False)
                            
                    #         fig, ax = plt.subplots(figsize=(10, 6))
                    #         bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
                    #         ax.set_xlabel('Importance')
                    #         ax.set_title('Feature Importance')
                            
                    #         # Add value labels
                    #         for bar in bars:
                    #             width = bar.get_width()
                    #             ax.text(width + 0.001, bar.get_y() + bar.get_height()/2., 
                    #                     f'{width:.4f}',
                    #                     ha='left', va='center')
                            
                    #         st.pyplot(fig)
                    #     else:
                    #         st.info("Feature importance data is not available.")

                except Exception as e:
                    st.error(f'Error during prediction or analysis: {e}')
                    import traceback
                    st.code(traceback.format_exc())

if __name__ == '__main__':
    main()