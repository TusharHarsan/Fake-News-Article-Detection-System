# Fake News Detection App

This web application allows users to classify news articles as either real or fake using a pre-trained machine learning model. It leverages natural language processing (NLP) techniques to process text data and predict the authenticity of news content.

![App Screenshot](image.jpg)

## Features

- **Input News Title Selection**: Users can select a news title from a dropdown menu to view its content.
- **Text Analysis**: The selected news content is analyzed using NLP methods to preprocess and vectorize the text.
- **Prediction**: Based on the processed text, the app predicts whether the news is real or fake using a trained machine learning model.
- **User Interface**: Built with Streamlit, the app offers an intuitive interface for easy interaction and understanding.

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Murasajo/Fake-News-Detection-App.git
   cd fake-news-classification-app
   ```

2. **Install dependencies:**

   Ensure you have Python 3.x installed. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   Start the Streamlit app locally:

   ```bash
   streamlit run app.py
   ```

   Open your web browser and go to `https://fake-news-detection-app-46bmwaxx4vesqj9dapp2ged.streamlit.app/`.

## Usage

- Select a news title from the dropdown menu.
- View the news content in the text area.
- Click the "Predict" button to classify the news as either fake or real.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- NLTK

## Dataset and Model

- **Dataset**: The app uses a sample dataset (`news.csv`) containing news titles and content for classification.
- **Model**: A pre-trained machine learning model (`pac_model.pkl`) is used for predicting the authenticity of news articles.

## Contributing

Contributions are welcome! Here's how you can contribute to the project:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.


## Acknowledgments

- This project was inspired by the need to combat misinformation and promote media literacy.
- Special thanks to [Streamlit](https://streamlit.io) for simplifying the creation of interactive web applications in Python.

