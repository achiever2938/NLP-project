
# Sentiment Analysis of Amazon Product Reviews

This project utilizes Natural Language Processing (NLP) and Deep Learning models to perform sentiment analysis 
on Amazon product reviews, providing insights into customer feedback. 
It includes multiple machine learning and deep learning approaches for classifying reviews 
as positive, neutral, or negative.

---

## Features

- **Data Loading and Preprocessing:**
  - Cleaning and tokenizing review text.
  - Removing stopwords and preparing data for model input.
- **Exploratory Data Analysis (EDA):**
  - Visualizations of sentiment distributions, word clouds, and review patterns.
- **Model Development:**
  - Feedforward Neural Network (FFNN).
  - Long Short-Term Memory (LSTM).
  - Transformer-based models like BERT.
- **Interactive Predictions:**
  - Gradio-based interface for real-time sentiment predictions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone
   cd sentiment-analysis
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

---

## Usage

1. Open the `sentiment_analysis.ipynb` notebook.
2. Execute cells sequentially to:
   - Load and preprocess the dataset.
   - Conduct exploratory data analysis.
   - Train and evaluate machine learning models.
   - Launch the interactive sentiment analysis app.

---

## Workflow Overview

### 1. Data Preprocessing
- Load the dataset from a CSV file.
- Map ratings to sentiment categories:
  - 1–2 stars → **Negative**
  - 3 stars → **Neutral**
  - 4–5 stars → **Positive**.
- Clean text data by:
  - Lowercasing.
  - Removing special characters.
  - Tokenizing and removing stopwords.

### 2. Exploratory Data Analysis (EDA)
- Analyze sentiment distributions.
- Generate word clouds for each sentiment.
- Plot frequency distributions and review lengths.

### 3. Model Development
- **Feedforward Neural Network (FFNN):**
  - Simple architecture for baseline sentiment classification.
- **Long Short-Term Memory (LSTM):**
  - Sequential model for processing word embeddings.
- **BERT-Based Models:**
  - Fine-tuned transformers for state-of-the-art performance.

### 4. Evaluation
- Assess models with metrics like precision, recall, F1-score, and accuracy.
- Compare performance across models.

### 5. Interactive Predictions
- Use the Gradio interface to input reviews and get predicted sentiments.

---

## Visualizations

### Sentiment Distribution
```python
sns.countplot(data=df, x='sentiment', palette='Set2')
plt.title('Sentiment Distribution')
plt.show()
```

### Word Clouds
- Generate word clouds for **positive**, **neutral**, and **negative** sentiments using the `WordCloud` library.

### Model Accuracy Comparison
```python
models = ['FFNN', 'LSTM', 'BERT']
accuracies = [93.17, 95.0, 96.0]
plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.show()
```

---

## Example Code

### Model Definition (FFNN)
```python
class SentimentNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
```

### Interactive Gradio App
```python
import gradio as gr

def predict_sentiment(review):
    # Preprocess review
    processed_review = preprocess_text(review)
    sentiment = model.predict(processed_review)
    return sentiment

iface = gr.Interface(fn=predict_sentiment, inputs="text", outputs="text", title="Sentiment Analysis")
iface.launch()
```

---

## Dependencies

Install the following Python libraries:
- `pandas`
- `numpy`
- `torch`
- `scikit-learn`
- `transformers`
- `matplotlib`
- `seaborn`
- `gradio`
- `wordcloud`

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---
