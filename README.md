Project 1 – AI Chatbot (Intent Classification with LSTM)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
texts = ["hi", "hello", "book ticket", "cancel reservation", "bye", "goodbye"]
labels = ["greet", "greet", "book", "cancel", "goodbye", "goodbye"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=4)

# Model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=16, input_length=4),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(len(set(labels)), activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
model.fit(X, y, epochs=30, verbose=0)

# Test
test_text = ["hello"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_seq = pad_sequences(test_seq, maxlen=4)
pred = model.predict(test_seq)
print("Predicted intent:", le.inverse_transform([np.argmax(pred)]))

Project 2 – French → English Translation (Seq2Seq LSTM)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Sample toy dataset
french = ["bonjour", "au revoir"]
english = ["hello", "goodbye"]

# Preprocess
input_tokens = set(" ".join(french).split())
target_tokens = set(" ".join(english).split())
input_dict = {w: i for i, w in enumerate(input_tokens)}
target_dict = {w: i for i, w in enumerate(target_tokens)}

# Encode data
encoder_input = np.array([[input_dict[w]] for w in french[0].split()])
decoder_input = np.array([[target_dict[w]] for w in english[0].split()])

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = tf.keras.layers.Embedding(len(input_tokens)+1, 8)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(16, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = tf.keras.layers.Embedding(len(target_tokens)+1, 8)(decoder_inputs)
decoder_lstm = LSTM(16, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(len(target_tokens)+1, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
print("Seq2Seq model built successfully")

Project 3 – Email Spam Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    "text": [
        "Win a free lottery now",
        "Hello, how are you?",
        "Get cheap meds online",
        "Meeting schedule attached"
    ],
    "label": ["spam", "ham", "spam", "ham"]
}
df = pd.DataFrame(data)

# Split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.25, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

Project 4 – Time Series Forecasting (ARIMA Example)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate synthetic data (e.g., sales)
np.random.seed(42)
data = np.cumsum(np.random.randn(100)) + 50
series = pd.Series(data)

# Fit ARIMA model
model = ARIMA(series, order=(2,1,2))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=10)
print("Next 10 predictions:", forecast)

# Plot
plt.plot(series, label="History")
plt.plot(range(len(series), len(series)+10), forecast, label="Forecast")
plt.legend()
plt.show()
