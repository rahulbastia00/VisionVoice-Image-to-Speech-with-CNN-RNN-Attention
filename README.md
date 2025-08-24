# 🌟 Image Captioning with CNN-RNN and Attention Mechanism

---

## 📖 1. Purpose & Overview

This project implements an end-to-end deep learning model to **automatically generate descriptive captions for images**. It combines:

* 🖼 **Convolutional Neural Network (CNN)** for image feature extraction
* ✍ **Recurrent Neural Network (RNN)** with an **Attention Mechanism** for sequential text generation

The **motivation** is to explore the synergy between computer vision and natural language processing. Using attention, the model dynamically focuses on the most relevant parts of an image while generating each word, leading to **more contextually accurate captions**.

👉 Ideal for developers and researchers interested in **image understanding, multimodal learning, and attention mechanisms**.

---

## ✨ 2. Highlights & Features

* ⚡ **Encoder-Decoder Architecture:** Pre-trained InceptionV3 (encoder) + GRU-based RNN (decoder)
* 🎯 **Bahdanau Attention Mechanism:** Focuses on specific image regions per word
* 🔊 **Text-to-Speech (gTTS):** Converts generated captions into speech
* 👀 **Attention Visualization:** Visualizes which image regions influence each word
* 📂 **Trained on Flickr8k:** 8,000 images, each with 5 captions

---

## ⚙️ 3. Installation & Setup

### 🔹 Step 1: Clone Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 🔹 Step 2: Create Virtual Environment

```bash
# Unix/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 🔹 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Step 4: Download Dataset

* Requires **Kaggle API token (`kaggle.json`)** in `~/.kaggle/`
* Or download manually → [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
* Place in directory:

```
.
├── dataset/
│   ├── Images/
│   └── captions.txt
```

---

## 🚀 4. Usage Examples

### 🔹 Train Model

Run the Jupyter Notebook:

```
training_cnn_rnn_odel.ipynb
```

This handles **data preprocessing, training, and model saving**.

### 🔹 Inference with Pre-Trained Models

```python
import tensorflow as tf
import pickle
from PIL import Image

# Load Models
encoder = tf.keras.models.load_model('models/encoder.keras')
decoder = tf.keras.models.load_model('models/decoder.keras')
image_features_extract_model = tf.keras.models.load_model('models/image_features_extract_model.keras')

with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define Helper Functions
def load_images(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def evaluate(image_path, max_length=31):
    hidden = decoder.init_state(batch_size=1)
    temp_input = tf.expand_dims(load_images(image_path)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()
        word = tokenizer.index_word[predicted_id]
        result.append(word)
        if word == '<end>':
            return ' '.join(result)
        dec_input = tf.expand_dims([predicted_id], 0)

    return ' '.join(result)

# Predict Caption
image_path = 'path/to/your/image.jpg'
print("Predicted Caption:", evaluate(image_path))
```

---

## 🏗️ 5. Architecture Diagram & Explanation

Below is the **model architecture diagram** that illustrates the entire pipeline:

<p align="center">
  <img src="doc/architecture_diagram.png" alt="Architecture Diagram" width="700"/>
</p>

### 🔹 Workflow Explanation

1. **CNN Encoder (InceptionV3):** Extracts image features.
2. **Feature Transformation:** Converts extracted features into a suitable representation.
3. **RNN Decoder with GRU:** Generates word sequences step-by-step.
4. **Attention Mechanism:** Highlights relevant image regions per generated word.
5. **Output:** Caption text (and optional text-to-speech).

---

## 📂 6. Folder Structure

```
.
├── doc/
│   ├── architecture_diagram.png    # Model architecture diagram
│   └── PROJECT_DOC.md              # Extended documentation
├── models/
│   ├── attention.keras
│   ├── decoder.keras
│   ├── encoder.keras
│   ├── image_features_extract_model.keras
│   ├── model_config.json
│   └── tokenizer.pkl
├── venv/
├── requirements.txt
├── .gitignore
└── training_cnn_rnn_odel.ipynb
```

---

## 🛠 7. Technologies & Requirements

* **Framework:** TensorFlow 2.x
* **Python:** 3.8+
* **Libraries:**

  * NumPy, Pandas
  * Matplotlib, Seaborn
  * gTTS (text-to-speech)
  * Pillow
  * WordCloud
  * scikit-learn
  * NLTK (BLEU score)

---

## 🔧 8. Configuration

Stored in `models/model_config.json`:

* `embedding_dim`: Word embedding size
* `units`: GRU units
* `vocab_size`: Vocabulary size
* `max_length`: Caption max length

---

## 🤝 9. Contributing

1. Fork the repo
2. Create branch → `feature/your-feature`
3. Commit → `git commit -m "Add feature"`
4. Push → `git push origin feature/your-feature`
5. Open Pull Request 🎉

---

## 📜 10. License

Licensed under **MIT License**. See [LICENSE](LICENSE).

---

## 🙏 11. Acknowledgements & Credits

* **Dataset:** Flickr8k creators
* **Pretrained Model:** InceptionV3 (Google)

---

## 📑 12. Advanced Documentation

See:

* `doc/PROJECT_DOC.md`
* `doc/architecture_diagram.png`
