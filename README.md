# 🍔 QuickBite Food Delivery Chatbot

> **Class Activity 6 – Human Computer Interaction (HCI)**  
> Subject: SE305T & MD445T (Spring-26) · ITU Lahore  
> Student: Abdul Samad · Roll No: BSSE23014  
> Instructor: Dr. Muhammad Asif

---

## 📌 Overview

QuickBite is a **Machine Learning–powered** food delivery chatbot built with Python.  
It uses **TF-IDF feature extraction** and a **Multinomial Naïve Bayes** classifier to understand user intent and respond conversationally.

### ✅ Supported Intents

| Intent | Example Queries |
|---|---|
| `greeting` | *"Hello"*, *"Hi there"*, *"Good morning"* |
| `order_food` | *"I want to order a pizza"*, *"Place my order"* |
| `menu` | *"Show me the menu"*, *"What food do you have?"* |
| `delivery_time` | *"When will my order arrive?"*, *"ETA?"* |
| `payment` | *"Do you accept JazzCash?"*, *"Cash on delivery?"* |
| `contact` | *"Customer support number"*, *"I have a complaint"* |

---

## 🗂️ Project Structure

```
food_delivery_chatbot/
│
├── chatbot/                  # Core Python package
│   ├── __init__.py           # Exposes ChatBot publicly
│   ├── config.py             # All tuneable constants & hyper-parameters
│   ├── intents.py            # Intent patterns and bot responses (Task 3)
│   ├── preprocessor.py       # Text cleaning / normalisation (Task 2)
│   ├── model.py              # TF-IDF + Naïve Bayes pipeline (Tasks 4-7)
│   └── chatbot.py            # Response logic & interactive loop (Tasks 8-9)
│
├── main.py                   # ▶ Entry point – run this file
├── requirements.txt          # Python dependencies
└── README.md                 # You are here
```

---

## ⚙️ Setup & Installation

### 1 · Clone / download the project

```bash
# If using Git
git clone <repo-url>
cd food_delivery_chatbot
```

### 2 · Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

> **First run** will automatically download the required NLTK data files (`punkt`, `stopwords`).

---

## ▶️ Running the Chatbot

```bash
python main.py
```

**Example session:**

```
============================================================
  Food Delivery Chatbot – Abdul Samad (BSSE23014)
  HCI – Class Activity 6 | ITU Lahore | Spring-26
============================================================

  🤖  QuickBite Food Delivery Chatbot  🍔🍕🛵
============================================================
  Type your message and press Enter.
  Type 'quit' to exit the chatbot.
============================================================

You   : show me the menu
   [DEBUG] Predicted intent : menu  |  Confidence : 98.73%

Bot   : 📋 Here's our delicious menu:
  🍕 Pizza        – Margherita, BBQ Chicken, Pepperoni
  🍔 Burgers      – Classic, Zinger, Mushroom Swiss
  ...

You   : quit
Bot   : 👋 Goodbye! Thank you for using QuickBite Food Delivery. Have a great meal! 🍽️
```

---

## 🔧 Configuration

All tuneable parameters are in **`chatbot/config.py`** — no need to touch any other file.

| Constant | Default | Description |
|---|---|---|
| `BOT_NAME` | `"QuickBite"` | Display name of the bot |
| `TFIDF_NGRAM_RANGE` | `(1, 2)` | Unigram + bigram features |
| `NB_ALPHA` | `0.1` | Laplace smoothing for Naïve Bayes |
| `CONFIDENCE_THRESHOLD` | `0.25` | Min confidence to use a domain reply |
| `DEBUG_MODE` | `True` | Print `[DEBUG]` lines during chat |

---

## 🧠 ML Pipeline (Task Summary)

| Task | Module | Description |
|---|---|---|
| Task 1 | `requirements.txt` | Environment setup |
| Task 2 | `preprocessor.py` | Lowercase, strip punctuation, normalise whitespace |
| Task 3 | `intents.py` | Define 6 intents with patterns & responses |
| Task 4 | `model.py → _build_corpus()` | Flatten patterns into training corpus |
| Task 5 | `model.py → train()` | TF-IDF vectorization (unigrams + bigrams) |
| Task 6 | `model.py → train()` | Train Multinomial Naïve Bayes |
| Task 7 | `model.py → predict()` | Intent prediction + confidence score |
| Task 8 | `chatbot.py → get_response()` | Threshold check + random response selection |
| Task 9 | `chatbot.py → run()` | Interactive REPL loop |

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `nltk` | Tokenization & stopword resources |
| `scikit-learn` | TF-IDF vectorizer + Naïve Bayes classifier |
| `numpy` | Probability array operations |

---

## 📄 License

This project is submitted as academic coursework for **HCI – Class Activity 6** at  
**Information Technology University (ITU) Lahore, Pakistan**.  
© 2026 Abdul Samad (BSSE23014). All rights reserved.
