"""
preprocessor.py
---------------
Text preprocessing utilities for the QuickBite Food Delivery Chatbot.

Responsible for cleaning and normalising raw user input before it is
passed to the TF-IDF vectorizer and Naive Bayes classifier.

============================================================
  Class Activity-6: Food Delivery Chatbot (Machine Learning)
  Course   : Human Computer Interaction (HCI)
  Subject  : SE305T & MD445T (Spring-26)
  Student  : Abdul Samad
  Roll No  : BSSE23014
  Date     : 16-April-2026
  Instructor: Dr. Muhammad Asif
  Institute: Information Technology University (ITU) – Lahore
============================================================
"""

import re
import nltk

# Download required NLTK resources (no-op if already present)
nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)


# ─────────────────────────────────────────────────────────────
# TASK 2: Preprocessing Function
# ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Normalise raw text before feature extraction.

    Steps
    -----
    1. Convert to lowercase.
    2. Remove non-alphanumeric characters (keeps spaces).
    3. Collapse multiple internal spaces / leading-trailing whitespace.

    Parameters
    ----------
    text : str
        Raw user input or training pattern.

    Returns
    -------
    str
        Cleaned, lowercase, whitespace-normalised text.
    """
    text = text.lower()                       # 1. lowercase
    text = re.sub(r"[^a-z0-9\s]", "", text)  # 2. remove punctuation / special chars
    text = " ".join(text.split())             # 3. normalise whitespace
    return text
