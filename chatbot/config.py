"""
config.py
---------
Central configuration file for the QuickBite Food Delivery Chatbot.

All tuneable constants live here so you never have to hunt through
business-logic files to adjust thresholds or model hyper-parameters.

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

# ──────────────────────────────────────────────
# Bot identity
# ──────────────────────────────────────────────
BOT_NAME: str = "QuickBite"

# ──────────────────────────────────────────────
# TF-IDF vectorizer settings
# ──────────────────────────────────────────────
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)   # unigrams + bigrams
TFIDF_ANALYZER: str = "word"
TFIDF_SUBLINEAR_TF: bool = True                # apply log(1+tf) scaling

# ──────────────────────────────────────────────
# Naive Bayes classifier settings
# ──────────────────────────────────────────────
NB_ALPHA: float = 0.1   # Laplace smoothing factor

# ──────────────────────────────────────────────
# Response logic
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.25  # min confidence to give a domain response

# ──────────────────────────────────────────────
# Debug / logging
# ──────────────────────────────────────────────
DEBUG_MODE: bool = True   # set False to suppress [DEBUG] lines in production
