"""
model.py
--------
ML pipeline: TF-IDF feature extraction + Multinomial Naive Bayes classifier.

The ``IntentModel`` class encapsulates:
    • Building the training corpus from the intents dictionary.
    • Fitting the TF-IDF vectorizer.
    • Training the Multinomial Naive Bayes classifier.
    • Exposing a ``predict`` method that returns (intent_label, confidence).

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

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from .config import (
    TFIDF_NGRAM_RANGE,
    TFIDF_ANALYZER,
    TFIDF_SUBLINEAR_TF,
    NB_ALPHA,
)
from .intents import INTENTS
from .preprocessor import preprocess


# ─────────────────────────────────────────────────────────────
# TASK 4-6: Prepare Data, Extract Features, Train Model
# ─────────────────────────────────────────────────────────────
class IntentModel:
    """
    Wraps the TF-IDF + Multinomial Naive Bayes intent-classification pipeline.

    Usage
    -----
    >>> model = IntentModel()
    >>> model.train(verbose=True)
    >>> intent, confidence = model.predict("Do you have pizza?")
    """

    def __init__(self) -> None:
        # Vectorizer (Task 5)
        self._vectorizer = TfidfVectorizer(
            ngram_range=TFIDF_NGRAM_RANGE,
            analyzer=TFIDF_ANALYZER,
            sublinear_tf=TFIDF_SUBLINEAR_TF,
        )

        # Classifier (Task 6)
        self._classifier = MultinomialNB(alpha=NB_ALPHA)
        self._label_encoder = LabelEncoder()

        # Training corpus (built in train())
        self._patterns: list[str] = []
        self._labels: list[str] = []

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────
    def _build_corpus(self) -> None:
        """Flatten intent patterns into parallel lists of texts and labels."""
        self._patterns.clear()
        self._labels.clear()

        for intent_name, intent_data in INTENTS.items():
            for pattern in intent_data["patterns"]:
                self._patterns.append(preprocess(pattern))
                self._labels.append(intent_name)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────
    def train(self, verbose: bool = True) -> None:
        """
        Build the training corpus, fit the vectorizer, and train the classifier.

        Parameters
        ----------
        verbose : bool
            If True, print training summary to stdout.
        """
        # Task 4 – build corpus
        self._build_corpus()
        if verbose:
            print(
                f"\n📊 Training data prepared: "
                f"{len(self._patterns)} patterns across {len(INTENTS)} intents.\n"
            )

        # Task 5 – TF-IDF feature extraction
        X_train = self._vectorizer.fit_transform(self._patterns)
        if verbose:
            print(f"✅ TF-IDF feature matrix shape : {X_train.shape}")
            print(f"   Vocabulary size              : {len(self._vectorizer.vocabulary_)}\n")

        # Task 6 – Encode labels and fit Naive Bayes
        y_train = self._label_encoder.fit_transform(self._labels)
        self._classifier.fit(X_train, y_train)

        if verbose:
            print("✅ Multinomial Naive Bayes model trained successfully!")
            print(f"   Classes: {list(self._label_encoder.classes_)}\n")

    def predict(self, user_input: str) -> tuple[str, float]:
        """
        Predict the intent label and confidence score for a user utterance.

        Parameters
        ----------
        user_input : str
            Raw (unprocessed) text from the user.

        Returns
        -------
        tuple[str, float]
            ``(predicted_intent, confidence_score)``
            where confidence_score is in the range [0, 1].
        """
        # Task 7 – prediction pipeline
        processed = preprocess(user_input)
        features  = self._vectorizer.transform([processed])

        predicted_index  = self._classifier.predict(features)[0]
        predicted_intent = self._label_encoder.inverse_transform([predicted_index])[0]

        probabilities    = self._classifier.predict_proba(features)[0]
        confidence       = float(np.max(probabilities))

        return predicted_intent, confidence
