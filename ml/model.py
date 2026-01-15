import pandas as pd
import numpy as np
import string
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

#General Class for preprocessing
class TicketClassifier:
    def __init__(self):

        # =========================
        # Models
        # =========================
        self.model_queue = LinearSVC(C=1.0)
        self.model_priority = LinearSVC(C=1.0)
        self.model_language = LinearSVC(C=1.0)

        # =========================
        # Vectorizers (SEPARATE!)
        # =========================

        # For queue & priority (semantic understanding)
        self.vectorizer_text = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3
        )

        # For language detection (character patterns)
        self.vectorizer_lang = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=5
        )

        # Stopwords ONLY for semantic tasks
        self.stop_words = set(
            stopwords.words("english")
        ).union(
            stopwords.words("german")
        )

    # =========================
    # Shared text builder
    # =========================
    def _build_text(self, x):
        x = x[['subject', 'body']].dropna().copy()
        x['text'] = x['subject'] + ' ' + x['body']
        return x['text']

    # =========================
    # Preprocessing for QUEUE & PRIORITY
    # =========================
    def _preprocess_semantic(self, texts, fit=False):

        # lowercase
        texts = texts.str.lower()

        # remove punctuation
        texts = texts.apply(
            lambda t: t.translate(
                str.maketrans('', '', string.punctuation)
            )
        )

        # remove numbers
        texts = texts.apply(
            lambda t: ''.join(ch for ch in t if not ch.isdigit())
        )

        # remove stopwords
        texts = texts.apply(
            lambda t: ' '.join(
                w for w in t.split() if w not in self.stop_words
            )
        )

        if fit:
            return self.vectorizer_text.fit_transform(texts)
        return self.vectorizer_text.transform(texts)

    # =========================
    # Preprocessing for LANGUAGE
    # =========================
    def _preprocess_language(self, texts, fit=False):

        # lowercase ONLY (do NOT remove stopwords!)
        texts = texts.str.lower()

        if fit:
            return self.vectorizer_lang.fit_transform(texts)
        return self.vectorizer_lang.transform(texts)

    # =========================
    # Train
    # =========================
    def train(self, x, y_queue, y_language, y_priority):

        texts = self._build_text(x)

        # Semantic features
        X_sem = self._preprocess_semantic(texts, fit=True)

        # Language features
        X_lang = self._preprocess_language(texts, fit=True)

        # Train models
        self.model_queue.fit(X_sem, y_queue)
        self.model_priority.fit(X_sem, y_priority)
        self.model_language.fit(X_lang, y_language)

        return self

    # =========================
    # Predict
    # =========================
    def predict(self, x):

        texts = self._build_text(x)

        X_sem = self._preprocess_semantic(texts, fit=False)
        X_lang = self._preprocess_language(texts, fit=False)

        queue = self.model_queue.predict(X_sem).tolist()
        priority = self.model_priority.predict(X_sem).tolist()
        language = self.model_language.predict(X_lang).tolist()

        return {
            "Queue": queue,
            "Language": language,
            "Priority": priority
        }