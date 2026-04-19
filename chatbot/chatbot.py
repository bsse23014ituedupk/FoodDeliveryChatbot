"""
chatbot.py
----------
High-level ``ChatBot`` class that glues together the ML model and the
interactive terminal loop.

Responsibilities
----------------
• ``get_response()``  – uses IntentModel to pick the best reply.
• ``run()``           – the REPL loop (Tasks 8 & 9).

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

import random

from .config import BOT_NAME, CONFIDENCE_THRESHOLD, DEBUG_MODE
from .intents import INTENTS
from .model import IntentModel


# ─────────────────────────────────────────────────────────────
# TASK 8 & 9: Response Logic + Interactive Loop
# ─────────────────────────────────────────────────────────────
class ChatBot:
    """
    QuickBite Food Delivery Chatbot.

    Parameters
    ----------
    verbose : bool
        Whether to print training progress when the model is initialised.

    Examples
    --------
    >>> bot = ChatBot()
    >>> bot.run()
    """

    _EXIT_KEYWORDS: frozenset[str] = frozenset({"quit", "exit", "bye", "goodbye"})

    def __init__(self, verbose: bool = True) -> None:
        self._model = IntentModel()
        self._model.train(verbose=verbose)

    # ──────────────────────────────────────────────────────────
    # Task 8 – Response function
    # ──────────────────────────────────────────────────────────
    def get_response(self, user_input: str) -> str:
        """
        Return an appropriate response for *user_input*.

        • Predicts the intent via the ML model.
        • If confidence ≥ threshold → pick a random reply for that intent.
        • Otherwise                → return a polite fallback message.

        Parameters
        ----------
        user_input : str
            Raw text entered by the user.

        Returns
        -------
        str
            The bot's reply.
        """
        intent, confidence = self._model.predict(user_input)

        if DEBUG_MODE:
            print(
                f"   [DEBUG] Predicted intent : {intent}"
                f"  |  Confidence : {confidence:.2%}"
            )

        if confidence >= CONFIDENCE_THRESHOLD:
            return random.choice(INTENTS[intent]["responses"])

        return (
            "🤔 I'm not quite sure I understood that. Could you please rephrase?\n"
            "   You can ask me about: menu, order food, delivery time, payment, or contact."
        )

    # ──────────────────────────────────────────────────────────
    # Task 9 – Interactive chat loop
    # ──────────────────────────────────────────────────────────
    def run(self) -> None:
        """
        Start the interactive terminal chat loop.

        The loop runs until the user types one of the exit keywords
        (``quit``, ``exit``, ``bye``, ``goodbye``) or sends EOF / Ctrl-C.
        """
        self._print_banner()

        while True:
            try:
                user_input = input("You   : ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n\nBot   : Goodbye! Thank you for using {BOT_NAME}. 🙏")
                break

            if not user_input:
                print("Bot   : Please type something. I'm here to help! 😊\n")
                continue

            if user_input.lower() in self._EXIT_KEYWORDS:
                print(
                    f"Bot   : 👋 Goodbye! Thank you for using {BOT_NAME} Food Delivery."
                    " Have a great meal! 🍽️\n"
                )
                break

            response = self.get_response(user_input)
            print(f"\nBot   : {response}\n")

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _print_banner() -> None:
        """Print the welcome banner to stdout."""
        print("\n" + "=" * 60)
        print(f"  🤖  {BOT_NAME} Food Delivery Chatbot  🍔🍕🛵")
        print("=" * 60)
        print("  Type your message and press Enter.")
        print("  Type 'quit' to exit the chatbot.")
        print("=" * 60 + "\n")
