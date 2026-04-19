"""
main.py
-------
Entry point for the QuickBite Food Delivery Chatbot.

Run from the project root:
    python main.py

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

from chatbot import ChatBot


def main() -> None:
    """Initialise the chatbot and start the interactive session."""
    print("=" * 60)
    print("  Food Delivery Chatbot – Abdul Samad (BSSE23014)")
    print("  HCI – Class Activity 6 | ITU Lahore | Spring-26")
    print("=" * 60)

    bot = ChatBot(verbose=True)
    bot.run()


if __name__ == "__main__":
    main()
