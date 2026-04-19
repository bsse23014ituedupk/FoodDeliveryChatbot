"""
intents.py
----------
Defines all chatbot intents, their training patterns, and
the candidate responses the bot can return.

Each key in INTENTS is an intent label.
Each value is a dict with two lists:
    "patterns"  – example user utterances used for training
    "responses" – possible bot replies (one is chosen randomly at runtime)

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

# ─────────────────────────────────────────────────────────────
# TASK 3: Define Chatbot Intents
# ─────────────────────────────────────────────────────────────
INTENTS: dict[str, dict] = {

    # ── Greeting ─────────────────────────────────────────────
    "greeting": {
        "patterns": [
            "hello",
            "hi there",
            "hey",
            "good morning",
            "good evening",
            "good afternoon",
            "howdy",
            "hi",
            "greetings",
            "what's up",
            "sup",
            "hello there",
            "yo",
        ],
        "responses": [
            "👋 Hello! Welcome to QuickBite Food Delivery! How can I help you today?",
            "Hi there! 😊 Ready to satisfy your cravings? What would you like to order?",
            "Hey! Great to see you at QuickBite! Shall we get your food ordered?",
            "Good day! I'm your QuickBite assistant. What can I do for you?",
        ],
    },

    # ── Order Food ───────────────────────────────────────────
    "order_food": {
        "patterns": [
            "I want to order food",
            "can I place an order",
            "order a pizza",
            "I would like to order",
            "place my order",
            "get me some food",
            "I want to buy food",
            "can you take my order",
            "I need to order something",
            "make an order",
            "add to cart",
            "order burger",
            "I want pizza",
            "get me a burger",
            "order some biryani",
            "I want to eat",
            "food order",
            "place order",
        ],
        "responses": [
            "🍕 Great choice! Please visit our app or website to complete your order, or tell me what you'd like and I'll guide you through it!",
            "✅ Sure! What would you like to order? We have pizza, burgers, biryani, fries, and much more!",
            "🛒 Your order is being set up! Could you please specify the items you'd like?",
            "😋 Let's get that order in! Just let me know what you're craving today.",
        ],
    },

    # ── Menu ─────────────────────────────────────────────────
    "menu": {
        "patterns": [
            "what is on the menu",
            "show me the menu",
            "what food do you have",
            "what can I order",
            "list of items",
            "what do you serve",
            "available dishes",
            "food options",
            "what meals are available",
            "do you have biryani",
            "do you serve pizza",
            "see menu",
            "browse menu",
            "menu please",
            "what food is available",
        ],
        "responses": [
            (
                "📋 Here's our delicious menu:\n"
                "  🍕 Pizza        – Margherita, BBQ Chicken, Pepperoni\n"
                "  🍔 Burgers      – Classic, Zinger, Mushroom Swiss\n"
                "  🍛 Biryani      – Chicken, Beef, Vegetable\n"
                "  🍟 Sides        – French Fries, Garlic Bread, Coleslaw\n"
                "  🥗 Salads       – Caesar, Greek, Garden Fresh\n"
                "  🧃 Beverages    – Coke, Juice, Water, Lassi\n"
                "  🍰 Desserts     – Brownie, Cheesecake, Ice Cream\n"
                "What would you like to try?"
            ),
            (
                "🍽️ Our menu highlights:\n"
                "  • Pizzas starting at Rs. 499\n"
                "  • Burgers from Rs. 299\n"
                "  • Biryani deals from Rs. 350\n"
                "  • Combo meals from Rs. 599\n"
                "Visit our app for the full menu with pictures!"
            ),
        ],
    },

    # ── Delivery Time ────────────────────────────────────────
    "delivery_time": {
        "patterns": [
            "how long will delivery take",
            "delivery time",
            "when will my order arrive",
            "estimated delivery",
            "how long does it take",
            "when will food arrive",
            "time for delivery",
            "how fast is delivery",
            "is delivery fast",
            "how many minutes",
            "ETA",
            "how soon can I get food",
            "track my order",
            "where is my order",
            "order status",
        ],
        "responses": [
            "⏱️ Our estimated delivery time is **30–45 minutes** depending on your location and current order volume.",
            "🚴 Your order is on its way! Typical delivery takes **25–40 minutes**. You can track your rider live in our app.",
            "📍 Delivery usually takes **30 minutes**. During peak hours it may take up to 50 minutes. We appreciate your patience!",
            "🕐 We aim to deliver within **30–45 minutes**. For real-time updates, use the tracking feature in our app.",
        ],
    },

    # ── Payment ──────────────────────────────────────────────
    "payment": {
        "patterns": [
            "how can I pay",
            "payment options",
            "do you accept credit card",
            "can I pay with cash",
            "payment methods",
            "is online payment available",
            "can I pay online",
            "accept debit card",
            "JazzCash payment",
            "Easypaisa payment",
            "cash on delivery",
            "COD",
            "pay by card",
            "payment gateway",
            "billing",
            "how to pay",
            "I want to pay online",
        ],
        "responses": [
            (
                "💳 We support the following payment methods:\n"
                "  ✅ Cash on Delivery (COD)\n"
                "  ✅ Credit / Debit Card (Visa, Mastercard)\n"
                "  ✅ JazzCash\n"
                "  ✅ Easypaisa\n"
                "  ✅ Bank Transfer\n"
                "Choose whichever is most convenient for you!"
            ),
            "💰 You can pay via Cash on Delivery, JazzCash, Easypaisa, or any major Credit/Debit card. All transactions are 100% secure!",
            "🔒 We accept Cash on Delivery as well as digital payments via JazzCash, Easypaisa, and Credit/Debit cards for your convenience.",
        ],
    },

    # ── Contact / Support ────────────────────────────────────
    "contact": {
        "patterns": [
            "how can I contact you",
            "customer support",
            "help",
            "contact number",
            "support email",
            "reach customer service",
            "I need help",
            "talk to agent",
            "live support",
            "phone number",
            "email address",
            "where are you located",
            "contact details",
            "complaint",
            "report a problem",
            "issue with order",
            "call support",
        ],
        "responses": [
            (
                "📞 You can reach QuickBite Support through:\n"
                "  📱 Phone    : 0300-1234567 (9AM – 11PM)\n"
                "  📧 Email    : support@quickbite.pk\n"
                "  💬 Live Chat: Available in our app\n"
                "  🌐 Website  : www.quickbite.pk\n"
                "We're happy to assist you!"
            ),
            "🆘 Need help? Call us at **0300-1234567** or email **support@quickbite.pk**. Our team is available from 9AM to 11PM.",
            "💬 For any concerns, reach us via Live Chat in the app or call **0300-1234567**. We'll resolve your issue ASAP!",
        ],
    },
}
