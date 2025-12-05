import os
import sys

# --------------------------------------------------------------------
# Make backend/ visible so we can reuse encryption_module.py
# --------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(CURRENT_DIR, "..", "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from collections import defaultdict

import requests
from dotenv import load_dotenv
import gradio as gr
from huggingface_hub import InferenceClient

from encryption_module import (
    create_ckks_context,
    serialize_public_context,
    encrypt_answers,
    serialize_ciphertext,
    deserialize_ciphertext,
    decrypt_vector,
)

# --------------------------------------------------------------------
# Config & constants
# --------------------------------------------------------------------

# Disable Gradio telemetry + API footer
os.environ["GRADIO_SHOW_API"] = "False"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_API_INFO"] = "False"

BACKEND_URL = "http://127.0.0.1:5000/process"

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

if not HF_TOKEN:
    raise RuntimeError("Hugging Face API token not found.")

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

SYSTEM_PROMPT = """
Your name is Kira.
You are a calm, compassionate mental-health support companion for people in Saudi Arabia.

Rules:
- NEVER guess the user's name.
- NEVER invent personal details about the user.
- Refer to the user only as "you".
- Do NOT assume gender unless they tell you.
- Warm, respectful, non-judgmental tone.
- Use simple supportive English.
- Brief, empathetic responses.
- No medical advice or diagnosis.
- Encourage professional help when needed.
"""

CRISIS_REPLY = (
    "I’m really sorry you're feeling this way. You are not alone. 💛\n\n"
    "📞 **Saudi Arabia Crisis Contacts:**\n"
    "• Ministry of Health Hotline: **937** (24/7)\n"
    "• Saudi Red Crescent: **997**\n\n"
    "If you feel in immediate danger, please contact emergency services or someone you trust."
)

# --------------------------------------------------------------------
# Homomorphic encrypted stress check
# --------------------------------------------------------------------


def run_private_stress_check(q1, q2, q3, q4, q5):
    """
    Client-side HE workflow for one user:
    1. Generate CKKS context + keys.
    2. Encrypt answers and serialize to hex.
    3. Send encrypted vector + public context to the Flask backend.
    4. Receive encrypted score (still ciphertext).
    5. Decrypt score locally and map to a stress category.
    """

    # 1) Collect answers into a list
    answers = [q1, q2, q3, q4, q5]

    # 2) Context + key generation (client side only)
    context = create_ckks_context()
    context_hex = serialize_public_context(context)

    # 3) Encrypt answers and serialize to hex
    enc_vec = encrypt_answers(context, answers)
    ct_hex = serialize_ciphertext(enc_vec)

    payload = {
        "context_hex": context_hex,
        "ciphertext": ct_hex,
    }

    try:
        # 4) Send to backend
        resp = requests.post(BACKEND_URL, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Error contacting backend: {e}"

    resp_json = resp.json()
    if "encrypted_score" not in resp_json:
        # Backend may have returned an error message
        return f"Backend error: {resp_json}"

    enc_score_hex = resp_json["encrypted_score"]

    # 5) Rebuild encrypted score and decrypt it locally
    enc_score = deserialize_ciphertext(context, enc_score_hex)
    score_list = decrypt_vector(context, enc_score)
    # score_list is a 1-element vector
    score = float(score_list[0])

    # 6) Map numeric score to category (tune thresholds later)
    # You can calibrate these thresholds using your dataset.
    if score < 1.5:
        label = "Low stress"
    elif score < 3.0:
        label = "Moderate stress"
    else:
        label = "High stress"

    # 7) Build a user-friendly message
    result_text = (
        "🔐 Encrypted Stress Assessment\n\n"
        f"Approximate stress score: **{score:.2f}**\n"
        f"Category: **{label}**\n\n"
        "(Your answers were encrypted before sending to the server.\n"
        "The server only saw ciphertext and never had the key to decrypt it.)"
    )

    return result_text


# --------------------------------------------------------------------
# Static data (cities, patterns, etc.)
# --------------------------------------------------------------------

SAUDI_CITIES = [
    "riyadh",
    "jeddah",
    "dammam",
    "khobar",
    "mecca",
    "medina",
    "tabuk",
    "qassim",
    "abha",
    "jizan",
    "hail",
]

SAUDI_MENTAL_HEALTH = {
    "riyadh": {
        "clinic": "Erada & Mental Health Complex – Riyadh",
        "phone": "011-435-8000",
        "resources": ["Nafsiya Platform", "Sehha App", "937 Hotline"],
    },
    "jeddah": {
        "clinic": "Erada & Mental Health Complex – Jeddah",
        "phone": "012-605-2400",
        "resources": ["Sehha App", "Nafsiya Platform"],
    },
    "dammam": {
        "clinic": "Erada & Mental Health Complex – Dammam",
        "phone": "013-826-1700",
        "resources": ["937 Hotline", "Saudi Red Crescent 997"],
    },
    "khobar": {
        "clinic": "King Fahad Hospital of the University – Khobar",
        "phone": "013-896-6666",
        "resources": ["Sehha App", "Nafsiya Platform"],
    },
    "national": {
        "clinic": "Saudi National Resources",
        "phone": "Multiple",
        "resources": [
            "937 – Ministry of Health",
            "Sehha App",
            "Red Crescent 997",
        ],
    },
}

EMOTIONAL_PATTERNS = [
    "i feel",
    "i'm feeling",
    "i am sad",
    "sad",
    "anxious",
    "stressed",
    "empty",
    "alone",
    "lost",
    "low",
    "i need help",
    "panic",
    "overwhelmed",
    "can't cope",
    "struggling",
    "exhausted",
    "hurt",
    "feeling down",
]

MENTAL_TOPICS = [
    "stress",
    "anxiety",
    "depression",
    "lonely",
    "panic",
    "burnout",
    "relationship",
    "grief",
    "worry",
    "fear",
]

CRISIS_KEYWORDS = [
    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "hurt myself",
    "self harm",
    "no point",
    "better off dead",
]

TEST_REQUEST_KEYWORDS = [
    "stress test",
    "take a test",
    "take test",
    "assessment",
    "check my stress",
    "evaluate",
    "measure my stress",
]

STRESS_TEST_QUESTIONS = [
    "On a scale of 1-5, how overwhelmed do you feel? (1 = not at all, 5 = extremely)",
    "How difficult is it for you to relax? (1 = very easy, 5 = very difficult)",
    "How much trouble are you having sleeping? (1 = none, 5 = severe)",
    "How irritable do you feel? (1 = not at all, 5 = extremely)",
    "How difficult is it to concentrate? (1 = very easy, 5 = very difficult)",
]


# --------------------------------------------------------------------
# User session & helpers
# --------------------------------------------------------------------


class UserSession:
    def __init__(self):
        self.current_location = None
        self.last_question_asked = None
        self.in_professional_help_mode = False
        self.llm_history = []  # internal history for LLM only
        # Stress test mode
        self.in_test_mode = False
        self.test_question_index = 0
        self.test_answers = []

    def reset(self):
        self.last_question_asked = None
        self.in_professional_help_mode = False
        self.llm_history = []
        self.in_test_mode = False
        self.test_question_index = 0
        self.test_answers = []


user_sessions = defaultdict(UserSession)


def is_greeting(text: str) -> bool:
    return text.lower().strip() in ["hi", "hello", "hey", "salam"]


def is_crisis(text: str) -> bool:
    return any(k in text.lower() for k in CRISIS_KEYWORDS)


def extract_location(text: str):
    t = text.lower()
    for city in SAUDI_CITIES:
        if city in t:
            return city
    return None


def semantic_mental_check(text: str) -> bool:
    t = text.lower()

    emotion_words = [
        "happy",
        "good",
        "great",
        "better",
        "calm",
        "fine",
        "sad",
        "bad",
        "hurt",
        "stressed",
        "anxious",
        "tired",
        "down",
        "upset",
        "angry",
        "worried",
        "scared",
    ]
    if any(word in t for word in emotion_words):
        return True

    if len(t.split()) >= 3:
        return True

    if any(p in t for p in EMOTIONAL_PATTERNS):
        return True

    if any(p in t for p in MENTAL_TOPICS):
        return True

    return False


def is_test_request(text: str) -> bool:
    """Check if user is requesting a stress test."""
    t = text.lower()
    return any(keyword in t for keyword in TEST_REQUEST_KEYWORDS)


def has_negative_emotions(text: str) -> bool:
    """Detect if user is expressing negative emotions that warrant a test suggestion."""
    t = text.lower()
    negative_indicators = ["stressed", "anxious", "overwhelmed", "exhausted", "tired","sad", "down", "depressed", "hopeless", "worried", "panic","can't cope", "struggling", "difficult", "hard time", "feeling bad","not well", "upset", "angry", "frustrated", "lonely", "isolated"]

    return any(indicator in t for indicator in negative_indicators)


# --------------------------------------------------------------------
# LLM Response
# --------------------------------------------------------------------


def handle_emotional(session: UserSession, user_text: str) -> str:
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(session.llm_history)
        messages.append({"role": "user", "content": user_text})

        resp = client.chat_completion(
            messages=messages,
            max_tokens=250,
            temperature=0.7,
        )
        assistant_reply = resp.choices[0].message["content"]

        # update internal history
        session.llm_history.append({"role": "user", "content": user_text})
        session.llm_history.append(
            {"role": "assistant", "content": assistant_reply}
        )

        return assistant_reply
    except Exception:
        return "I'm here with you. Tell me more about what you're feeling. 💛"


def handle_resources(location: str) -> str:
    if not location:
        return "Which city are you in? (Riyadh, Jeddah, Dammam, Khobar...)"

    loc = location.lower()
    if loc in SAUDI_MENTAL_HEALTH:
        data = SAUDI_MENTAL_HEALTH[loc]
        extra = "\n".join(data["resources"])
        return (
            f"Here are mental-health resources in **{location.title()}**:\n\n"
            f"📍 **Clinic:** {data['clinic']}\n"
            f"📞 **Phone:** {data['phone']}\n\n"
            f"Additional Services:\n{extra}\n\n"
            "You deserve care and support. 💛"
        )

    return "I’ll help — which Saudi city are you in?"


def generate_reply(history, user_message, conversation_mode, session_id="default"):
    session = user_sessions[session_id]
    text = user_message.strip()

    # Crisis
    if is_crisis(text):
        session.reset()
        return CRISIS_REPLY, ""

    # Handle stress test mode
    if session.in_test_mode:
        # Try to parse the answer (should be 1-5)
        try:
            answer = int(text)
            if 1 <= answer <= 5:
                session.test_answers.append(answer)
                session.test_question_index += 1
                
                # Check if we have all 5 answers
                if session.test_question_index >= len(STRESS_TEST_QUESTIONS):
                    # Run the encrypted stress test
                    result = run_private_stress_check(*session.test_answers)
                    session.in_test_mode = False
                    session.test_answers = []
                    session.test_question_index = 0
                    return f"{result}\n\nHow are you feeling about these results? Would you like to talk about it?", "support"
                else:
                    # Ask next question
                    return STRESS_TEST_QUESTIONS[session.test_question_index], "support"
            else:
                return "Please enter a number between 1 and 5.", "support"
        except ValueError:
            if "cancel" in text.lower() or "stop" in text.lower():
                session.in_test_mode = False
                session.test_answers = []
                session.test_question_index = 0
                return "Test cancelled. How else can I support you?", "support"
            return "Please enter a number between 1 and 5, or say 'cancel' to stop the test.", "support"

    # Check if user is requesting a test
    if is_test_request(text):
        session.in_test_mode = True
        session.test_answers = []
        session.test_question_index = 0
        return (
            "I'll guide you through a quick stress assessment. 🔐\n\n"
            "Your answers will be encrypted before being sent to the server for analysis.\n\n"
            f"{STRESS_TEST_QUESTIONS[0]}"
        ), "support"

    # Greeting
    if is_greeting(text):
        session.reset()
        return "Hi there! 👋 How are you feeling today? 💛", "support"

    # Location
    loc = extract_location(text)
    if loc:
        session.current_location = loc
        return handle_resources(loc), "support"

    # Professional help
    if any(
        w in text.lower()
        for w in ["doctor", "therapist", "professional", "appointment"]
    ):
        session.in_professional_help_mode = True
        return "Of course. Which city are you in?", "support"

    # Emotional support
    if semantic_mental_check(text):
        reply = handle_emotional(session, text)
        
        # Suggest test if user is experiencing negative emotions
        if has_negative_emotions(text):
            reply += "\n\nWould you like to take a quick stress assessment? It might help us understand how you're feeling better. Just say 'take a test' if you're interested. 💛"
        
        return reply, "support"

    return "I'm here to support you. How are you feeling today?", "support"


# --------------------------------------------------------------------
# Gradio Handlers
# --------------------------------------------------------------------


def chat_response(user_message, history, conversation_mode, session_id="default"):
    if history is None:
        history = []

    assistant_text, new_mode = generate_reply(
        history, user_message, conversation_mode, session_id
    )

    history.append([user_message, assistant_text])

    return history, new_mode, ""


def clear_chat(session_id="default"):
    if session_id in user_sessions:
        del user_sessions[session_id]
    return [], None


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------

with gr.Blocks(title="Mental Wellness Chatbot", theme=gr.themes.Soft()) as demo:
    # REMOVE FOOTER
    gr.HTML(
        """
    <style>
    .footer, .svelte-py6t96, .svelte-1ipelgc, #footer {display:none !important;}
    </style>
    """
    )

    # ---- Tab 1: Chatbot ----
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=500)
        txt = gr.Textbox(
            label="How are you feeling?", placeholder="Share your thoughts... 💛"
        )

        mode_state = gr.State(value=None)
        session_state = gr.State(value="default")

        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Chat")

        send.click(
            chat_response,
            inputs=[txt, chatbot, mode_state, session_state],
            outputs=[chatbot, mode_state, txt],
        )

        txt.submit(
            chat_response,
            inputs=[txt, chatbot, mode_state, session_state],
            outputs=[chatbot, mode_state, txt],
        )

        clear_btn.click(
            clear_chat,
            inputs=[session_state],
            outputs=[chatbot, mode_state],
        )

    # ---- Tab 2: Encrypted Stress Check ----
    with gr.Tab("Private Stress Check"):
        gr.Markdown(
            "### 🔐 Private Stress Check\n"
            "Move the sliders to answer a few quick questions about your stress.\n"
            "Your answers will be **encrypted locally** before being sent to the server."
        )

        with gr.Row():
            q1 = gr.Slider(1, 5, step=1, label="Feeling overwhelmed")
            q2 = gr.Slider(1, 5, step=1, label="Difficulty relaxing")
        with gr.Row():
            q3 = gr.Slider(1, 5, step=1, label="Trouble sleeping")
            q4 = gr.Slider(1, 5, step=1, label="Feeling irritable")
        q5 = gr.Slider(1, 5, step=1, label="Difficulty concentrating")

        run_btn = gr.Button("Analyze (Encrypted)")
        result_box = gr.Markdown()

        run_btn.click(
            run_private_stress_check,
            inputs=[q1, q2, q3, q4, q5],
            outputs=result_box,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
