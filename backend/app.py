# ============================================================
# FILE: app.py
#
# This file has TWO AI components:
#
# 1. ML MODEL (Random Forest)
#    - Trained on data to predict best AWS instance
#    - Based on: users, storage, budget, traffic
#
# 2. LLM (Google Gemini)
#    - Large Language Model (like ChatGPT)
#    - Answers cloud computing questions in natural language
#    - FREE to use with a Google API key
#    - If no API key, uses pre-written answers (demo mode)
#
# HOW TO GET FREE GEMINI API KEY:
#   1. Go to: https://aistudio.google.com/app/apikey
#   2. Sign in with Google account
#   3. Click "Create API Key"
#   4. Copy the key and paste it below where it says YOUR_KEY_HERE
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import urllib.request
import urllib.error
import json

app = Flask(__name__)
CORS(app)

# ============================================================
# PASTE YOUR FREE GEMINI API KEY HERE
# Get it free from: https://aistudio.google.com/app/apikey
# ============================================================
GEMINI_API_KEY = "YOUR_KEY_HERE"

# ============================================================
# LOAD ML MODEL
# ============================================================
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'cloud_model.pkl')
model = joblib.load(model_path)
print("ML Model loaded!")

# ============================================================
# AWS INSTANCE DETAILS (5 instances only)
# ============================================================
instances = {
    0: {
        "name": "t2.micro",
        "category": "Basic — Starter Instance",
        "vcpu": 1,
        "ram_gb": 1,
        "storage_included": 30,
        "monthly_cost": 588,
        "max_users": 100,
        "max_traffic": 10,
        "good_for": "Student projects, learning, testing small ideas",
        "not_good_for": "Real business apps, more than 100 users",
        "color": "#10b981",
        "tip": "FREE on AWS Free Tier for 12 months! Perfect for beginners."
    },
    1: {
        "name": "t3.small",
        "category": "Small — Personal & Blog Sites",
        "vcpu": 2,
        "ram_gb": 2,
        "storage_included": 50,
        "monthly_cost": 1260,
        "max_users": 500,
        "max_traffic": 50,
        "good_for": "Personal websites, blogs, portfolios, small tools",
        "not_good_for": "High traffic, online stores, large databases",
        "color": "#3b82f6",
        "tip": "Great starting point for freelancers and small projects."
    },
    2: {
        "name": "t3.medium",
        "category": "Medium — Standard Web Applications",
        "vcpu": 2,
        "ram_gb": 4,
        "storage_included": 100,
        "monthly_cost": 2520,
        "max_users": 2000,
        "max_traffic": 200,
        "good_for": "E-commerce sites, company websites, REST APIs, startups",
        "not_good_for": "Very high traffic apps, heavy data processing",
        "color": "#6366f1",
        "tip": "Most popular choice for web applications worldwide."
    },
    3: {
        "name": "t3.large",
        "category": "Large — Growing Applications",
        "vcpu": 2,
        "ram_gb": 8,
        "storage_included": 200,
        "monthly_cost": 5040,
        "max_users": 5000,
        "max_traffic": 500,
        "good_for": "Growing startups, medium traffic apps, small databases",
        "not_good_for": "Massive scale apps, 10k+ daily users",
        "color": "#f59e0b",
        "tip": "Upgrade to this when your t3.medium starts getting slow."
    },
    4: {
        "name": "m5.xlarge",
        "category": "Enterprise — High Performance",
        "vcpu": 4,
        "ram_gb": 16,
        "storage_included": 500,
        "monthly_cost": 11760,
        "max_users": 50000,
        "max_traffic": 1500,
        "good_for": "Large platforms, high traffic, big databases, 50k users",
        "not_good_for": "Simple websites (too powerful and expensive)",
        "color": "#8b5cf6",
        "tip": "Use Auto Scaling Groups so AWS adds more servers automatically."
    }
}

# ============================================================
# PREDICT ROUTE — ML Model
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json()
    users   = float(data['users'])
    storage = float(data['storage'])
    budget  = float(data['budget'])
    traffic = float(data['traffic'])

    # Run ML model
    features    = np.array([[users, storage, budget, traffic]])
    prediction  = int(model.predict(features)[0])
    probs       = model.predict_proba(features)[0]

    recommended = instances[prediction]
    confidence  = round(probs[prediction] * 100, 1)

    # Budget check
    is_over_budget    = recommended['monthly_cost'] > budget
    budget_alternative = None

    if is_over_budget:
        for i in range(prediction - 1, -1, -1):
            if instances[i]['monthly_cost'] <= budget:
                budget_alternative = instances[i]
                break
        if budget_alternative is None:
            budget_alternative = instances[0]

    # Extra storage cost
    extra_storage      = max(0, storage - recommended['storage_included'])
    extra_storage_cost = round(extra_storage * 6.72, 2)
    total_cost         = round(recommended['monthly_cost'] + extra_storage_cost, 2)

    # All options list
    all_options = []
    for i in range(5):
        all_options.append({
            "name":        instances[i]["name"],
            "probability": round(probs[i] * 100, 1),
            "monthly_cost":instances[i]["monthly_cost"],
            "color":       instances[i]["color"],
            "fits_budget": instances[i]["monthly_cost"] <= budget
        })
    all_options.sort(key=lambda x: x["probability"], reverse=True)

    return jsonify({
        "success": True,
        "name":               recommended["name"],
        "category":           recommended["category"],
        "vcpu":               recommended["vcpu"],
        "ram_gb":             recommended["ram_gb"],
        "storage_included":   recommended["storage_included"],
        "monthly_cost":       recommended["monthly_cost"],
        "total_monthly_cost": total_cost,
        "extra_storage_cost": extra_storage_cost,
        "max_users":          recommended["max_users"],
        "max_traffic":        recommended["max_traffic"],
        "good_for":           recommended["good_for"],
        "not_good_for":       recommended["not_good_for"],
        "color":              recommended["color"],
        "tip":                recommended["tip"],
        "confidence":         confidence,
        "user_budget":        budget,
        "is_over_budget":     is_over_budget,
        "budget_alternative": budget_alternative,
        "all_options":        all_options
    })


# ============================================================
# CHAT ROUTE — LLM (Google Gemini)
# ============================================================
@app.route('/chat', methods=['POST'])
def chat():
    data     = request.get_json()
    question = data.get('question', '')
    context  = data.get('context', '')

    # Try Gemini LLM first (if API key is set)
    if GEMINI_API_KEY and GEMINI_API_KEY != "AIzaSyDuw0bLwoZQTBEj-AHix0kXc6yBANXTMQ8":
        try:
            answer = ask_gemini(question, context)
            return jsonify({"success": True, "answer": answer, "powered_by": "Google Gemini LLM"})
        except Exception as e:
            print(f"Gemini error: {e} — falling back to demo mode")

    # Fallback: rule-based answers (no API key needed)
    answer = rule_based_answer(question, context)
    return jsonify({"success": True, "answer": answer, "powered_by": "Demo Mode"})


# ============================================================
# GEMINI LLM FUNCTION
# This sends the user's question to Google Gemini AI
# and gets a smart, natural language answer back
# ============================================================
def ask_gemini(question, context):
    # Build the prompt we send to Gemini
    # We tell it to act as a cloud computing expert
    prompt = f"""You are a friendly Cloud Computing Assistant helping a beginner student understand AWS cloud resources.

Context about the current recommendation (if any): {context}

Student's question: {question}

Answer in simple, beginner-friendly language. 
Use bullet points where helpful.
Keep the answer short (under 150 words).
Use simple words — this is a beginner student.
If relevant, mention AWS instance names like t2.micro, t3.medium etc."""

    # Build the API request to Gemini
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    body = json.dumps({
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    # Send request to Gemini and read response
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    # Extract the text from Gemini's response
    answer = result["candidates"][0]["content"]["parts"][0]["text"]
    return answer


# ============================================================
# RULE-BASED ANSWERS (used when no Gemini API key)
# ============================================================
def rule_based_answer(question, context):
    q = question.lower()

    if 'budget' in q or 'cheap' in q or 'cost' in q or 'price' in q:
        return """💰 AWS Instance Monthly Costs:

• t2.micro  → ₹588/month   (up to 100 users)  ← FREE on Free Tier!
• t3.small  → ₹1,260/month  (up to 500 users)
• t3.medium → ₹2,520/month  (up to 2,000 users)
• t3.large  → ₹5,040/month  (up to 5,000 users)
• m5.xlarge → ₹11,760/month (up to 50,000 users)

💡 Tips to save money:
→ Use AWS Free Tier (t2.micro free for 12 months)
→ Start small, upgrade only when needed
→ Turn off your server when not using it"""

    elif 'over budget' in q or 'cannot afford' in q:
        return """📉 When over budget, the system shows you a cheaper alternative automatically!

What you can do:
→ Choose the affordable alternative shown in green
→ Reduce your app features to need less resources
→ Start with t2.micro (FREE on AWS Free Tier!)
→ Upgrade later when you start earning money

Remember: You can upgrade anytime on AWS with no data loss!"""

    elif 'user' in q or 'how many' in q:
        return """👥 Users each instance can handle per day:

• t2.micro  → up to 100 users
• t3.small  → up to 500 users
• t3.medium → up to 2,000 users
• t3.large  → up to 5,000 users
• m5.xlarge → up to 50,000 users

These are estimates. Actual numbers depend on what your app does."""

    elif 'storage' in q:
        return """💾 Storage on AWS:

Each instance includes some storage.
If you need more, AWS charges extra:
→ ₹6.72 per GB per month

Example: Need 200 GB, instance includes 50 GB
→ Extra = 150 GB × $0.08 = $12/month extra

Tip: Store files/images in Amazon S3 (cheaper at ₹1.93/GB)"""

    elif 'scale' in q or 'upgrade' in q or 'grow' in q:
        return """📈 How to grow your app on AWS:

Upgrade path (start small, grow step by step):
t2.micro ($7) → t3.small ($15) → t3.medium ($30)
→ t3.large ($60) → m5.xlarge ($140)

Upgrading on AWS is easy:
1. Stop your server (1 minute)
2. Change instance type
3. Start again — all data is safe!

Upgrade when your CPU or RAM usage goes above 70%."""

    elif 'what is aws' in q or ('aws' in q and 'what' in q):
        return """☁️ What is AWS?

AWS = Amazon Web Services

Instead of buying your own server (costs lakhs of rupees),
you RENT a server from Amazon and pay monthly.

Benefits:
→ Pay only what you use (like mobile recharge)
→ Start in minutes, not weeks
→ Scale up or down anytime
→ Amazon handles all maintenance

Most used AWS services:
→ EC2  = Virtual servers (what we recommend)
→ S3   = File storage (like Google Drive)
→ RDS  = Managed database"""

    elif 'ec2' in q or 'instance' in q:
        return """🖥️ What is an EC2 Instance?

EC2 = Elastic Compute Cloud = a virtual server on AWS

Think of it as renting a computer that:
→ Runs 24 hours a day, 7 days a week
→ Is connected to the internet
→ Can be made bigger or smaller anytime

Our ML model recommends which EC2 instance
is best for your specific requirements!"""

    elif context and ('explain' in q or 'why' in q):
        return f"""🤖 About your recommendation:

{context}

How the ML model decided:
→ It looked at your users, traffic, storage and budget
→ Compared with 3,000 training examples
→ 100 decision trees voted on the best instance
→ The majority vote = your recommendation!

Add a free Gemini API key to get smarter AI answers!
Get it free at: https://aistudio.google.com/app/apikey"""

    else:
        return """👋 I am your Cloud Resource Planning Assistant!

I can help you with:
→ AWS instance costs and comparisons
→ What to do when over budget
→ How many users each instance handles
→ How to upgrade as your app grows
→ What is AWS and EC2

💡 To get smarter AI answers powered by Google Gemini LLM:
1. Go to https://aistudio.google.com/app/apikey
2. Create a free API key
3. Paste it in app.py where it says YOUR_KEY_HERE

Try asking:
→ "What is AWS?"
→ "What if I am over budget?"
→ "How do I upgrade my instance?" """


# ============================================================
# HEALTH CHECK
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    key_status = "Gemini LLM connected!" if GEMINI_API_KEY != "YOUR_KEY_HERE" else "Demo mode (no API key)"
    return jsonify({
        "status": "Server is running!",
        "ml_model": "Random Forest loaded!",
        "llm": key_status
    })


# ============================================================
# START SERVER
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  Cloud Resource Planning Agent — Starting...")
    print("=" * 55)
    key_status = "✅ Gemini LLM active!" if GEMINI_API_KEY != "YOUR_KEY_HERE" else "⚠️  No Gemini key — using demo mode"
    print(f"  ML Model : ✅ Random Forest loaded")
    print(f"  LLM Chat : {key_status}")
    print(f"  Visit    : http://localhost:5000/health")
    print("=" * 55)
    app.run(debug=True, port=5000)