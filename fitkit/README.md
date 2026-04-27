# 💪 FitKit — AI Fitness Coach

AI-powered fitness tracking app with real-time pose detection, rep counting, and an AI chatbot coach.

## 🚀 Deploy to Streamlit Cloud (Free — works on mobile!)

### Step 1 — Fork / Push to GitHub
Upload this entire `fitkit_output` folder to a **GitHub repository**.

### Step 2 — Add your Gemini API Key to Streamlit Cloud
In the Streamlit Cloud dashboard → your app → **Settings → Secrets**, add:
```toml
GEMINI_API_KEY = "your-actual-gemini-api-key"
```

### Step 3 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

### Step 4 — Install on Mobile as PWA
1. Open your app URL in **Chrome** on Android
2. Tap the **three dots menu** (⋮)
3. Tap **"Add to Home Screen"**
4. FitKit appears on your home screen like a real app ✅

---

## 📱 Mobile Camera
The app will ask for camera permission when you start an exercise — tap **Allow**.

## 🔑 Getting a Gemini API Key
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key**
3. Copy and paste into Streamlit Cloud Secrets

## 🏃 Exercises Supported
Squats, Push-ups, Bicep Curls, Lunges, Deadlifts, Bench Press, Pull-ups, Shoulder Press, and 20+ more.

---

## Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```
