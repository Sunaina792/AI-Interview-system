# 🚀 Quick Start: Deploy to Hugging Face Spaces in 5 Minutes

## Prerequisites ✅
- GitHub account (already set up ✓)
- Hugging Face account (free at huggingface.co)
- GROQ API key (free at console.groq.com)

---

## 🎯 5-Minute Setup

### 1️⃣ Get Your GROQ API Key (1 min)
```
1. Visit: https://console.groq.com/keys
2. Create a new API key
3. Copy it (you'll need it in step 4)
```

### 2️⃣ Create Hugging Face Space (2 min)
```
1. Go to: https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - Name: ai-interview-system
   - License: Apache 2.0
   - SDK: Docker
   - Hardware: CPU basic (works fine, or GPU if available)
4. Click "Create Space"
```

### 3️⃣ Link Your GitHub Repo (1 min)
```
1. Click "Settings" in your new Space
2. Under "Linked Repositories", click "Link a repo"
3. Select: Sunaina792/AI-Interview-system
4. Spaces will auto-sync and auto-deploy!
```

### 4️⃣ Add Your API Key (1 min)
```
1. In Space Settings → "Repository secrets"
2. Click "New secret"
3. Name: GROQ_API_KEY
4. Value: [paste your key from step 1]
5. Click "Save"
```

---

## ✨ Done! Your App is Live

Your Space will automatically build and deploy. Check the **Logs** tab to monitor:
```
Cloning repository...
Building Docker image...
Starting container...
App is running at: https://huggingface.co/spaces/YOUR_USERNAME/ai-interview-system
```

---

## 📱 Using Your Deployed App

Once live, users can:
1. Open the Space link in their browser
2. Upload their resume (PDF/DOCX/TXT)
3. Paste job description
4. Choose difficulty level
5. Upload interview video or record in-browser
6. Get AI-scored feedback report

---

## 🐛 If Something Goes Wrong

**Check these in order:**
1. **Logs tab** - Most errors show here with solutions
2. **GROQ_API_KEY** - Verify it's set in Repository secrets
3. **requirements.txt** - Verify all dependencies are listed
4. **Dockerfile** - Check it's in the root directory

Common issues & fixes are in `HF_SPACES_SETUP.md`

---

## 📚 Next Steps

- 📖 Read: [HF_SPACES_SETUP.md](./HF_SPACES_SETUP.md) - Full detailed guide
- 🔗 Docs: [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- 💬 Share: Post your Space link to show it off!

---

## 🎉 Congratulations!

Your AI Interview System is now live on Hugging Face Spaces!
Share it with your network: `https://huggingface.co/spaces/YOUR_USERNAME/ai-interview-system`

