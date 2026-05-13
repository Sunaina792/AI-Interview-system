# 🚀 Hugging Face Spaces Deployment Guide

## Step 1: Create a Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up or log in
3. Go to [huggingface.co/spaces](https://huggingface.co/spaces)

## Step 2: Create a New Space
1. Click **"Create new Space"**
2. Fill in the details:
   - **Space name:** `ai-interview-system` (or your choice)
   - **License:** Choose appropriate license (e.g., Apache 2.0)
   - **Space SDK:** Select **"Docker"**
   - **Space hardware:** Choose **"CPU basic"** (or GPU if you have quota)
   - **Visibility:** Public or Private (your choice)

3. Click **"Create Space"**

## Step 3: Configure Your Space

### Option A: Connect GitHub (Recommended)
1. In your new Space, click **"Settings"** (top right)
2. Go to **"Linked Repositories"**
3. Link your GitHub repository: `https://github.com/Sunaina792/AI-Interview-system`
4. Set repository settings to auto-sync

### Option B: Manual Upload
If not connecting GitHub, upload files manually via the web interface

## Step 4: Set Environment Variables
1. In your Space, go to **Settings** → **Repository secrets**
2. Add your secrets:
   - **Name:** `GROQ_API_KEY`
   - **Value:** `your_actual_groq_api_key`
   
3. Click **"Add secret"**

## Step 5: Verify app.py Configuration
The Dockerfile will automatically:
- Install all dependencies from `requirements.txt`
- Copy all files
- Run `python gradio_app.py`
- Expose the app on port 7860

## Step 6: Monitor Deployment
1. Your Space will start building automatically
2. Watch the **"Logs"** tab to see build progress
3. Once complete, your app will be live at:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/ai-interview-system
   ```

## 🔧 Troubleshooting

### Build Fails
- Check **Logs** tab for errors
- Verify `requirements.txt` has all dependencies
- Ensure Python version compatibility (3.11+)

### App Crashes After Deploy
- Check **Logs** for runtime errors
- Verify environment variables are set correctly
- Check that media models can load (may need GPU)

### Webcam Not Working
- This is expected in browser-based deployment
- Consider using Gradio's `gr.Interface` with image upload instead
- Users can record video offline and upload

### Out of Memory
- Switch to a GPU machine if available (Settings → Space hardware)
- Optimize model loading to reduce memory footprint

## 📱 Access Your App
Once deployed, share the Space URL with others:
```
https://huggingface.co/spaces/YOUR_USERNAME/ai-interview-system
```

Users can:
- Access via browser (no installation needed)
- Upload resume files
- Record/upload interview video
- Get AI-scored feedback

## 💡 Pro Tips

1. **Custom README:** Create a `README_HF.md` in your repo for Space-specific instructions
2. **Persistent Storage:** Use Hugging Face datasets for storing results
3. **Scheduled Runs:** Use GitHub Actions to trigger updates
4. **Analytics:** Monitor Space usage in the Space settings

## 🔗 Useful Links
- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/)
- [Docker Guide for Spaces](https://huggingface.co/docs/hub/spaces-docker)

---

**Next Steps:**
1. Get your GROQ API key from [groq.com](https://groq.com)
2. Follow the steps above to create your Space
3. Deploy and share with others!
