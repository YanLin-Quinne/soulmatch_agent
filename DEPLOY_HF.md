# HuggingFace Spaces Deployment Guide

## Step 1: Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Owner**: Quinnnnnne
   - **Space name**: soulmatch-agent
   - **License**: MIT
   - **Select the Space SDK**: Docker
   - **Space hardware**: CPU basic (free) or CPU upgrade (faster)

## Step 2: Configure Environment Variables

In the Space Settings, go to Variables and Secrets and add:

```
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key
QWEN_API_KEY=your_qwen_key
DEEPSEEK_API_KEY=your_deepseek_key
```

Configure at least one LLM provider key.

## Step 3: Push Code

```bash
cd /Users/quinne/Desktop/soulmatch_agent_test

# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/Quinnnnnne/soulmatch-agent

# Push code
git push hf main
```

## Step 4: Wait for Build

HuggingFace will automatically:
1. Read the Dockerfile
2. Build the Docker image (approximately 5-10 minutes)
3. Start the container
4. Run the application on port 7860

## Step 5: Access the Application

Once the build is complete, visit:
https://huggingface.co/spaces/Quinnnnnne/soulmatch-agent

## Troubleshooting

### Build Failures
- Check Dockerfile syntax
- Review Build logs in the Space UI
- Ensure requirements.txt includes all dependencies

### Runtime Errors
- Verify environment variables are set correctly
- Check Container logs
- Ensure at least one LLM API key is valid

### Frontend Not Loading
- Ensure frontend/dist directory exists
- Check the npm build step in Dockerfile
- Verify static file paths are correct

## Local Docker Testing

```bash
# Build image
docker build -t soulmatch-agent .

# Run container
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e GEMINI_API_KEY=your_key \
  -e DEEPSEEK_API_KEY=your_key \
  soulmatch-agent

# Access http://localhost:7860
```

## Updating the Space

Every push to the main branch triggers an automatic rebuild:

```bash
git add .
git commit -m "Update: description of changes"
git push hf main
```

## Notes

1. **API key security**: Never hardcode keys in source code; use environment variables
2. **CORS**: Configured in main.py to allow all origins
3. **WebSocket**: HuggingFace Spaces supports WebSocket connections
4. **Persistent storage**: Data is lost on container restart; ChromaDB uses in-memory mode
5. **Logs**: Click "Logs" on the Space page to view runtime logs
