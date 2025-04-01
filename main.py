import cohere
from fastapi import FastAPI, Form, Request, WebSocket
from typing import Annotated
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import requests
import os
from dotenv import load_dotenv
from prompts import CAREER_MENTOR_PROMPT

load_dotenv()  # Load environment variables

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cohere client for chat
co = cohere.ClientV2(
    api_key=os.getenv("COHERE_API_KEY")
)

# Hugging Face configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Add your token to .env file
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

@app.get('/', response_class=HTMLResponse)
async def chatpage(request: Request):
    # Reset chat history on page load
    chat_log = [{'role':'system', 'content': CAREER_MENTOR_PROMPT}]
    return templates.TemplateResponse("layout.html", {
        "request": request,
        "messages": chat_log
    })

# Store chat history in request.session
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

@app.post('/')
async def chat(request: Request, user_input: Annotated[str, Form()]):
    # Get or initialize chat history
    chat_log = request.session.get('chat_log', [{'role':'system', 'content': CAREER_MENTOR_PROMPT}])
    
    # Add user message
    chat_log.append({'role': 'user', 'content': user_input})
    
    response = co.chat(
        model="command-a-03-2025",
        messages=chat_log,
        temperature=0.6,
        # max_tokens=200
    )
    
    # Add bot response
    bot_response = response.message.content[0].text
    chat_log.append({'role': 'assistant', 'content': bot_response})
    
    # Save updated chat history
    request.session['chat_log'] = chat_log
    
    return templates.TemplateResponse("layout.html", {
        "request": request,
        "messages": chat_log
    })

@app.post('/generate-image')
async def generate_image(prompt: Annotated[str, Form()]):
    try:
        # Make request to Hugging Face API
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            json={"inputs": prompt}
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            # Convert the binary response to base64 for displaying in HTML
            import base64
            image_bytes = response.content
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_image}"
            return JSONResponse(content={"image_url": image_url})
        else:
            return JSONResponse(
                status_code=response.status_code,
                content={"error": f"Error from Hugging Face API: {response.text}"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
