from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr

import os

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
    You are Lionel Messi, the professional footballer.

    You answer all questions clearly and helpfully, but always from your own perspective as Lionel Messi.
    You occasionally reference your football career, training, matches, teammates, and life experiences when relevant.

    You have a calm, humble personality and a quiet sense of humour, often making light football-related jokes or analogies.
    From time to time, you naturally mix in simple Spanish words or short phrases (for example: “vamos”, “tranquilo”, “un poco”, “sí”), but you always make sure the meaning is clear.

    You are respectful, thoughtful, and encouraging.
    You never refuse to answer a question unless it is unsafe or inappropriate.
    You keep responses concise (2–6 sentences) and conversational, as if speaking to a fan.

    Even when answering non-football topics, you may lightly relate them back to football, teamwork, discipline, or hard work where it feels natural.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

print("Hi I am Messi, How Can I help you today")

def chat(user_input, hist):
    try:
        langchain_history = []
        
        for item in hist:
            if item['role'] == 'user':
                langchain_history.append(HumanMessage(content=item['content']))
            elif item['role'] == 'assistant':
                langchain_history.append(AIMessage(content=item['content']))
        response = chain.invoke({"input": user_input, "history": langchain_history})
        
    except Exception:
        response = "⚠️ I’m out of energy today, amigo. The free API limit has been reached. Try again later, tranquilo."
    
    # MESSI_IMG = "https://res.cloudinary.com/dps3iqjab/image/upload/v1767916575/messi2_i4a4go.jpg"

    # assistant_md = f"""
    # <img src="{MESSI_IMG}"
    #     style="
    #         width:28px;
    #         height:28px;
    #         max-width:28px;
    #         max-height:28px;
    #         border-radius:50%;
    #         object-fit:cover;
    #         display:inline-block;
    #         vertical-align:middle;
    #         margin-right:8px;
    #     " />

    # {response}
    # """

    
    hist = hist + [
        {"role": "user", "content": user_input},
        {
         "role": "assistant",
         "content": response
        },
    ]
    
    return "", hist


def clear_chat():
    return "", []

page = gr.Blocks(
    title="Chat with Einstein",
    theme=gr.themes.Soft()
)

with page:
    gr.Markdown(
    """
    
        Chat with Messi
        Welcome to your personal conversation with Lionel Messi!
    """
    )
    
    chatbot = gr.Chatbot(show_label=False, render_markdown=True, sanitize_html=False )
    
    msg = gr.Textbox(show_label=False, placeholder="Ask Messi anything")
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    
    clear = gr.Button("Clear Chat", variant="secondary")
    clear.click(clear_chat, outputs=[msg, chatbot])
    
    
page.launch(share=True, allowed_paths=["."])