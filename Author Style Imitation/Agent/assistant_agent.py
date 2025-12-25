import os
import io
from fpdf import FPDF
import json
import re
import base64
import requests

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
import chainlit as cl
import google.generativeai as genai
from PIL import Image

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

api_key='AIzaSyCszgyOveUzf6LozYB6YYtSEABXR03UFkU'
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
vector_store = Chroma(embedding_function = embeddings)

loader = TextLoader("Lydia Davis Stories.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)    
_ = vector_store.add_documents(documents=all_splits)


@cl.on_chat_start
async def on_chat_start():
    
    await cl.Message(
        content="Welcome User! Please specify the creation model you want to use:",
        actions=[
            cl.Action(name="simple_llm", value="simple_llm", label="Simple LLM", payload={}),
            cl.Action(name="rag_llm", value="rag_llm", label="RAG LLM", payload={}),
             cl.Action(name="fine_tuned_llm", value="fine_tuned_llm", label="Fine-Tuned LLM", payload={})
        ],
    ).send() 
    
    
@cl.action_callback("simple_llm")
async def handle_simple_llm(action: cl.Action):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key,streaming = True)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI writing assistant. Write a short story following the user's instruction. Please provide a title of the story as well. Note that the title should always have ##"),
            ("human", "{chat_history}\nHuman: {question}")
        ]
    )
        
    runnable = prompt | model | StrOutputParser()
    await cl.Message(content="You selected **Simple LLM**. Happy chatting :)").send()
    
    cl.user_session.set("model_type", "simple")
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("chat_history", [])


@cl.action_callback("rag_llm")
async def handle_rag_llm(action: cl.Action):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key,streaming = True)
    
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system","You are to generate a short story based on the theme and description given by the user. Please provide a title of the story as well. Note that the title should always have ##. The context is as follows:\n\n {context}"),
                ("human", "{chat_history}\nHuman: {question}"),
            ]
        )
        
    runnable = prompt | model | StrOutputParser()
    await cl.Message(content="You selected **RAG LLM**. Happy chatting :)").send()       

    cl.user_session.set("model_type", "rag")
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("chat_history", [])
    
    
@cl.action_callback("fine_tuned_llm")
async def handle_fine_tuned_llm(action: cl.Action):
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(model_name="tunedModels/my-tuned-gemini-lydia-davis-37740")
    
    await cl.Message(content="You selected **Fine-Tuned LLM**. Happy chatting :)").send()       

    cl.user_session.set("model_type", "fine_tuned")
    cl.user_session.set("model", model)
    cl.user_session.set("chat_history", [])
    print("Hello")


def get_rag_context(details: dict):
    query_parts = [details.get("theme", ""), details.get("setting", ""), details.get("characters", "")]
    query_text = " ".join([q for q in query_parts if q != "Unknown"])

    # Query the vector store and filter by author details
    results = vector_store.similarity_search(query_text)
    i = 1
    context = ""
    for res in results:
        context += "Context Document " + str(i) + ":\n" + res.page_content + "\n\n"
        i += 1

    return context

def extract_story_details(user_prompt: str) -> dict:
    prompt = f"""
    You are a helpful assistant. A user wants to generate a short story and gives the following prompt:

    "{user_prompt}"

    Extract the following fields from the prompt:
    - Theme
    - Setting (if any)
    - Characters (if any)
    - Style/Author

    If any field is missing, leave it as "Unknown".

    Return the result in this JSON format:
    {{"theme": "...", "setting": "...", "characters": "...", "author": "..." }}
    """
    extractor_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
    response = extractor_llm.invoke(prompt)
    pattern = "```json\n(.*?)```"
    json_str = re.findall(pattern, response.content, re.DOTALL)
    # story_details = json.loads(json_str[0])
    # print(story_details[0]['theme'])
    try:
        return json.loads(json_str[0])
    except:
        print("Parsing failed. Response was:", response)
        return {}



@cl.on_message
async def on_message(message: cl.Message):
    model_type = cl.user_session.get(("model_type"))
    chat_history = cl.user_session.get("chat_history", []) 
    # print("CHAT HISTORY:> \n" + "\n".join(chat_history))
    
    ## Its either simple or rag
    if model_type != "fine_tuned":
        runnable = cast(Runnable, cl.user_session.get("runnable")) # type casting is done as cl.user_session.get() returns Any object
    
        msg = cl.Message(content="")
    
        input_schema = runnable.input_schema.schema()
    
        inputs = {
            "chat_history": "\n".join(chat_history),
            "question": message.content
        }
    
        if "context" in input_schema.get('properties', {}):
            
            story_details = extract_story_details(message.content)
            if len(story_details) == 0:
                print("Could not extract the story details")
                results = vector_store.similarity_search(message.content)
                i = 1
                context = ""
                for res in results:
                    context += "Context Document " + str(i) + ":\n" + res.page_content + "\n\n"
                i += 1
                inputs["context"] = context
            else:
                context = get_rag_context(story_details)
                inputs["context"] = context

        async for chunk in runnable.astream(
            inputs,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
        
        chat_history.append(f"Human: {message.content}")
        chat_history.append(f"Writer AI: {msg.content}")
        cl.user_session.set("chat_history", chat_history)
    
        msg.actions = [
            cl.Action(name="create_pdf", value="create_pdf", label="Create PDF", payload={"story": msg.content})
        ]
        await msg.send()
    
    else:
        llm = cast(genai.GenerativeModel, cl.user_session.get(("model")))
        
        system_prompt = "You are a helpful AI writing assistant. Write a short story following the user's instruction. Please provide a title of the story as well. Note that the title should always have ## in the start and at the end. You are provided with the chat history and the current human question\n\n" + "\n".join(chat_history) + "\n\n"
        prompt = "Current Human Question: " + message.content
        prompt = system_prompt + prompt
        
        response = llm.generate_content(prompt)
        
        if response.candidates and response.candidates[0].finish_reason == 3:
            await cl.Message(
                content= "Your prompt is triggering the saftey guardrails. Please try to re-phrase your prompt"
            ).send()
        
        else:
        
            chat_history.append(f"Human: {message.content}")
            chat_history.append(f"Writer AI: {response.text}")
            cl.user_session.set("chat_history", chat_history)
                
            await cl.Message(
                content = response.text,
                actions = [
                    cl.Action(name="create_pdf", value="create_pdf", label="Create PDF", payload={"story": response.text})
                ]
            ).send()
    
    


async def extract_sd_prompt_with_llm(story: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

    prompt_template = PromptTemplate.from_template(f"""
You are an expert prompt engineer for AI art tools like Stable Diffusion.

Given the short story below, write a one-paragraph image prompt optimized for AI image generation. Follow these rules:
- Describe one key scene or moment.
- Mention the main subject (character(s), object, creature, etc.).
- Include vivid environmental details (e.g., forest, ruins, neon city, spaceship).
- Add visual keywords like lighting (e.g., glowing, backlit, soft lighting), mood (e.g., eerie, peaceful), and detail (e.g., intricate, sharp, cinematic).
- Use artistic style modifiers like "digital painting", "artstation", "octane render", "studio lighting", "8k", etc.
- DO NOT include verbs like “standing”, “looking”, “walking” — just describe the scene composition as if it already exists.

Short Story:
\"\"\"{{story}}\"\"\"

Stable Diffusion Prompt (one paragraph):
""")

    chain = prompt_template | llm | StrOutputParser()
    return await chain.ainvoke(story)


def generate_image_from_prompt(prompt: str, output_path: str = "generated_image.png") -> str:
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, cropped, bad anatomy",
        "width": 384,
        "height": 384,
        "steps": 30,
        "cfg_scale": 7.0
    }
    response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload)

    if response.status_code == 200:
        r = response.json()
        img_data = base64.b64decode(r['images'][0])
        with open(output_path, "wb") as f:
            f.write(img_data)
        print(f"Image saved as {output_path}")
        return output_path
    else:
        print("Image generation failed:", response.text)
        return None


def validate_image(image_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            print(f"[DEBUG] Image format: {img.format}, size: {img.size}")
            return True
    except Exception as e:
        print(f"[ERROR] Invalid image file: {e}")
        return False


from PIL import Image

@cl.action_callback("create_pdf")
async def download_pdf(action: cl.Action):
    story = action.payload.get("story")
    if not story:
        await cl.Message(content="No story to convert!").send()
        return

    await cl.Message(content="Generating image for your story...").send()

    image_prompt = await extract_sd_prompt_with_llm(story)
    image_path = generate_image_from_prompt(image_prompt)

    if not image_path or not os.path.exists(image_path):
        await cl.Message(content="Image not found. PDF will continue without it.").send()
        image_path = None

    pdf = FPDF()
    pdf.add_page()

    try:
        pdf.add_font('TimesNewRoman', '', 'Times New Roman.ttf', uni=True)
        pdf.add_font('TimesNewRomanBold', 'B', 'Times New Roman Bold.ttf', uni=True)
    except RuntimeError:
        await cl.Message(content="Font files not found!").send()
        return

    pdf.set_font('TimesNewRoman', '', 14)
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf_name = "story"
    story_lines = story.split("\n")

    
    for i, line in enumerate(story_lines):
        if line.startswith("##"):
            title = line.strip("#").strip()
            pdf_name = title
            pdf.set_font('TimesNewRomanBold', 'B', 22)
            pdf.cell(0, 10, title, ln=True, align='C')
            pdf.ln(5)
            pdf.set_font('TimesNewRoman', '', 14)
            story_lines = story_lines[i+1:]
            break

    
    if image_path:
        img_w_mm = 100
        img_x = (210 - img_w_mm) / 2  # A4 width = 210mm
        img_y = pdf.get_y()

       
        with Image.open(image_path) as img:
            dpi = 96  # assumed DPI (used for scaling pixels to mm)
            px_w, px_h = img.size
            img_h_mm = (img_w_mm / px_w) * px_h

        pdf.image(image_path, x=img_x, y=img_y, w=img_w_mm)

        
        pdf.set_y(img_y + img_h_mm + 5)

   
    for line in story_lines:
        if line.strip():
            pdf.multi_cell(0, 7, line)
            pdf.ln(1)

    pdf_str = pdf.output(dest='S')
    pdf_bytes = pdf_str.encode('latin1')

    await cl.Message(content="Here's your final PDF with no overlap and clean layout!", elements=[
        cl.File(name=pdf_name + ".pdf", content=pdf_bytes, mime="application/pdf")
    ]).send()
