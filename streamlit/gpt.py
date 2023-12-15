import os 
import re
import openai
import requests
import streamlit as st
import hashlib
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')

from pytube.exceptions import VideoUnavailable
from urllib.parse import urlparse, parse_qs
from moviepy.editor import *
from pytube import YouTube

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

question_count = 0
prompt = []

def summarize_with_gpt3(api_key: str, text):
    openai.api_key = api_key   
    max_sentences = 20
    sentences = sent_tokenize(text)
    key_sentences = sentences[:max_sentences//2] + sentences[-max_sentences//2:]
    text_shortened = ''.join(key_sentences)
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content":"Summarize the following text: " + text_shortened}],
        max_tokens=200,
        stop=None,
        temperature=0.7,
    )       
    for choice in summary.choices:
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
 
# extract topics
def extract_topics_from_summary(summary):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content":"Please only list the topics for this summary (one per line): " + summary}],
        max_tokens=200,
        stop=None,
        temperature=0.7,
    )          
    latest_response = response['choices'][0]['message']['content']
    topics = latest_response.strip().split('\n')  
    return topics   

# generate definitions
def generate_definitions_for_topics(topics):
    definitions = {}
    for topic in topics:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content":"Reiterate the topic, define the following topic, and provide practice questions with answers too: " + topic}],
            max_tokens=200,
            stop=None,
            temperature=0.7,
        )          
        latest_response = response['choices'][0]['message']['content']
        definitions[topic] = latest_response
    return definitions

# refresh gpt prompt
def refresh_prompt():
    global prompt
    prompt = []
    global question_count
    question_count = 0

 
 # Generate Answer from Transcript
def generate_answer_transcript(api_key: str, transcript: str, question: str) -> str:

    global question_count
    question_count += 1
    
    openai.api_key = api_key

    # The path of the transcript
    transcript_id = hashlib.sha256(transcript.encode()).hexdigest()
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    transcript_filepath = os.path.join("tmp/", f"{transcript_id}.txt")
    with open(transcript_filepath, 'w') as file:
        file.write(transcript)
    
    # Check if the transcript file already exist
    if os.path.exists(transcript_filepath):
        global prompt
        if question_count == 1:
            with open(transcript_filepath, 'r', encoding='utf8') as file:
                file_content = file.read()
            max_sentences = 150
            sentences = sent_tokenize(file_content)
            key_sentences = sentences[:max_sentences//2] + sentences[-max_sentences//2:]
            file_content = ''.join(key_sentences)
            print("AFTER:\n", file_content)
            summary = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user", "content":"Summarize the following video transcript: " + file_content}],
                max_tokens=200,
                stop=None,
                temperature=0.7,
            )    
            for choice in summary.choices:
                if "message" in choice and "content" in choice["message"]:
                    prompt.append({"role": "assistant", "content": choice["message"]["content"]})
                
        prompt.append({"role":"user", "content": question[-1]["content"]})
        print(prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
            messages=prompt,   # The conversation history up to this point, as a list of dictionaries
            max_tokens=3000,        # The maximum number of tokens (words or subwords) in the generated response
            stop=None,              # The stopping sequence for the generated response, if any (not used here)
            temperature=0.7,        # The "creativity" of the generated response (higher temperature = more creative)
        )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "text" in choice:
            prompt.append({"role":"assistant", "content":choice.text})
            return choice.text

    # If no response with text is found, return the first response's content (which may be empty)
    prompt.append({"role":"assistant", "content":response.choices[0].message.content})
    return response.choices[0].message.content
