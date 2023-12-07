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

# Validation Functions
def is_valid_openai_key(api_key) -> bool:
    try:
        # Test OpenAI API key by making a request to the authentication endpoint
        auth_response = requests.get("https://api.openai.com/v1/engines", headers={"Authorization": f"Bearer {api_key}"})
        auth_response.raise_for_status()

        # Test OpenAI Python package by creating an instance of the openai.api object
        openai.api_key = api_key
        openai.Completion.create(engine="text-davinci-002", prompt="Hello, World!")

        # If both tests pass, the API key is valid
        return True

    except (requests.exceptions.HTTPError, Exception):
        # If either test fails, the API key is invalid
        return False
    
def is_valid_youtube_url(url: str) -> bool:
    # Check if the URL is a valid YouTube video URL
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # Check if the video is available
        if not yt.video_id:
            return False

    except (VideoUnavailable, Exception):
        return False

    # Return True if the video is available
    return yt.streams.filter(adaptive=True).first() is not None

# Calculate YouTube video duration
def get_video_duration(url: str) -> float:
    yt = YouTube(url)  
    video_length = round(yt.length / 60, 2)

    return video_length

# Calculate API call cost
def calculate_api_cost(video_length: float, option: str) -> float:
    if option == 'summary':
        api_call_cost = round(video_length * 0.009, 2)
    elif option == 'answer':
        api_call_cost = round(video_length * 0.006, 2)

    return api_call_cost

# Get Video Thumbnail URL & Title
def video_info(url: str):

    yt = YouTube(url)

    # Get the thumbnail URL and title
    thumbnail_url = yt.thumbnail_url
    title = yt.title

    return thumbnail_url, title

# Download YouTube video as Audio
def download_audio(url: str):

    yt = YouTube(url)

    # Extract the video_id from the url
    query = urlparse(url).query
    params = parse_qs(query)
    video_id = params["v"][0]

    # Get the first available audio stream and download it
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(output_path="tmp/")

    # Convert the downloaded audio file to mp3 format
    audio_path = os.path.join("tmp/", audio_stream.default_filename)
    audio_clip = AudioFileClip(audio_path)
    audio_clip.write_audiofile(os.path.join("tmp/", f"{video_id}.mp3"))

    # Delete the original audio stream
    os.remove(audio_path)

# Transcription 
def transcribe_audio(file_path, video_id):
        # The path of the transcript
        transcript_filepath = f"tmp/{video_id}.txt"

        # Get the size of the file in bytes
        file_size = os.path.getsize(file_path)

        # Convert bytes to megabytes
        file_size_in_mb = file_size / (1024 * 1024)

        # Check if the file size is less than 25 MB
        if file_size_in_mb < 25:
            with open(file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                
                # Writing the content of transcript into a txt file
                with open(transcript_filepath, 'w') as transcript_file:
                    transcript_file.write(transcript['text'])

            # Deleting the mp3 file
            os.remove(file_path)

        else:
            print("Please provide a smaller audio file (less than 25mb).")

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

# generate practice problems
def generate_practice_problems_for_topics(topics):
    problems_with_solutions = {}
    for topic in topics:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content":"Create practice problems and solutions for this topic: " + topic}],
            max_tokens=200,
            stop=None,
            temperature=0.7,
        )
        latest_response = response['choices'][0]['message']['content']
        problems_with_solutions[topic] = latest_response
    return problems_with_solutions

# refresh gpt prompt
def refresh_prompt():
    global prompt
    prompt = []
    global question_count
    question_count = 0

# Generate Youtube Answer
def generate_answer_youtube(api_key: str, url: str, question: str) -> str:

    global question_count
    question_count += 1
    
    openai.api_key = api_key

    # Extract the video_id from the url
    query = urlparse(url).query
    params = parse_qs(query)
    video_id = params["v"][0]

    # The path of the transcript
    transcript_filepath = f"tmp/{video_id}.txt"

    # Check if the transcript file already exist
    if os.path.exists(transcript_filepath):
        global prompt
        if question_count == 1:
            download_audio(url)
            transcribe_audio(transcript_filepath,video_id)
            with open(transcript_filepath, 'r', encoding='utf8') as file:
                file_content = file.read()
            print("BEFORE:\n", file_content)
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

    for choice in response.choices:
        if "text" in choice:
            prompt.append({"role":"assistant", "content":choice.text})
            return choice.text

    # If no response with text is found, return the first response's content (which may be empty)
    prompt.append({"role":"assistant", "content":response.choices[0].message.content})
    return response.choices[0].message.content
    #return answer.strip()
 
 
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
    print("FILE: ", os.path.exists(transcript_filepath))
    
    # Check if the transcript file already exist
    if os.path.exists(transcript_filepath):
        global prompt
        if question_count == 1:
            with open(transcript_filepath, 'r', encoding='utf8') as file:
                file_content = file.read()
            # question.append({"role":"user", "content":"Please provide me the summary of this video."})            print("BEFORE:\n", file_content)
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
    #return answer.strip()
 
# Generating Video Summary 
@st.cache_data(show_spinner=False)
def generate_summary(api_key: str, url: str) -> str:

    openai.api_key = api_key

    llm = OpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-3.5-turbo")
    text_splitter = CharacterTextSplitter()

    # Extract the video_id from the url
    query = urlparse(url).query
    params = parse_qs(query)
    video_id = params["v"][0]

    # The path of the audio file
    audio_path = f"tmp/{video_id}.mp3"

    # The path of the transcript
    transcript_filepath = f"tmp/{video_id}.txt"

    # Check if the transcript file already exist
    if os.path.exists(transcript_filepath):
        # Generating summary of the text file
        with open(transcript_filepath) as f:
            transcript_file = f.read()

        texts = text_splitter.split_text(transcript_file)
        docs = [Document(page_content=t) for t in texts[:3]]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
    
    else: 
        download_audio(url)

        # Transcribe the mp3 audio to text
        transcribe_audio(audio_path, video_id)

        # Generating summary of the text file
        with open(transcript_filepath) as f:
            transcript_file = f.read()

        texts = text_splitter.split_text(transcript_file)
        docs = [Document(page_content=t) for t in texts[:3]]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)

    return summary.strip()