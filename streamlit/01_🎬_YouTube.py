import streamlit as st
import tempfile
import pyttsx3
import os
import time
from io import StringIO
from datetime import date
import pandas as pd
import fitz
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import base64
from google.cloud import storage
from google.oauth2 import service_account
# import threading
# from constants import APIKEY
# import queue

from gptube import generate_answer_youtube, generate_answer_transcript, generate_summary, video_info, is_valid_openai_key, is_valid_youtube_url, get_video_duration, calculate_api_cost, summarize_with_gpt3, extract_topics_from_summary, generate_definitions_for_topics, generate_practice_problems_for_topics, refresh_prompt
from elevenlabs import generate, set_api_key

st.set_page_config(page_title="Interactive Content")

set_api_key(st.secrets["APIKEY"])

# Authenticate to GCS
credentials = service_account.Credentials.from_service_account_file('./streamlit/elemental-icon-407905-833146d629c5.json')
client = storage.Client(credentials=credentials)

# Function to upload file to GCS
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    # Return a public URL to the file
    return blob.public_url

def get_pdf_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    if not blob.exists():
        print("File does not exist in bucket")
        return None

    return blob.public_url

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob("transcripts/" + source_blob_name)

    blob.download_to_filename(destination_file_name)
    
bucket_name = 'lign167-pdf-storage'

# Define passwords (in a real app, use a more secure method)
ADMIN_PASSWORD = "admin_pass"
USER_PASSWORD = "user_pass"

pdf_storage_path = './streamlit/pdf_storage'
transcript_storage_path = './streamlit/transcript_storage'

if 'generated_pdfs' not in st.session_state:
    st.session_state['generated_pdfs'] = {}
    
if 'transcripts' not in st.session_state:
    st.session_state['transcripts'] = {}
    
if not os.path.exists(pdf_storage_path):
    os.makedirs(pdf_storage_path)
    
# Sidebar for login
password = st.sidebar.text_input("Enter your password", type="password")

# Initialize a session state variable for login status
if 'login_status' not in st.session_state:
    st.session_state['login_status'] = None

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    
if 'pdf_generated' not in st.session_state:
    st.session_state['pdf_generated'] = False    
    
# Check password
if password:
    if password == ADMIN_PASSWORD:
        st.session_state['logged_in'] = True
        st.session_state['login_status'] = "admin"
    elif password == USER_PASSWORD:
        st.session_state['logged_in'] = True
        st.session_state['login_status'] = "user"
    else:
        st.sidebar.error("Incorrect password")


def parse_chapters_from_file(file_path):
    with open(file_path, 'r') as file:
        chapters = file.read().splitlines()
    return chapters

sections_filepath = './streamlit/sections.txt'
chapters = parse_chapters_from_file(sections_filepath)

def load_topics():
    try:
        with open("topics_and_sections.txt", "r") as file:
            # Create a formatted string for each line
            formatted_topics = [' | '.join(line.strip().split('|')) for line in file.readlines()]
            return formatted_topics
    except FileNotFoundError:
        return []

def create_chat_history_doc(key_input):
    buffer = io.BytesIO()
    title = f"Chat History | {key_input}"
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20, title=title)
    styles = getSampleStyleSheet()
    custom_style = styles['Normal']
    custom_style.fontName = "Times-Roman"
    Story = [Spacer(1, 2)]
    if key_input in st.session_state['chat_history']: 
        for speaker, message, in st.session_state['chat_history'][key_input]:
            p1 = Paragraph(speaker, styles["Normal"])
            p2 = Paragraph(message, styles["Normal"])
            Story.append(p1)
            Story.append(p2)
            Story.append(Spacer(1, 12))
    doc.build(Story)
    buffer.seek(0)
    return buffer 

def create_curriculum_pdf(curriculum_content, topic):
    buffer = io.BytesIO()
    title = f"Study Guide | {topic}"
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20, title=title)
    styles = getSampleStyleSheet()
    custom_style = styles['Normal']
    custom_style.fontName = "Times-Roman"
    Story = [Spacer(1, 2)]
    for line in curriculum_content.split('\n'):
        p = Paragraph(line, styles["Normal"])
        Story.append(p)
        Story.append(Spacer(1, 12))

    doc.build(Story)
    buffer.seek(0)
    return buffer

def generate_pdf(pdf_path, selected_topic, index):
    global pdf_storage_path
    pdf_filename = selected_topic + ".pdf"
    pdf_file_path = os.path.join(pdf_storage_path, pdf_filename)
    progress_bar = st.progress(0)
    # result_queue.put(selected_topic)
    
    topic_index = chapters.index(selected_topic)
    next_topic = ""
    if (topic_index + 1) < len(chapters):
        next_topic = chapters[topic_index + 1]
    topic_key = selected_topic.replace(" ", "\n", 1)
    next_topic_key = ""
    if " " in next_topic:
        next_topic_key = next_topic.replace(" ", "\n", 1)
    progress_bar.progress(10)
    section_text = extract_specific_section(pdf_path, topic_key, next_topic_key)
    progress_bar.progress(20)
    openai_api_key = st.secrets["APIKEY"]
    summarized_section = summarize_with_gpt3(openai_api_key, section_text)
    progress_bar.progress(40)
    key_concepts = extract_topics_from_summary(summarized_section)
    progress_bar.progress(60)
    details = generate_definitions_for_topics(key_concepts)
    curriculum = ""
    for concept in key_concepts:
        curriculum += f"{details.get(concept, 'N/A')}\n"
    progress_bar.progress(80)
    generated_curriculum = create_curriculum_pdf(curriculum, selected_topic)
    generated_curriculum.seek(0)
    with open(pdf_file_path, 'wb') as pdf_file:
        pdf_file.write(generated_curriculum.read())
    progress_bar.progress(100)
    return pdf_file_path

def extract_specific_section(pdf_path, start_section, end_section=None):
    document = fitz.open(pdf_path)
    extracted_text = ""
    in_section = False

    for page in document:
        text = page.get_text()
        # Check for the start of the section
        if start_section in text and not in_section:
            start_index = text.find(start_section) + len(start_section)
            text = text[start_index:]  # Start capturing text after the section header
            in_section = True
        # Check for the end of the section
        if end_section and end_section in text and in_section:
            end_index = text.find(end_section)
            extracted_text += text[:end_index]  # Stop capturing text at the start of the next section
            break
        # Append text if we're in the correct section
        if in_section:
            extracted_text += text
    
    document.close()
    return extracted_text

# Function to clear the topic list
def clear_topic_list():
    open("topics_and_sections.txt", "w").close()  # This will clear the file content
    
# Function to add a new topic and corresponding textbook section
def add_topic_section_date(topic, topic_date):
    with open("topics_and_sections.txt", "a") as file:
        file.write(f"{topic_date.isoformat()}|{topic}\n")

def show_pdf(file_path):
    # with open(file_path, "rb") as f:
    #     base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    # st.markdown(pdf_display, unsafe_allow_html=True)
    st.markdown(f'<iframe src="{file_path}" width="700" height="1000"></iframe>', unsafe_allow_html=True)

# Function to display existing topics and textbook sections
def display_topics_and_sections_ordered():

    try:
        with open("topics_and_sections.txt", "r") as file:
            data = [line.strip().split('|') for line in file.readlines()]
            if data:
                df = pd.DataFrame(data, columns=['Date', 'Topic'])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values(by='Date')
                df = df.reset_index(drop=True)
                df.index = df.index + 1
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

                for index, row in df.iterrows():
                    col1, col2, col3 = st.columns([3, 7, 5])

                    with col1:
                        st.write(row['Date'])
                    
                    with col2:
                        st.write(row['Topic'])

                    with col1:
                        file_path = ""
                        public_url = ""
                        topic = row['Topic']
                        if topic not in st.session_state['generated_pdfs'] or not st.session_state['generated_pdfs'][topic]:
                            file_path = generate_pdf('./streamlit/Goldberg.pdf', topic, index)
                            public_url = upload_to_gcs(bucket_name, file_path, f"pdfs/{topic}")
                            st.session_state['generated_pdfs'][topic] = True
                        else:
                            file_path = f"pdfs/{topic}"
                            public_url = get_pdf_file(bucket_name, file_path) + f"?{int(time.time())}"
                            # file_path = os.path.join(pdf_storage_path, topic + ".pdf")
                            
                        with st.expander(f"View"):
                            show_pdf(public_url)
                            
                    with col3:    

                        with st.expander(f"Update Study Guide PDF"):
                            uploaded_file = st.file_uploader(f"Upload here:", type="pdf", key=f"file_uploader_{index}")
                            file_path = os.path.join(pdf_storage_path, row['Topic'] + '.pdf')
                            if uploaded_file is not None:
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                upload_to_gcs(bucket_name, file_path, f"pdfs/{row['Topic']}")
                                
                                st.session_state['generated_pdfs'][topic] = True
                                st.success("PDF updated successfully!")
                                st.button(label='Refresh', key=f"refresh_{index}")
                        
                        with st.expander(f"Upload Transcript"):
                            uploaded_file = st.file_uploader(f"Upload here:", type="txt", key=f"transcript_{index}")
                            transcript_path = os.path.join(transcript_storage_path, row['Topic'] + ".txt")
                            if uploaded_file is not None:
                                file_content = uploaded_file.getvalue().decode("utf-8")
                                with open(transcript_path, "a") as f:
                                    st.markdown("HERE")
                                    f.write(file_content)
                                upload_to_gcs(bucket_name, transcript_path, f"transcripts/{row['Topic']}")
                                st.session_state['transcripts'][uploaded_file.name] = True
                                st.success("Transcript uploaded!")
                            
            else:
                st.write("No topics and sections added yet.")

    except FileNotFoundError:
        st.write("No topics and sections added yet.")
        
# Admin UI
def show_admin_ui():
    st.title("Admin Dashboard")

    # Form for adding new topic, textbook section, and date
    st.subheader("Add New Topic: Corresponding Textbook Section, and Date")
    with st.form(key='topic_textbook_form'):
        new_topic = st.selectbox("Select a chapter", chapters, index=8)
        topic_date = st.date_input("Date", date.today())
        submit_button = st.form_submit_button(label='Add')

        if submit_button and new_topic:
            add_topic_section_date(new_topic, topic_date)

    # Button to clear the topic list
    if st.button("Clear Topic List"):
        clear_topic_list()
        st.success("Topic list cleared!")

    # Display existing topics and textbook sections ordered by date
    st.subheader("Current Topics and Textbook Sections")
    display_topics_and_sections_ordered()

# User UI
generated_curriculum = None
pdf_topic = None
previous_file = None
def show_user_ui():
    
    # Function to update chat history
    def update_chat_history(key, user_message, response):
        if key not in st.session_state['chat_history']:
            st.session_state['chat_history'].setdefault(key, [])
        st.session_state['chat_history'][key].append(("user", user_message))
        st.session_state['chat_history'][key].append(("assistant", response))

    def display_typing_effect_success(message, delay=0.01):
        placeholder = st.empty()
        display_text = ""
        for char in message:
            display_text += char
            placeholder.success(display_text)
            time.sleep(delay)
    
    def show_specific_chat(key_input):
        global previous_file
        if previous_file != key_input:
            if key_input in st.session_state['chat_history']:
                second_last_message = st.session_state['chat_history'][key_input][-2][1]
                st.markdown(f"### {second_last_message}")
                last_message = st.session_state['chat_history'][key_input][-1][1]
                display_typing_effect_success(last_message)
                for i in range(len(st.session_state['chat_history'][key_input])-3, -1, -2):
                    message = st.session_state['chat_history'][key_input][i][1]
                    prev_message = st.session_state['chat_history'][key_input][i-1][1]
                    st.markdown(f"### {prev_message}")
                    st.success(message)   
    
    topics = load_topics()
    
    if 'pdf_buffer' not in st.session_state:
        st.session_state['pdf_buffer'] = st.empty()                
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = dict()
        
    if 'select_chat' not in st.session_state:
        st.session_state['select_chat'] = st.empty() 
    
    if st.session_state['select_chat']:
        selected_chat = st.session_state['select_chat']  
        if selected_chat in st.session_state['chat_history']: 
            chat_pdf = create_chat_history_doc(selected_chat)
            download_button = st.download_button(
                label="Download Chat History",
                data=chat_pdf,
                file_name=f"ChatHistory_{selected_chat}.pdf",
                mime="application/pdf"
            )

    # App UI
    def youtube_app():

        with st.sidebar:
            st.markdown("<h2 style='color: black;'>Study Enhancer Tool: Your Companion for In-Depth Learning</h4>", unsafe_allow_html=True)

            # st.markdown("<h2 style='color: white;'>How It Works</h4>", unsafe_allow_html=True)
            summary = """

            <p style='color: white;'>Welcome to the Study Enhancer Tool, an innovative solution designed to assist in your educational journey. Whether you're diving into a complex YouTube tutorial or navigating through an intricate lecture, our tool is here to streamline your learning experience.</p>

            <h3 style='color: white;'>How It Works:</h3>
            <ol style='color: white;'>
                <li><strong>Input Your Learning Material</strong>: Start by providing the source of your study material. You have two convenient options:
                    <ul>
                        <li><strong>YouTube Video</strong>: Insert the URL of a YouTube video related to your study topic.</li>
                        <li><strong>Lecture Transcript</strong>: Upload a transcript text file from a lecture you're reviewing.</li>
                    </ul>
                </li>
                <li><strong>Content Summarization</strong>: Our tool efficiently processes your input, distilling the essential information. It creates a concise summary of the video or lecture, making it easier for you to grasp the core concepts without the fluff.</li>
                <li><strong>Interactive Q&A</strong>: Got questions? Just ask! The tool is equipped to answer specific queries you have about the material, offering detailed explanations to enhance your understanding.</li>
                <li><strong>Problems and Solutions</strong>: If you need to test your knowledge or tackle practice problems, our tool has got you covered. It provides relevant problems and solutions, aiding in reinforcing your learning.</li>
            </ol>
            <p style='color: white;'>Whether you're preparing for an exam, researching a topic, or simply exploring new subjects, the <strong>Study Enhancer Tool</strong> is your go-to resource for an enriched learning experience. Dive in and transform the way you learn!</p>
            """

            st.markdown(summary, unsafe_allow_html=True)

            
            st.markdown("####")

            st.markdown("<p style='color: white;'>💻 Resource Credit [GitHub](https://github.com/Hamagistral/GPTube)</p>", unsafe_allow_html=True)
            
        st.markdown('#####')
        if topics:
            st.markdown("## Create Study Guide for Specific Topic")
            selected_topic = st.selectbox("Topic:", topics).split("| ")[1]
            file_path = f"pdfs/{selected_topic}"
            pdf_url = get_pdf_file(bucket_name, file_path) + f"?{int(time.time())}"
            with st.expander(f"View"):    
                show_pdf(pdf_url)    
        else:
            st.write("No topics available right now.")
        
        st.markdown('## or Chat with Your Lecture') 

        st.markdown('######') 
                
        # OPENAI API KEY
        openai_api_key = st.secrets["APIKEY"]

        # Disable YouTube URL field until OpenAI API key is valid
        if openai_api_key:

            transcript_filepath = os.path.join(transcript_storage_path, selected_topic + ".txt")
            download_blob(bucket_name, selected_topic, transcript_filepath)
            uploaded_file = None
            if os.path.exists(transcript_filepath):
                uploaded_file = open(transcript_filepath, 'r')

        if openai_api_key and uploaded_file:
            st.markdown('#### Enter your Question')
            question = st.text_input("What can I help you with?", placeholder="Enter your question here")
        else:
            st.markdown('#### Enter your Question')
            st.write("There is no transcript for this topic, so please select another topic!")
        
        # Add a button to run the function
        if st.button("Send"):
            if not question:
                st.warning("Please enter a question.")
            else:
                # os.environ["OPENAI_API_KEY"] = openai_api_key
                if uploaded_file.name not in st.session_state['chat_history']:
                    st.session_state['chat_history'][uploaded_file.name] = []
                global previous_file
                if (previous_file != uploaded_file):
                    refresh_prompt()
                with st.spinner("Generating answer..."):
                    # Call the function with the user inputs
                    context = []
                    for speaker, message in st.session_state['chat_history'][uploaded_file.name]:
                        if speaker == "user":
                            context.append({"role":"user", "content": message})
                        elif speaker == "assistant":
                            context.append({"role":"assistant", "content": message})
                    prompt = context
                    prompt.append({"role":"user", "content": question})
                    if uploaded_file is not None:
                        key_input = uploaded_file.name
                        transcript = uploaded_file.read()
                        answer = generate_answer_transcript(openai_api_key, transcript, prompt)
                    
                    previous_file = uploaded_file
                    update_chat_history(key_input, question, answer)
                
            show_specific_chat(key_input)
            
    youtube_app()
    
if 'chat_history' in st.session_state:
    if not st.session_state['chat_history']:
        print("No chat history")
        st.session_state['select_chat'] = st.markdown("#### No Chat History")
    else:
        options = {}
        options['Select a chat history...'] = ''
        options.update(st.session_state['chat_history'])
        st.session_state['select_chat'] = st.selectbox(
            'Select Chat History',
            list(options)
        )
# Display UI based on login status
if not st.session_state['logged_in']:
    st.title("Welcome to your LIGN 167 Assistant!")
    st.write("Please open the sidebar to log in.")
elif st.session_state['login_status'] == "admin":
    show_admin_ui()
elif st.session_state['login_status'] == "user":
    show_user_ui()
else:
    st.sidebar.write("Please log in")

# Hide Left Menu
st.markdown("""<style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)



