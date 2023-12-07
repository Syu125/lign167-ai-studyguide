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
# import threading
# from constants import APIKEY
# import queue

from gptube import generate_answer_youtube, generate_answer_transcript, generate_summary, video_info, is_valid_openai_key, is_valid_youtube_url, get_video_duration, calculate_api_cost, summarize_with_gpt3, extract_topics_from_summary, generate_definitions_for_topics, generate_practice_problems_for_topics
from elevenlabs import generate, set_api_key

st.set_page_config(page_title="Interactive Content")

set_api_key(st.secrets["APIKEY"])

# Define passwords (in a real app, use a more secure method)
ADMIN_PASSWORD = "admin_pass"
USER_PASSWORD = "user_pass"

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

# if 'in_thread' not in st.session_state:    
#     st.session_state['in_thread'] = False
    
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

# Admin UI
def show_admin_ui():
    st.title("Admin Dashboard")

    # Form for adding new topic, textbook section, and date
    st.subheader("Add New Topic, Corresponding Textbook Section, and Date")
    with st.form(key='topic_textbook_form'):
        new_topic = st.selectbox("Select a chapter", chapters, index=7)
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

def extract_specific_section(pdf_path, start_section, end_section=None):
    document = fitz.open(pdf_path)
    st.markdown("[DOCUMENT]\n", document)
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
    
    st.markdown("[EXTRACTED TEXT]\n", extracted_text)
    document.close()
    return extracted_text

# Function to clear the topic list
def clear_topic_list():
    open("topics_and_sections.txt", "w").close()  # This will clear the file content
    
# Function to add a new topic and corresponding textbook section
def add_topic_section_date(topic, topic_date):
    with open("topics_and_sections.txt", "a") as file:
        file.write(f"{topic_date.isoformat()}|{topic}\n")

# Function to display existing topics and textbook sections
def display_topics_and_sections_ordered():
    try:
        with open("topics_and_sections.txt", "r") as file:
            # Parse the file into a DataFrame
            data = [line.strip().split('|') for line in file.readlines()]
            if data:
                df = pd.DataFrame(data, columns=['Date', 'Topic'])
                # Convert 'Date' to datetime for proper sorting
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Sort the DataFrame by the 'Date' column
                df = df.sort_values(by='Date')

                # Reset index to start from 1 after sorting
                df = df.reset_index(drop=True)
                df.index = df.index + 1

                # Format the date for display
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

                st.table(df)  # Display as a table
            else:
                st.write("No topics and sections added yet.")
    except FileNotFoundError:
        st.write("No topics and sections added yet.")
        
# User UI
generated_curriculum = None
pdf_topic = None
# download_button = st.empty()
# result_queue = queue.Queue()
def show_user_ui():
    topics = load_topics()
    
    if 'pdf_buffer' not in st.session_state:
        st.session_state['pdf_buffer'] = st.empty()
    
    def generate_pdf(pdf_path, selected_topic, download_button):
        progress_bar = st.progress(0)
        selected_topic = selected_topic[selected_topic.index("|") + 2:]
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
        print(section_text)
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
        download_button = st.download_button(
            label="Download PDF",
            data=generated_curriculum,
            file_name=f"StudyGuide_{selected_topic}.pdf",
            mime="application/pdf"
        )
        progress_bar.progress(100)
    
    # Display the dropdown and handle topic selection
    if topics:
        st.markdown("## Create Study Guide for Specific Topic")
        selected_topic = st.selectbox("Topic:", topics)
        generate_button = st.button("Generate Study Guide")
        if selected_topic != "None" and generate_button:
            pdf_path = './streamlit/Goldberg.pdf'
            generate_pdf(pdf_path, selected_topic, generate_button)
    else:
        st.write("No topics available right now.")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = dict()
    
    # Function to update chat history
    def update_chat_history(key, user_message, response):
        if key not in st.session_state['chat_history']:
            st.session_state['chat_history'].setdefault(key, [])
        st.session_state['chat_history'][key].append(("user", user_message))
        st.session_state['chat_history'][key].append(("assistant", response))

    if st.session_state['chat_history']:
        selected_key = st.sidebar.selectbox(
            'Select Chat History',
            list(st.session_state['chat_history'].keys())
        )
        with st.expander(f"Chat History: {selected_key}"):
            for speaker, message, in st.session_state['chat_history'][selected_key]:
                st.markdown(speaker)
                st.success(message)
            

    def display_typing_effect_markdown(message, delay=0.1):
        placeholder = st.empty()
        display_text = ""
        for char in message:
            display_text += char
            placeholder.markdown(display_text)
            time.sleep(delay)

    def display_typing_effect_success(message, delay=0.01):
        placeholder = st.empty()
        display_text = ""
        for char in message:
            display_text += char
            placeholder.success(display_text)
            time.sleep(delay)
            
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

        st.markdown('## or Chat with Your Lecture') 

        st.markdown('######') 
                
        # OPENAI API KEY
        openai_api_key = st.secrets["APIKEY"]
        
        # Disable YouTube URL field until OpenAI API key is valid
        if openai_api_key:
            st.markdown('#### Step 1 : Enter your YouTube Video or Lecture Transcript')

            youtube_url = st.text_input("URL :", placeholder="https://www.youtube.com/watch?v=************")
            uploaded_file = st.file_uploader("Text File:", type=['txt'])

        if openai_api_key and (youtube_url or uploaded_file):
            st.markdown('#### Step 2 : Enter your Question')
            question = st.text_input("What can I help you with?", placeholder="Enter your question here")
        else:
            st.markdown('#### Step 2 : Enter your Question')
            st.write("Please enter a YouTube URL or upload a file first!")
        
        # Add a button to run the function
        if st.button("Send"):
            if not question:
                st.warning("Please enter a question.")
            else:
                # os.environ["OPENAI_API_KEY"] = openai_api_key
                with st.spinner("Generating answer..."):
                    # Call the function with the user inputs
                    context = []
                    prompt = context
                    prompt.append({"role":"user", "content": question})
                    if youtube_url:
                        if is_valid_youtube_url(youtube_url):
                            key_input = youtube_url
                            answer = generate_answer_youtube(openai_api_key, youtube_url, prompt)
                        else:
                            st.warning("Youtube URL is invalid.")
                    elif uploaded_file is not None:
                        key_input = uploaded_file.name
                        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                        transcript = stringio.read()
                        answer = generate_answer_transcript(openai_api_key, transcript, prompt)
                        
                    update_chat_history(key_input, question, answer)
                
            for speaker, message in st.session_state['chat_history'][key_input][:-1]:
                if speaker == "user":
                    st.markdown(f"#### {message}")
                elif speaker == "assistant":
                    st.success(message)
            last_speaker = st.session_state['chat_history'][key_input][-1][0]
            last_message = st.session_state['chat_history'][key_input][-1][1]
            if last_speaker == "user":
                display_typing_effect_markdown(last_message)
            elif last_speaker == "assistant":
                display_typing_effect_success(last_message)
    youtube_app()
    
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
    
# if not st.session_state['in_thread']:
#     st.download_button(
#         label="Download PDF",
#         data=result_queue.get(),
#         file_name=f"StudyGuide_{result_queue.get()}.pdf",
#         mime="application/pdf"
#     )

# Hide Left Menu
st.markdown("""<style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)



