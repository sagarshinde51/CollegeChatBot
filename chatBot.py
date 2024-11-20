import streamlit as st
import os
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import PyPDF2

# Initialize session state for chat history and message display
if "messages" not in st.session_state:
    st.session_state.messages = []

if "display_message" not in st.session_state:
    st.session_state.display_message = ""

# Custom CSS for styling the page (fixed buttons, sticky input container, and title animation)
st.markdown(
    """
    <style>
    /* Disable Streamlit's default header and footer */
    header {display: none;}
    footer {display: none;}

    /* Sticky input container at the bottom */
    .sticky-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 10px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        z-index: 1000;
    }

    /* Sidebar styling - adjusting top padding to avoid overlap with fixed buttons */
    .sidebar .sidebar-content {
        padding-top: 60px;
    }

    /* Keyframes for title animation (fade-in effect) */
    @keyframes fadeIn {
        0% {
            opacity: 0;
        }
        100% {
            opacity: 1;
        }
    }

    /* Apply the fadeIn animation to the title */
    .title {
        animation: fadeIn 2s ease-in-out;
        font-size: 3em;  /* Make title bigger */
        font-weight: bold;  /* Make title bold */
        background: linear-gradient(to right, #ff7e5f, #feb47b); /* Linear gradient from red to yellow */
        -webkit-background-clip: text; /* Clip background to text */
        color: transparent; /* Make text transparent to show the background gradient */
    }

    /* Add some styling for the enter message label */
    .enter-message-label {
        font-size: 1.1em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply the animated and gradient class to the title
st.markdown('<h1 class="title">College ChatBot System</h1>', unsafe_allow_html=True)

# Display login, signup buttons, and image in one row in the sidebar
with st.sidebar:
    st.header("Welcome to SIT ChatBot")
    
    # Create three columns: one for the image and two for buttons
    col1, col2, col3 = st.columns([0.5, 0.5, 0.5])  # Equal width columns for buttons and image
    
    # Image in the first column
    with col1:
        logo_path = r"E://logo.jpeg"  # Update this path to your image file on the E drive
        if os.path.exists(logo_path):
            st.image(logo_path, width=70)  # Display image with width of 70 pixels
        else:
            st.error("Logo file not found at the specified path.")
    
    # Login button in the second column
    with col2:
        login_button = st.button("Login")
    
    # Signup button in the third column
    with col3:
        signup_button = st.button("Signup")
    
    # Handle Login and Signup actions
    if login_button:
        # Show login form
        with st.form("login_form"):
            st.text_input("Email")
            st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login")
            if login_submit:
                st.sidebar.write("Logged in successfully!")
    
    if signup_button:
        # Show signup form
        with st.form("signup_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            signup_submit = st.form_submit_button("Sign Up")
            
            if signup_submit:
                if password == confirm_password:
                    st.sidebar.write("Registration successful! You can now login.")
                else:
                    st.sidebar.write("Passwords do not match. Please try again.")
    
    # Display Chat History in the sidebar
    st.header("Chat History")
    for msg in st.session_state.messages:
        st.write(f"**You:** {msg['content']}")

# Create an empty container for the sticky input area
input_container = st.empty()

# Load pre-trained BERT model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Extract PDF content (use your actual path here)
pdf_path = "data.pdf"  # Update this to the actual path of your PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Create columns for the user input text field and the "Send" button
col1, col2 = st.columns([4, 1])  # Adjust column width ratio to suit the layout

# Input text box in the first column
with col1:
    user_input = st.text_input("Your Question:", "")

# Send button in the second column
with col2:
    send_button = st.button("Send")

# Handle the "Send" button click
if send_button and user_input:
    # Add the user's message to session state history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Tokenize the question
    question_inputs = tokenizer.encode(user_input, add_special_tokens=True)

    # Split the context (PDF text) into chunks ensuring it stays within token limits
    max_chunk_len = 512 - len(question_inputs) - 2  # 2 for [CLS] and [SEP]
    context_tokens = tokenizer.encode(pdf_text, add_special_tokens=False)

    chunks = [context_tokens[i:i+max_chunk_len] for i in range(0, len(context_tokens), max_chunk_len)]

    best_answer = ""
    best_score = float('-inf')

    # Process each chunk separately
    for chunk in chunks:
        # Combine question tokens and context chunk
        inputs = tokenizer.encode_plus(user_input, chunk, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)

        # Get the model outputs
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Get the most likely start and end of the answer
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        # Extract and decode the answer
        answer_tokens = inputs["input_ids"][0][start_index:end_index+1]
        answer = tokenizer.decode(answer_tokens)

        # Calculate score (sum of logits for start and end positions)
        score = start_scores[0][start_index].item() + end_scores[0][end_index].item()

        # Keep track of the best answer
        if score > best_score:
            best_score = score
            best_answer = answer

    # Display the repeated answer
    st.session_state.display_message = best_answer

# Display the sticky input container with the user input history and the answer message
input_container = st.empty()

with input_container.container():
    st.markdown('<div class="sticky-input-container">', unsafe_allow_html=True)

    # Display the repeated answer after clicking "Send"
    if st.session_state.display_message:
        st.markdown(f"<div style='font-size:20px; color: green;'>{st.session_state.display_message}</div>", unsafe_allow_html=True)

    # Close the sticky input container div
    st.markdown('</div>', unsafe_allow_html=True)
