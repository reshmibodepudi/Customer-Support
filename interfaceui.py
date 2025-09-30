#venv\Scripts\activate
import streamlit as st
import requests
from datetime import datetime
import markdown

# Page configuration
st.set_page_config(
    page_title="DBS AI Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

#css styles
st.markdown("""
<style>
    /* Compact Header */
    .main-header {
        background: linear-gradient(135deg, #d32f2f, #b71c1c);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: sans-serif;
    }
    .main-header h2 {
        margin:0;
        font-size:1.2rem;
    }
            .answer-box {
    background: #ffffff;
    padding: 0.6rem 0.9rem;
    border-radius: 10px;
    margin-top: 0.5rem;
    line-height: 1.5;
    font-size: 0.95rem;
    border-left: 4px solid #d32f2f;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

.answer-box ul {
    margin: 0.4rem 0 0.4rem 1.2rem;
    padding: 0;
}

.answer-box li {
    margin-bottom: 0.3rem;
}

.answer-box strong {
    color: #b71c1c;
}

    /* Chat container */
    .chat-container {
        background: #fdfdfd;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #d32f2f;
    }

    .user-message {
        background: linear-gradient(135deg, #d32f2f, #e57373);
        color: white;
        padding: 0.6rem 0.8rem;
        border-radius: 12px 12px 6px 12px;
        margin: 0.3rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        font-size: 0.95rem;
    }

    .assistant-message {
        background: #f9f9f9;
        color: #1a1a1a;
        padding: 0.6rem 0.8rem;
        border-radius: 12px 12px 12px 6px;
        margin: 0.3rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        font-size: 0.95rem;
    }

    /* Input container */
    .input-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-top: 0.5rem;
    }
    .input-container h4 {
        color: #d32f2f;
        margin: 0 0 0.3rem 0;
    }

    /* Sidebar */
    .sidebar-content {
        background: #f5f5f5;
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #1a1a1a;
    }

    .feature-card {
        background: white;
        color: #1a1a1a;
        padding: 0.6rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 3px solid #d32f2f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        font-size: 0.9rem;
    }

    .stats-container {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
    }

    .stat-item {
        text-align: center;
        padding: 0.5rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        font-size: 0.9rem;
        width: 48%;
        color: #1a1a1a;
    }
            

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #d32f2f, #b71c1c);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.4rem 1.5rem;
        font-weight: bold;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        width: 100%;
        margin-bottom: 0.3rem;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }

    /* Text input */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 1.5px solid #e0e0e0;
        padding: 0.5rem 0.8rem;
        font-size: 0.95rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: #d32f2f;
        box-shadow: 0 0 0 3px rgba(211, 47, 47, 0.1);
    }
</style>
""", unsafe_allow_html=True)

#dates
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'main_input_value' not in st.session_state:
    st.session_state.main_input_value = ""

#header
st.markdown("""
<div class="main-header">
    <h2>🏦 DBS AI Assistant</h2>
    <p>Your intelligent banking companion powered by AI</p>
</div>
""", unsafe_allow_html=True)


#sidebar
with st.sidebar:
    # DBS Logo
    st.image(r"dbs_logo.png", width='stretch')  
    st.markdown(
        '<p style="text-align:center; font-weight:bold; color:#d32f2f; margin-top:0.5rem;">LIVE MORE, BANK LESS</p>',
        unsafe_allow_html=True
    )

    st.markdown("### Features")
    st.markdown("""
    <div class="sidebar-content">
        <div class="feature-card"> Smart Chat: Instant banking answers</div>
        <div class="feature-card"> Knowledge Base: Comprehensive info</div>
        <div class="feature-card"> Secure & Private: Your data is safe</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Chat history")
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-item">{st.session_state.query_count}<br>Queries</div>
        <div class="stat-item">{len(st.session_state.chat_history)}<br>Messages</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.session_state.main_input_value = ""
        st.rerun()

#chat and input area
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 💬 Chat")
    if st.session_state.chat_history:
        for timestamp, user_msg, assistant_msg in st.session_state.chat_history:
            formatted_answer = markdown.markdown(assistant_msg)
            st.markdown(f"""
            <div class="chat-container">
        <div class="user-message">
            <strong>You:</strong> {user_msg}<br><small>{timestamp}</small>
        </div>
        <div class="assistant-message">
            <strong>🏦 DBS Assistant:</strong><br>
            <div class="answer-box">{formatted_answer}</div>
        </div>
    </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-container">
            <div class="assistant-message">
                <strong>🏦 Welcome!</strong><br>Ask me anything about DBS banking.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Ququick actions
with col2:
    st.markdown("### Quick Actions")
    quick_buttons = [
        
        ("💰 Loans & Eligibility", "What types of loans are offered?"),
        ("🏧 Debit & Credit Cards", "What is daily limit of debit card?"),
        ("📱 Customer Care & Support", "What is the DBS customer care number?")
    ]
    for idx, (label, query) in enumerate(quick_buttons):
        if st.button(label, key=f"quick_{idx}"):
            st.session_state.main_input_value = query 

#input
st.markdown('<div class="input-container"><h4>💭 Ask your question</h4></div>', unsafe_allow_html=True)
question = st.text_input(
    "Your question:",
    placeholder="e.g., How do I open a savings account?",
    key="main_input",
    value=st.session_state.main_input_value
)

#submit button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Get Answer", key="submit_btn") and question.strip():
        with st.spinner("🤖 AI is thinking..."):
            try:
                response = requests.post("http://127.0.0.1:8000/ask", json={"question": question})
                answer = response.json().get("answer", "No answer received.")
                timestamp = datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append((timestamp, question, answer))
                st.session_state.query_count += 1
                st.session_state.main_input_value = ""  
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown('<div style="text-align:center; color:#666; padding:0.5rem;">🏦 DBS AI Assistant | Powered by AI</div>', unsafe_allow_html=True)
