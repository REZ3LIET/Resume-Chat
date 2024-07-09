import streamlit as st
from agent.resume_chat import ResumeAgent

def upload_file():
    resume = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)
    if resume is not None:
        with open("./Resume_Chat/resume/resume.pdf", mode='wb') as w:
            w.write(resume.getvalue())
        return True
    return False

def check_start_state():
    if st.session_state.job_desc == "":
        st.error("Enter Job Description")
        return False

    if not st.session_state.file_status:
        st.error("Resume not uploaded!")
        return False

    if st.session_state.agent_type == "---":
        st.error("Choose Chat Option!")
        return False

    st.session_state.start_chat = True
    return True
        

def main():
    if "file_status" not in st.session_state:
        st.session_state.file_status = False
    
    if "start_chat" not in st.session_state:
        st.session_state.start_chat = False
    
    if "job_desc" not in st.session_state:
        st.session_state.job_desc = ""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "first" not in st.session_state:
        st.session_state.first = True

    if "agent_type" not in st.session_state:
        st.session_state.agent_type = "---"

    st.set_page_config(
        page_title="Resume Chat",
        page_icon="👋",
    )

    st.write("# Welcome to Resume-Chat! 👋")
    api_key = st.sidebar.text_input('Gemini API Key', type='password')
    if api_key == "":
        st.warning("Please set the API KEY in sidebar.")
    st.session_state.job_desc = st.text_area("Enter you job description", placeholder="Comapany's Requirements...")
    col1, col2 = st.columns(2)

    with col1:
        st.session_state.file_status = upload_file()

    with col2:
        chat_type = st.selectbox("Choose Preferred Chat Option", ["---", "Improve Resume", "Dummy Interview"])
        if st.session_state.agent_type != chat_type:
            st.session_state.messages = []
            st.session_state.agent_type = chat_type
            if chat_type == "Improve Resume":
                st.session_state.agent = ResumeAgent(api_key, "improve", st.session_state.job_desc)
            elif chat_type == "Dummy Interview":
                st.session_state.agent = ResumeAgent(api_key, "interview", st.session_state.job_desc)
            else:
                st.session_state.agent = None

        st.button("Start Chat", on_click=check_start_state)

    if st.session_state.start_chat:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            response = st.session_state.agent.agent_chat(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
