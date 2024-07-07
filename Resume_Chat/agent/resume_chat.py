from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

class ResumeAgent:
    def __init__(self, agent_type, job_summary):
        llm = Ollama(model="llama2")
        print("Model Loaded")

        self.job_summary = job_summary
        retriver = self.data_loader()
        context_prompt = self.contextualize_history()
        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriver,
            context_prompt
        )
        qa_prompt = self.get_system_prompt(agent_type)
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        self.chat_history = {}
        self.chat_model = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        print("Model Ready!!")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.chat_history:
            self.chat_history[session_id] = ChatMessageHistory()
        return self.chat_history[session_id]
    
    def data_loader(self, path="./resume/resume.pdf"):
        loader = PyMuPDFLoader(path)
        data = loader.load_and_split()
        embeds = embeddings.OllamaEmbeddings(model='nomic-embed-text')
        vectorstore = FAISS.from_documents(data, embeds)
        retriever = vectorstore.as_retriever()
        return retriever

    def contextualize_history(self):
        ### Contextualize question ###
        
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        return contextualize_q_prompt

    def get_system_prompt(self, type):
        if type == "improve":
            system_prompt = (
                "You are a proficient hiring manager, with vast "
                "experience in reviewing resumes. You will be given "
                "a job summary and the user's resume. Your task is to "
                "analyze the resume in strict fashion and rate it out "
                " of 10 based on job summary. Further suggest improvements in bullet "
                "points to tailor the resume according to job summary. "
                "Do NOT reformat the resume, only give suggestions. "
                "\n\n"
                f"Job Summary: {self.job_summary}"
                "\n\n"
                "You will have additional context which can be "
                "utilized to answer user's queries. "
                "\n\n"
                "Context: {context}"
            )
        else:
            system_prompt = (
                "You are a proficient hiring manager, with vast "
                "experience in interviewing candidates. You will be given "
                "a job summary and the user's resume. Your task is to "
                "interview the user as in a professional setting. "
                "To interview ask a single question at a time. Make sure "
                "the questions are only related to the user's resume. "
                "After the interview is done, tell the user his chances "
                "of getting in with a feedback. "
                "\n\n"
                f"Job Summary: {self.job_summary}"
                "\n\n"
                "You will have additional context which can be "
                "utilized to answer user's queries. "
                "\n\n"
                "Context: {context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        return qa_prompt

    def agent_chat(self, usr_prompt):
        response = self.chat_model.invoke(
            {
                "input": usr_prompt,
                # "job_summary": "Accountant"
            },
                config={
                    "configurable": {"session_id": "acc_setup"}
                }
        )["answer"]
        return response

def main():
    chat_agent = ResumeAgent()
    print("You are now conversing with your assistant. Good Luck!")
    print("On what thoughts do you want to discuss today: ")
    while True:
        prompt = input("Enter you query|('/exit' to quit session): ")
        if prompt == "/exit":
            print("You will have a fortuitous encounter soon, God Speed!")
            break
        response = chat_agent.agent_chat(prompt)
        print(f"Assistant: {response}")
        print("-"*30)

if __name__ == "__main__":
    main()
