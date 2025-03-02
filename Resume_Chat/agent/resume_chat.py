from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

class ResumeAgent:
    def __init__(self, job_summary=""):
        self.llm = ChatOllama(model="qwen2.5-coder:7b")
        print("Model Loaded")

        self.job_summary = job_summary
        self.vector_store = self.data_loader()
        self.chat_model = self.build_graph()
        print("Model Ready!!")
    
    def data_loader(self, path="./Resume_Chat/resume/sample_resume.pdf"):
        # TODO(REZ3LIET): ChunkRAG
        loader = PyMuPDFLoader(path)
        data = loader.load_and_split()
        embeds = OllamaEmbeddings(model="qwen2.5-coder:7b")
        vectorstore = FAISS.from_documents(data, embeds)
        return vectorstore
    
    @tool(response_format="content_and_artifact")
    def retrieve(self, query: str):
        """Retrieve information related to a string query."""
        retrieved_docs = self.vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""
        # Step 1: Generate an AIMessage that may include a tool-call to be sent.
        llm_with_tools = self.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    def generate(self, state: MessagesState):
        """Generate answer."""
        # Step 3: Generate a response using the retrieved content.
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def build_graph(self):
        # Build graph
        tools = ToolNode([self.retrieve])
        graph_builder = StateGraph(MessagesState)

        graph_builder.add_node(self.query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(self.generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        return graph

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
        config = {"configurable": {"thread_id": "abc123"}}

        for step in self.chat_model.stream(
            {"messages": [{"role": "user", "content": usr_prompt}]},
            stream_mode="values",
            config=config,
        ):
            step["messages"][-1].pretty_print()
        return True

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
