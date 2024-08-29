import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.base import Runnable
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import HumanMessage, AIMessage

from modules import LLMApp, EmbeddingModel, DatabaseModel

from dotenv import load_dotenv

load_dotenv()

class ChatAgent:
    def __init__(self):
        """
        Initialize the ChatAgent.
        """
        self.history = StreamlitChatMessageHistory(key="chat_history")
        self.chat_history = []

    def display_messages(self):
        """
        Display messages in the chat interface.
        If no messages are present, adds a default AI message.
        """
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            st.session_state.chat_history.append(AIMessage(content="How can I help you?"))
            st.chat_message("ai").write("How can I help you?")


        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)
            else:
                with st.chat_message("AI"):
                    st.markdown(message.content)


    def start_conversation(self, app, retriever):
        """
        Start a conversation in the chat interface.
        Displays messages, prompts user for input, and handles AI response.
        """
        self.display_messages()
        user_question = st.chat_input(placeholder="Ask me anything!")
        if user_question:
            st.chat_message("human").write(user_question)
            #config = {"configurable": {"session_id": "any"}}
            #response = self.chain.invoke({"question": user_question}, config)
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            self.chat_history.append(HumanMessage(content=user_question))
            #with st.chat_message("ai"):
            #    response = st.write_stream(app.llm_processor(retriever, self.chat_history, user_question))
            response = app.llm_processor(retriever, self.chat_history, user_question)
            st.chat_message("ai").write(response)
            
            self.chat_history.append(AIMessage(content=response))
            st.session_state.chat_history.append(AIMessage(content=response))


texts = """
Waste management or waste disposal includes the processes and actions required to manage waste from its inception to its final disposal.[1] This includes the collection, transport, treatment, and disposal of waste, together with monitoring and regulation of the waste management process and waste-related laws, technologies, and economic mechanisms.

Waste can either be solid, liquid, or gases and each type has different methods of disposal and management. Waste management deals with all types of waste, including industrial, biological, household, municipal, organic, biomedical, radioactive wastes. In some cases, waste can pose a threat to human health.[2] Health issues are associated with the entire process of waste management. Health issues can also arise indirectly or directly: directly through the handling of solid waste, and indirectly through the consumption of water, soil, and food.[2] Waste is produced by human activity, for example, the extraction and processing of raw materials.[3] Waste management is intended to reduce the adverse effects of waste on human health, the environment, planetary resources, and aesthetics.

The aim of waste management is to reduce the dangerous effects of such waste on the environment and human health. A big part of waste management deals with municipal solid waste, which is created by industrial, commercial, and household activity

Waste management practices are not the same across countries (developed and developing nations); regions (urban and rural areas), and residential and industrial sectors can all take different approaches.

Proper management of waste is important for building sustainable and liveable cities, but it remains a challenge for many developing countries and cities. A report found that effective waste management is relatively expensive, usually comprising 20%–50% of municipal budgets. Operating this essential municipal service requires integrated systems that are efficient, sustainable, and socially supported. A large portion of waste management practices deal with municipal solid waste (MSW) which is the bulk of the waste that is created by household, industrial, and commercial activity. According to the Intergovernmental Panel on Climate Change (IPCC), municipal solid waste is expected to reach approximately 3.4 Gt by 2050; however, policies and lawmaking can reduce the amount of waste produced in different areas and cities of the world.[8] Measures of waste management include measures for integrated techno-economic mechanisms[9] of a circular economy, effective disposal facilities, export and import control and optimal sustainable design of products that are produced.

In the first systematic review of the scientific evidence around global waste, its management, and its impact on human health and life, authors concluded that about a fourth of all the municipal solid terrestrial waste is not collected and an additional fourth is mismanaged after collection, often being burned in open and uncontrolled fires – or close to one billion tons per year when combined. They also found that broad priority areas each lack a "high-quality research base", partly due to the absence of "substantial research funding", which motivated scientists often require.[12][13] Electronic waste (ewaste) includes discarded computer monitors, motherboards, mobile phones and chargers, compact discs (CDs), headphones, television sets, air conditioners and refrigerators. According to the Global E-waste Monitor 2017, India generates ~ 2 million tonnes (Mte) of e-waste annually and ranks fifth among the e-waste producing countries, after the United States, the People's Republic of China, Japan and Germany.

Effective 'Waste Management' involves the practice of '7R' - 'R'efuse, 'R'educe', 'R'euse, 'R'epair, 'R'epurpose, 'R'ecycle and 'R'ecover. Amongst these '7R's, the first two ('Refuse' and 'Reduce') relate to the non-creation of waste - by refusing to buy non-essential products and by reducing consumption. The next two ('Reuse' and 'Repair') refer to increasing the usage of the existing product, with or without the substitution of certain parts of the product. 'Repurpose' and 'Recycle' involve maximum usage of the materials used in the product, and 'Recover' is the least preferred and least efficient waste management practice involving the recovery of embedded energy in the waste material. For example, burning the waste to produce heat (and electricity from heat). Certain non-biodegradable products are also dumped away as 'Disposal', and this is not a "waste-'management'" practice.
"""

def main():
    st.set_page_config(
        page_title="AI Chatbot"
    )

    app = LLMApp(texts=texts)

    encoder = EmbeddingModel()

    collection_name = "data"
    db_model = DatabaseModel(embedding_model=encoder.embedding_model, collection_name=collection_name)

    db_model.add_to_db(app)
    retriever = db_model.get_qdrant_as_retriever()
    
    # This is the chatbot component
    chat_agent = ChatAgent()
    chat_agent.start_conversation(app, retriever)

if __name__ == "__main__":
    main()
