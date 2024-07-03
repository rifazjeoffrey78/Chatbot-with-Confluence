import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from help_desk import HelpDesk
from emailSend import Email


if "helpdesk_instance" not in st.session_state:
    #email = Email()
    #model = HelpDesk(new_db=False)
    st.session_state.helpdesk_instance = HelpDesk(new_db=False)
    st.session_state.email_instance = Email()
    print("******* Instance initiated *******")

# app config
st.set_page_config(page_title="Chat with confluence", page_icon="🤖")
st.title("Welcome to PST Helpdesk")

# sidebar
with st.sidebar:
    st.header("Actions")
    website_url = "Test"#st.text_input("Space Key")

    if st.button("Summarize"):
        history = st.session_state.chat_history
        model = st.session_state.helpdesk_instance
        print("******")
        print(history)
        print("******")
        summary = model.get_conversation_summary(history)
        st.write(summary)
        st.session_state.convo_summary = summary

    if st.button("Create Ticket"):
        convoSummary = st.session_state.convo_summary
        email = st.session_state.email_instance
        print("******")
        print(convoSummary)
        print("******")
        email.sendEmail(convoSummary)


if website_url is None or website_url == "":
    st.info("Please enter the Confluence space key")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    # if "vector_store" not in st.session_state:
    #     st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        model = st.session_state.helpdesk_instance
        model.chatHistory = st.session_state.chat_history
        response, sources = model.retrieval_qa_inference(user_query)
        #print(sources)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

# Demo
#if __name__ == '__main__':
 #   pass
    # from help_desk import HelpDesk
    #
    # model = HelpDesk(new_db=True)
    #
    # print(model.db._collection.count())
    #
    # prompt = 'who is PST HR?'
    # result, sources = model.retrieval_qa_inference(prompt)
    # print(result, sources)
