import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType

# 1. Streamlit Page Configuration
st.set_page_config(page_title="🤖 My First AI Agent", layout="centered")
st.title("🤖 Web-Searching AI Research Agent")
st.write("Ask me anything! If I don't know the answer, I'll browse the web to find out.")

# 2. Sidebar for API Keys
with st.sidebar:
    st.header("🔑 API Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password")
    serp_key = st.text_input("SerpAPI Key", type="password")
    st.markdown("[Get OpenAI Key](https://platform.openai.com/) | [Get SerpAPI Key](https://serpapi.com/)")

# 3. Main Agent Logic
if st.button("Initialize Agent") or "agent" in st.session_state:
    if not openai_key or not serp_key:
        st.warning("Please enter both API keys in the sidebar to proceed.")
    else:
        # Save keys to session state so they persist
        if "agent" not in st.session_state:
            # Initialize the search tool
            search = SerpAPIWrapper(serpapi_api_key=serp_key)
            
            tools = [
                Tool(
                    name="Search",
                    func=search.run,
                    description="Useful for when you need to answer questions about current events or real-time data."
                )
            ]
            
            # Initialize the LLM (Brain)
            llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=openai_key)
            
            # Initialize the Agent
            st.session_state.agent = initialize_agent(
                tools, 
                llm, 
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                verbose=True
            )
            st.success("Agent is ready to action!")

# 4. User Interaction
if "agent" in st.session_state:
    user_query = st.text_input("What would you like me to research today?")
    
    if user_query:
        with st.spinner("🧠 Agent is thinking and searching..."):
            try:
                # Run the agent
                response = st.session_state.agent.run(user_query)
                st.markdown("### 📋 Result:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
