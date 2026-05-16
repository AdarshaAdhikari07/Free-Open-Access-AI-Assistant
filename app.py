import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain import hub

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
if st.button("Initialize Agent") or "agent_executor" in st.session_state:
    if not openai_key or not serp_key:
        st.warning("Please enter both API keys in the sidebar to proceed.")
    else:
        # Save to session state so it persists across user inputs
        if "agent_executor" not in st.session_state:
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
            
            # Pull the standard ReAct prompt template from LangChain Hub
            prompt = hub.pull("hwchase17/react")
            
            # Construct the modern ReAct agent
            agent = create_react_agent(llm, tools, prompt)
            
            # Create the executor that handles the agent's loop
            st.session_state.agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True,
                handle_parsing_errors=True # Gracefully handles formatting errors
            )
            st.success("Agent is ready for action!")

# 4. User Interaction
if "agent_executor" in st.session_state:
    user_query = st.text_input("What would you like me to research today?")
    
    if user_query:
        with st.spinner("🧠 Agent is thinking and searching..."):
            try:
                # Run the agent using invoke (the modern replacement for .run())
                response = st.session_state.agent_executor.invoke({"input": user_query})
                st.markdown("### 📋 Result:")
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred: {e}")
