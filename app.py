import streamlit as st
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS

# 1. Page Configuration
st.set_page_config(page_title="🌐 Live-Search AI Assistant", layout="centered")
st.title("🤖 Live-Search AI Assistant")
st.write("Welcome! This AI utilizes live DuckDuckGo web scraping to access real-time 2026 data—completely free.")

# Clear Chat History Utility
with st.sidebar:
    if st.button("Core 🧹 Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am connected to the live web. What would you like me to look up today?"}
        ]
        st.rerun()

# 2. Pull the token from Streamlit Secrets
hf_token = st.secrets.get("HF_TOKEN", None)

if not hf_token:
    with st.sidebar:
        st.subheader("⚙️ Local Development Setup")
        hf_token = st.text_input("Enter Hugging Face Token to test locally:", type="password")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am connected to the live web. What would you like me to look up today?"}
    ]

# Display older messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- 🛰️ FREE LIVE SEARCH ENGINE HELPER ---
def fetch_live_web_context(query: str, max_results: int = 3) -> str:
    """Scrapes DuckDuckGo for the query and builds an LLM context block."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No real-time web results found."
            
            context_string = "ADDITIONAL REAL-TIME WEB CONTEXT:\n"
            for i, res in enumerate(results, 1):
                context_string += f"[{i}] Source: {res.get('href')}\nTitle: {res.get('title')}\nSnippet: {res.get('body')}\n\n"
            return context_string
    except Exception as search_error:
        return f"Could not pull live web results due to an error: {search_error}"

# 4. User Interaction Loop
if hf_token:
    if user_query := st.chat_input("Ask me about recent events, code, or general topics..."):
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Display assistant streaming response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                client = InferenceClient(api_key=hf_token)
                
                # 📢 STEP 1: Search the web live using the user's latest query
                with st.spinner("🔍 Scanning live web search indexes..."):
                    web_context = fetch_live_web_context(user_query)
                
                # 📢 STEP 2: Inject the live web data dynamically into the System Prompt
                system_instruction = (
                    "You are a cutting-edge real-time AI assistant running natively in 2026. "
                    "You are grounded with live web search snippets. Analyze the provided context carefully "
                    "to formulate an accurate, direct response. Cite source numbers like [1] or [2] if helpful. "
                    f"\n\n{web_context}"
                )
                
                # Assemble system instructions + standard chat memory array
                formatted_messages = [{"role": "system", "content": system_instruction}]
                for m in st.session_state.messages:
                    formatted_messages.append({"role": m["role"], "content": m["content"]})
                
                # Stream response from Llama-3.3-70B
                for chunk in client.chat_completion(
                    model="meta-llama/Llama-3.3-70B-Instruct",
                    messages=formatted_messages,
                    max_tokens=1200,
                    stream=True,
                ):
                    if chunk.choices:
                        token = chunk.choices[0].delta.content
                        if token:
                            full_response += token
                            response_placeholder.markdown(full_response + "▌")
                
                # Clean render output
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please add a Hugging Face token to begin.")
