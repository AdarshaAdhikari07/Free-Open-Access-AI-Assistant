import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page Configuration
st.set_page_config(page_title="🤖 Free Public AI", layout="centered")
st.title("🤖 Free Open-Access AI Assistant")
st.write("Welcome! This AI is completely free to use for everyone—no account or subscription required.")

# 2. Securely pull the token from Streamlit Secrets (We will set this up in Step 3!)
# If not deployed yet, it falls back to checking a local text field for testing.
hf_token = st.secrets.get("HF_TOKEN", None)

if not hf_token:
    with st.sidebar:
        st.subheader("⚙️ Local Development Setup")
        hf_token = st.text_input("Enter Hugging Face Token to test locally:", type="password")
        st.caption("Note: This sidebar will disappear once you add the token to Streamlit Secrets!")

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am a free, open-source AI assistant. How can I help you today?"}
    ]

# Display older messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 4. User Interaction Loop
if hf_token:
    if user_query := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Display assistant streaming response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Initialize free serverless client running a top-tier open-source model
                client = InferenceClient(provider="hf-inference", api_key=hf_token)
                
                # Format messages correctly for the model
                formatted_messages = [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                ]
                
                # Stream the text chunks live as they generate
                for chunk in client.chat_completion(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages=formatted_messages,
                    max_tokens=1000,
                    stream=True,
                ):
                    token = chunk.choices[0].delta.content
                    if token:
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                
                # Render final clean response
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please add a Hugging Face token to begin.")
