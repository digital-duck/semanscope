import streamlit as st
from typing import List, Dict

class ChatbotInterface:
    def __init__(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            
        if 'current_language' not in st.session_state:
            st.session_state.current_language = "English"
            
    def render(self):
        """Render the chatbot interface"""
        st.subheader("Language Learning Assistant")
        
        # Language selection
        st.session_state.current_language = st.selectbox(
            "Learning Language:",
            ["English", "Chinese", "Spanish", "French"]
        )
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Chat input
        if prompt := st.chat_input("What would you like to learn?"):
            self._add_message("user", prompt)
            # Here you would integrate with your chatbot backend
            response = self._get_bot_response(prompt)
            self._add_message("assistant", response)
            
    def _add_message(self, role: str, content: str):
        """Add a message to the chat history"""
        st.session_state.messages.append({"role": role, "content": content})
        
    def _get_bot_response(self, prompt: str) -> str:
        """Get response from bot (placeholder)"""
        return f"I'm here to help you learn {st.session_state.current_language}! [Replace with actual bot response]"
        
    def clear_history(self):
        """Clear chat history"""
        st.session_state.messages = []