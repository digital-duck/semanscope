import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional

class NoteEditor:
    def __init__(self):
        if 'notes' not in st.session_state:
            st.session_state.notes = []
            
    def render(self):
        """Render the note editor interface"""
        st.subheader("Language Learning Notes")
        
        # Add new note
        with st.expander("Add New Note", expanded=True):
            note_text = st.text_area("Note content:")
            tags = st.text_input("Tags (comma separated):")
            
            if st.button("Save Note"):
                if note_text:
                    self._save_note(note_text, tags)
                    st.success("Note saved!")
                    
        # Display existing notes
        if st.session_state.notes:
            st.subheader("Your Notes")
            for note in st.session_state.notes:
                with st.expander(f"{note['date']} - {note['tags']}", expanded=False):
                    st.write(note['content'])
                    if st.button("Delete", key=f"del_{note['id']}"):
                        self._delete_note(note['id'])
                        st.experimental_rerun()
                        
    def _save_note(self, content: str, tags: str):
        """Save a new note"""
        note = {
            'id': len(st.session_state.notes),
            'content': content,
            'tags': tags,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.notes.append(note)
        
    def _delete_note(self, note_id: int):
        """Delete a note by ID"""
        st.session_state.notes = [
            note for note in st.session_state.notes 
            if note['id'] != note_id
        ]