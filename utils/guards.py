# utils/guards.py

import streamlit as st
from utils.auth import get_current_user

def login_required(func):
    """Decorator to ensure user is logged in before accessing a page."""
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user:
            st.warning("You must log in first.")
            return
        return func(*args, **kwargs)
    return wrapper
