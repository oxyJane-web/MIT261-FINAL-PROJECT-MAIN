import streamlit as st

def verify_login(username: str, password: str) -> bool:
    if username == "admin" and password == "admin123":
        set_current_user({"username": username, "role": "admin"})
        return True
    return False

def ensure_default_admin():
    if "users" not in st.session_state:
        st.session_state["users"] = [{"username": "admin", "password": "admin123"}]

def set_current_user(user: dict):
    st.session_state["current_user"] = user

def get_current_user():
    return st.session_state.get("current_user")

# ✅ Instead of property, just a normal function/alias
def current_user():
    return get_current_user()

def sign_out():
    if "current_user" in st.session_state:
        del st.session_state["current_user"]
