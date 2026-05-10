import streamlit as st

def get_auth_user():
    return st.session_state.get("auth_user")

def set_auth_user(user_dict: dict):
    st.session_state["auth_user"] = user_dict

def clear_auth_user():
    st.session_state.pop("auth_user", None)