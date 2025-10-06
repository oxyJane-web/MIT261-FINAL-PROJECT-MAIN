# app.py — MIT261 Student Analytics (Dashboard v2 fixed)
from __future__ import annotations
import streamlit as st

# ── Page config
st.set_page_config(
    page_title="MIT261 Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import auth and DB helpers
from utils.auth import (
    verify_login, ensure_default_admin,
    current_user, get_current_user, set_current_user, sign_out,
)
from db import col

# Cross-version rerun helper
def _rerun():
    r = getattr(st, "rerun", None)
    if callable(r):
        r()
    else:
        e = getattr(st, "experimental_rerun", None)
        if callable(e):
            e()

# ---------- CSS ----------
st.markdown(
    """
    <style>
      .main { padding-top: 1rem; }
      .login-card {
        background: #1a2234; border:1px solid rgba(255,255,255,.1);
        border-radius:18px; padding:1.5rem; box-shadow:0 4px 16px rgba(0,0,0,.4);
      }
      .dashboard-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit,minmax(260px,1fr));
        gap: 1.25rem;
        margin-top:1.5rem;
      }
      .card {
        background:#202a40; border-radius:16px; padding:1.2rem;
        box-shadow:0 2px 10px rgba(0,0,0,.35);
        color:#fff;
      }
      .card h3 { margin:0; font-size:1.25rem; }
      .card p { margin:.25rem 0 0; color:#cfd6e6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar Footer ----------
def render_sidebar_footer(user: dict) -> None:
    if not user:
        return
    with st.sidebar:
        st.divider()
        st.markdown(f"**👤 {user.get('email','')}**")
        st.caption(f"Role: {user.get('role','').title()}")
        if st.button("🚪 Log out", use_container_width=True, key="logout_btn"):
            sign_out()
            _rerun()

# ---------- Login ----------
def render_login():
    st.title("📊 MIT261 Dashboard Login")
    st.caption("Sign in to continue")

    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="name@su.edu", key="login_email")
        pw = st.text_input("Password", type="password", placeholder="••••••••", key="login_pw")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login", type="primary", use_container_width=True):
                u = verify_login(email, pw)
                if not u:
                    st.error("❌ Invalid credentials.")
                else:
                    set_current_user(u)
                    _rerun()
        with c2:
            if st.button("Create Admin", use_container_width=True):
                ensure_default_admin("admin@su.edu", password="Admin1234", reset_password=True)
                st.success("✅ Default Admin ready: **admin@su.edu / Admin1234**")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Dashboard ----------
def render_dashboard(user: dict):
    st.title("🎓 MIT261 Student Analytics")
    st.caption(f"Welcome back, **{user.get('email','')}** — role: **{user.get('role','').title()}**")

    # Display dashboard cards
    st.markdown('<div class="dashboard-cards">', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>📈 Enrollment</h3><p>Monitor student registration status</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>👨‍🏫 Faculty Access</h3><p>Faculty dashboards with course stats</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>🎓 Student Records</h3><p>Academic performance & profile</p></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.info("Use the **sidebar** to navigate dashboards. Access depends on your role.")

# ---------- Entry ----------
def main():
    user = get_current_user() or current_user()

    # ✅ Always show something
    if not user:
        render_login()
    else:
        render_sidebar_footer(user)
        render_dashboard(user)

if __name__ == "__main__":
    main()
