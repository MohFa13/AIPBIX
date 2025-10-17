# streamlit_app.py — safe entrypoint for Streamlit Cloud
# Set Streamlit Cloud "Main file path" to streamlit_app.py
# Delegates to app.main() if available; otherwise shows a sanity page.
import traceback
import importlib
import streamlit as st

st.set_page_config(page_title="AIPBIX", layout="wide")

def _run_delegate():
    # Try common module candidates. Adjust if your main module has a different name.
    for mod_name in ("app", "src.app", "aipbix.app"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "main") and callable(mod.main):
                mod.main()
                return True
        except Exception:
            # Keep trying the next candidate
            continue
    return False

def _sanity():
    st.title("AIPBIX — Streamlit Sanity Check")
    st.success("Streamlit server is healthy ✅")
    st.write(
        "This page is shown because no `main()` function was found in your app module.\n"
        "If your existing code lives in `app.py`, define a `main()` function and this entrypoint will route to it."
    )
    with st.expander("How to enable your real app"):
        st.code(
            """
# app.py
import streamlit as st

def main():
    st.title("AIPBIX — Real App")
    st.write("Your real UI goes here. Move heavy work under functions and cache as needed.")

if __name__ == "__main__":
    main()
            """, language="python"
        )

if not _run_delegate():
    _sanity()
