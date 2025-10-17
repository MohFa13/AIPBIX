# Streamlit Cloud Hotfix Overlay (Applied)

This package includes a **safe Streamlit entrypoint** and **explicit server config** to fix the
`healthz → connection refused` error on Streamlit Community Cloud.

## What was added
- `streamlit_app.py` at the repository root — a minimal, safe entrypoint that:
  - sets page config,
  - attempts to **delegate** to `app.main()` if an `app.py` exists,
  - otherwise shows a **sanity page** so health checks pass.
- `.streamlit/config.toml` — explicit server settings compatible with Streamlit Cloud.

## What you need to do on Streamlit Cloud
1. In the app settings, set **Main file path** to:
   ```
   streamlit_app.py
   ```
2. Ensure your actual app exposes a **`main()` function** (commonly in `app.py`). Example:
   ```python
   # app.py
   import streamlit as st

   def main():
       st.title("AIPBIX — Real App")
       st.write("Your real UI goes here.")

   if __name__ == "__main__":
       main()
   ```
3. Avoid running heavy work at import time. Move heavy code inside functions and cache with `st.cache_data` / `st.cache_resource`.

## Notes
- If your main module is not `app.py`, update the list inside `streamlit_app.py` (variable `mod_name` loop).
- Keep your existing dependency file (e.g., `uv.lock`) consistent with your chosen manager to reduce startup surprises.
