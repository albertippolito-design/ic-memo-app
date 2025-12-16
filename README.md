# Test Web Streamlit App

This small app collects text from the user and appends it to `user.txt`.

Run locally:

```powershell
python -m pip install -r requirements.txt
streamlit run test_web.py
```

Open the URL shown by Streamlit in your browser. Enter text in the box and click `Save` to append it to `user.txt`.

Notes:
- `user.txt` will be created in the same folder as `test_web.py` when you save.
- If VS Code asks to `Select Interpreter`, pick the Python executable you installed (e.g. `C:\Users\aiferreira\AppData\Local\Programs\Python\Python314\python.exe`).
