import streamlit as st
import requests
import pandas as pd

def main():
    st.title("Machine Learning Application with FastAPI Backend")
    
    st.sidebar.title("Upload Data")
    file_upload = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    
    if file_upload:
        data = pd.read_csv(file_upload)
        st.write("## Data Preview")
        st.write(data.head())

        if st.sidebar.button("Run Analysis"):
            response = requests.post("http://localhost:8000/uploadfile/", files={"file": file_upload.getvalue()})
            if response.status_code == 200:
                st.write("## Analysis Results")
                st.write(response.json())
            else:
                st.write("Error in backend processing")

if __name__ == '__main__':
    main()
