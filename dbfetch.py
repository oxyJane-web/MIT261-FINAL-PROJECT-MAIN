import streamlit as st
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient("mongodb+srv://janeoroalferes2727:pass1234@cluster0.fxg6hbl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["mit261"]
students_collection = db["curriculum,students,teachers, enrollments"]

# Fetch all students
students = list(students_collection.find({}, {"_id": 0}))  # hide MongoDB _id

# Convert to DataFrame for nicer display
if students:
    df = pd.DataFrame(students)
    st.dataframe(df)
else:
    st.info("No student data found in MongoDB.")
