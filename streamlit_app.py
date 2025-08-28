import streamlit as st
import pandas as pd
import random

st.title("AI Review Classification Demo")

st.write("Upload a CSV file containing reviews with the following headers:")
st.write("`business_name, author_name, text, photo, rating, rating_category`")

# upload
uploaded_file = st.file_uploader("Upload your reviews CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df)

    # mock ai op
    def mock_ai_prediction(row):
        valid = random.choice([True, False])
        reason = "Contains policy violation" if not valid else "Looks fine"
        return pd.Series([valid, reason], index=["is_valid", "reason"])

    df[["is_valid", "reason"]] = df.apply(mock_ai_prediction, axis=1)

    st.subheader("AI Classification Results")
    st.dataframe(df)

    # download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "classified_reviews.csv", "text/csv")
