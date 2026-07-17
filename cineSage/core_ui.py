import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatMistralAI(model="mistral-small-latest")

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a professional Movie Information Extraction Assistant.

Your task:
Extract useful structured information from the given movie paragraph.

Rules:
- Return ONLY Markdown.
- Follow the exact format.
- Every heading must be on a separate line.
- Leave one blank line between every section.
- Never merge two fields into one line.
- Do NOT add explanations.
- Do NOT add extra commentary.
- If information is missing, write NULL.
- Do NOT guess unknown facts.

Example Output:

## Movie Title
The Dark Knight

## Release Year
2008

## Genre
Superhero Crime Thriller

## Director
Christopher Nolan

## Main Cast
Christian Bale, Heath Ledger, Aaron Eckhart, Gary Oldman

## Setting/Location
Gotham City

## Plot
Batman battles the Joker, who spreads fear and chaos throughout Gotham City.

## Themes
Chaos vs Order, Justice, Sacrifice

## Ratings
IMDb: 9.0

## Notable Features
- Heath Ledger's Oscar-winning performance
- Grounded realism

## Short Summary
Batman fights the Joker to save Gotham while struggling with difficult moral choices.
"""
    ),
    (
        "human",
        """
Extract information from this paragraph:

{paragraph}
"""
    )
])

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬"
)

st.title("🎬 Movie Information Extractor")

paragraph = st.text_area(
    "Enter the paragraph",
    height=250,
    placeholder="Paste your movie paragraph here..."
)

if st.button("🎬 Extract Information"):

    if not paragraph.strip():
        st.warning("Please enter a movie paragraph.")
        st.stop()

    final_prompt = prompt_template.invoke(
        {"paragraph": paragraph}
    )

    with st.spinner("Extracting movie information..."):
        response = model.invoke(final_prompt)

    st.subheader("📋 Extracted Information")
    st.markdown(response.content)
