import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

try:
    from schemas import Movie
except ImportError:
    from cineSage.schemas import Movie


load_dotenv()

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #d1d1d1;
    margin-bottom: 30px;
}

.movie-card {
    background: white;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, .25);
}

.card-title {
    font-size: 35px;
    font-weight: bold;
    color: #111;
}

.small {
    font-size: 18px;
    margin-top: 10px;
    color: #111;
}

.summary {
    background: #f4f4f4;
    padding: 20px;
    border-radius: 15px;
    font-size: 17px;
    line-height: 1.8;
    color: #111;
}

.stButton > button {
    width: 100%;
    height: 60px;
    font-size: 22px;
    border-radius: 15px;
    background: #00ADB5;
    color: white;
    font-weight: bold;
}

.stButton > button:hover {
    background: #00cfd8;
}

textarea {
    font-size: 18px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-title">🎬 Movie Information Extractor</div>
<div class="subtitle">Powered by LangChain • Mistral AI • Pydantic</div>
""",
    unsafe_allow_html=True,
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You extract structured movie information from user text.
Use only the supplied paragraph.
If a value is missing, use null for optional scalar fields and an empty list for list fields.
Do not invent unknown facts.
""",
        ),
        ("human", "Movie paragraph:\n\n{paragraph}"),
    ]
)

paragraph = st.text_area(
    "Movie Paragraph",
    height=280,
    placeholder="Paste movie description here...",
)

extract = st.button("✨ Extract Information", use_container_width=True)

if extract:
    if not paragraph.strip():
        st.warning("Please enter a movie paragraph.")
        st.stop()

    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    structured_model = model.with_structured_output(Movie)
    final_prompt = prompt_template.invoke({"paragraph": paragraph})

    try:
        with st.spinner("Analyzing movie..."):
            movie = structured_model.invoke(final_prompt)
    except Exception as exc:
        st.error(f"Could not extract movie information: {exc}")
        st.stop()

    genres = ", ".join(movie.genre) if movie.genre else "NULL"
    cast = ", ".join(movie.cast) if movie.cast else "NULL"

    st.markdown(
        f"""
<div class="movie-card">
    <div class="card-title">🎬 {movie.title}</div>
    <div class="small">📅 <b>Release Year</b>: {movie.release_year or "NULL"}</div>
    <div class="small">🎬 <b>Director</b>: {movie.director or "NULL"}</div>
    <div class="small">⭐ <b>Rating</b>: {movie.rating or "NULL"}</div>
    <div class="small">🎭 <b>Genre</b>: {genres}</div>
    <div class="small">👥 <b>Cast</b>: {cast}</div>
    <br>
    <div class="summary"><b>Summary</b><br><br>{movie.summary}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("📄 Raw JSON"):
        st.json(movie.model_dump())
