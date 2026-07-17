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
    page_title="CineSage AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    background-size: 400% 400%;
    animation: bg 12s ease infinite;
    color: white;
}

@keyframes bg {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.main-title {
    font-size: 55px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 40px;
}

.glass {
    background: rgba(255, 255, 255, .07);
    border: 1px solid rgba(255, 255, 255, .15);
    backdrop-filter: blur(20px);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, .4);
}

.chip {
    display: inline-block;
    padding: 9px 15px;
    margin: 5px 6px 5px 0;
    border-radius: 999px;
    background: #2563eb;
    color: white;
    font-weight: 600;
}

div.stButton > button {
    width: 100%;
    height: 55px;
    border: none;
    border-radius: 15px;
    background: linear-gradient(90deg, #3b82f6, #9333ea);
    color: white;
    font-size: 18px;
    font-weight: 600;
    transition: .3s;
}

div.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px #3b82f6;
}

textarea {
    border-radius: 15px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<h1 class="main-title">🎬 CineSage AI</h1>
<p class="subtitle">Extract beautiful structured movie information using AI</p>
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
    "Enter Movie Paragraph",
    height=220,
    placeholder="""
Example:

Inception is a 2010 science fiction film directed by Christopher Nolan.
It stars Leonardo DiCaprio, Tom Hardy and Joseph Gordon-Levitt.
The movie received an IMDb rating of 8.8.
""",
)

extract = st.button("🚀 Extract Movie Information")

if not extract:
    st.info("Paste a movie paragraph and click Extract Movie Information.")
    st.stop()

if not paragraph.strip():
    st.warning("Please enter a movie paragraph.")
    st.stop()

model = ChatMistralAI(model="mistral-small-latest", temperature=0)
structured_model = model.with_structured_output(Movie)
final_prompt = prompt_template.invoke({"paragraph": paragraph})

try:
    with st.spinner("🍿 AI is extracting movie information..."):
        movie = structured_model.invoke(final_prompt)
except Exception as exc:
    st.error(f"Could not extract movie information: {exc}")
    st.stop()

genres = movie.genre or []
cast = movie.cast or []

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown(
        """
<div class="glass" style="text-align:center; min-height:260px;">
    <div style="font-size:100px; padding-top:15px;">🎬</div>
    <h4>Movie Poster</h4>
    <p style="color:#94a3b8;">Coming soon</p>
</div>
""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
<div class="glass">
    <h2 style="margin-bottom:5px;">{movie.title}</h2>
    ⭐ <b>{movie.rating or "NULL"}</b> / 10
    <br><br>
    📅 <b>Release:</b> {movie.release_year or "NULL"}<br><br>
    🎥 <b>Director:</b> {movie.director or "NULL"}
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🎭 Genres")

if genres:
    st.markdown(
        "".join(f'<span class="chip">{genre}</span>' for genre in genres),
        unsafe_allow_html=True,
    )
else:
    st.info("No genres found.")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="glass">
    <h3>📝 AI Summary</h3>
    <p>{movie.summary}</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📊 Movie Statistics")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("⭐ Rating", movie.rating if movie.rating is not None else "NULL")

with c2:
    st.metric("🎬 Genres", len(genres))

with c3:
    st.metric("👥 Cast Members", len(cast))

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🎭 Cast")

if cast:
    cols = st.columns(2)
    for index, actor in enumerate(cast):
        with cols[index % 2]:
            st.markdown(
                f"""
<div class="glass">
    👤 <b>{actor}</b>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
else:
    st.info("No cast members found.")

with st.expander("📦 View JSON Output"):
    st.json(movie.model_dump())

st.success("✅ Movie information extracted successfully.")
