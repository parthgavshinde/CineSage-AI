from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()
model = ChatMistralAI(model="mistral-small-latest")





prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a professional Movie Information Extraction Assistant.

Rules:
- Return ONLY Markdown.
- Follow the exact format below.
- Every field MUST be on a separate line.
- Leave one blank line between every section.
- Never merge two fields into one line.
- If information is missing, write NULL.
- Do NOT add explanations.

Output Format:

## Movie Title
<Movie Title>

## Release Year
<Release Year>

## Genre
<Genre>

## Director
<Director>

## Main Cast
<Person 1>, <Person 2>, <Person 3>

## Setting/Location
<Location>

## Plot
<Brief Plot>

## Themes
<Themes>

## Ratings
<Ratings>

## Notable Features
- Feature 1
- Feature 2

## Short Summary
<2-3 line summary>
"""
    ),
    ("human", "Extract information from:\n\n{paragraph}")
])

print("Enter movie paragraph (Press Enter twice to finish):")

lines = []

while True:
    line = input()
    if line == "":
        break
    lines.append(line)

paragraph = "\n".join(lines)

final_prompt = prompt_template.invoke({
    "paragraph": paragraph
})

response = model.invoke(final_prompt)

print("\n")
print(response.content)
