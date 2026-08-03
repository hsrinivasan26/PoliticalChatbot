import os
import re

#imports
#PyPDFLoader only gets called once ever, after that the constitution is loaded instantly from FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import OPENAI_API_KEY

#All the pages in this particular PDF start with "C O N S T I T U T I O N O F T H E U N I T E D S T A T E S"
#We want to strip that out  so it doesn't get returned in results
#Section headers (which could be amendments or articles) start with "Amendment" or "Article",
#Then they proceed with (possibly a space) and a string of roman numerals; hence the [IVXLCDM] character class
PAGE_HEADER = "C O N S T I T U T I O N O F T H E U N I T E D S T A T E S"
HEADER_RE = re.compile(r"(Article\.\s+(?:[IVXLCDM]\s?)+\.|Amendment\s+(?:[IVXLCDM]\s?)+\.)")
SUBSECTION_RE = re.compile(r"SECTION\.?\s+(\d+)\s*\.?") #used for chunking
INDEX_PATH = "constitution_index"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

#gets rid of spaces/dots between the roman numerals in the header
def normalize_header(header):
    return re.sub(r"(?<=[IVXLCDM])\s+(?=[IVXLCDM])", "", header)

def clean_header(header):
    return header.replace(".", "").strip()


def load_text():
    loader = PyPDFLoader("constitution.pdf")
    pages = loader.load()
    text = "\n".join(p.page_content for p in pages)
    text = text.replace(PAGE_HEADER, " ")
    text = re.sub(r"-\s*\n\s*", "", text) #these two lines get rid of paragraph breaks and hyphenations in the text
    text = re.sub(r"\s+", " ", text).strip()
    return text

#RAG chunking. We need atomicity in the RAG breakdown so that vector similarity actually matches the sections with queries
#A lot of constitutional clauses talk about a lot of different things at the same time so they score poorly on relevance
def split_into_subsections(header, body):
    matches = list(SUBSECTION_RE.finditer(body)) #break down clause into subsections
    if not matches:
        return {clean_header(header): body} #if there are no subsections we are forced to return the clause

    subsections = {}
    intro = body[:matches[0].start()].strip()
    for i, m in enumerate(matches): #compile subsections into a dict
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[m.start():end].strip()
        if i == 0 and intro:
            section_text = f"{intro} {section_text}"
        subsections[f"{clean_header(header)}, Section {m.group(1)}"] = section_text
    return subsections


def extract_sections(text):
    parts = HEADER_RE.split(text) #split text by sections labeled "Article" or "Amendment"
    sections = {}
    preamble = parts[0].strip()
    if preamble: #one-time case for the preamble which doesn't have one of the predefined headers
        sections["Preamble"] = preamble
    for i in range(1, len(parts) - 1, 2):
        header = normalize_header(parts[i].strip())
        body = parts[i + 1].strip()
        sections.update(split_into_subsections(header, body))
    return sections


def build_index(): #store sections under their header name (section key)
    sections = extract_sections(load_text())
    texts = list(sections.values())
    metadatas = [{"reference": header} for header in sections.keys()]

    index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    index.save_local(INDEX_PATH)
    return index


#FAISS allows us to save embeddings to disk on first run. get_index determines if these
#embeddings already exist, in which case we load them instantly with load-index. Otherwise
#they don't exist yet, so we build them fresh with build-index (which also saves them for next time)
def load_index():
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def get_index():
    if os.path.exists(INDEX_PATH):
        return load_index()
    return build_index()


