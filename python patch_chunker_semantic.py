from pathlib import Path
import subprocess


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


BASE = Path.cwd()

print("Applying semantic chunking upgrade...")

# improved chunker

write(
    BASE / "rag/chunker.py",
"""
import re


def split_sentences(text):

    # split on sentence boundaries

    sentences = re.split(r'(?<=[.!?])\\s+', text)

    return [

        s.strip()

        for s in sentences

        if len(s.strip()) > 10

    ]


def chunk_document(text, max_sentences=3):

    sentences = split_sentences(text)

    chunks = []

    for i in range(0, len(sentences), max_sentences):

        chunk = " ".join(sentences[i:i+max_sentences])

        chunks.append(chunk)

    return chunks
"""
)

print("Chunker upgraded.")


# commit changes

run("git add rag/chunker.py")

try:

    run('git commit -m "upgrade chunking to semantic sentence-based strategy"')

except:

    print("No changes detected")


run("git push")

print("Patch complete and pushed to GitHub.")