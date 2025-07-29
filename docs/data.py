from dataclasses import dataclass
import warcio
import os
from typing import List, Iterable
from file_util import download_file
from markdownify import markdownify
from gzip import GzipFile
import re

@dataclass(frozen=True)
class Document:
    """A document with a URL and content."""
    url: str
    content: str


def get_common_crawl_urls(snapshot: str = "CC-MAIN-2024-18") -> List[str]:
    """Return the list of all the WARC files in the latest crawl."""
    download_file(f"https://data.commoncrawl.org/crawl-data/{snapshot}/warc.paths.gz", "var/warc.paths.gz")
    with GzipFile("var/warc.paths.gz") as f:
        urls = ["https://data.commoncrawl.org/" + line.decode("utf-8").rstrip() for line in f]
    return urls


def read_common_crawl(url: str, limit: int) -> Iterable[Document]:
    """Return the list of at most `limit` documents in the WARC file at `url`."""
    # Download the contents of the first URL
    path = os.path.join("var", os.path.basename(url))
    download_file(url, path)

    num_documents = 0
    for record in warcio.ArchiveIterator(open(path, "rb")):
        if num_documents >= limit:
            break
        if record.rec_type == "response":
            url = record.rec_headers.get_header("WARC-Target-URI")
            content_bytes = record.content_stream().read()
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            num_documents += 1
            yield Document(url, content)


def preprocess(documents: list[Document]) -> list[Document]:
    return [Document(url=document.url, content=markdownify(document.content)) for document in documents]


def write_documents(documents: Iterable[Document], path: str):
    with open(path, "w") as out:
        for i, document in enumerate(documents):
            print(f"--- PAGE {i}: url = {document.url}", file=out)
            print(document.content, file=out)
            print("", file=out)


def markdownify_documents(documents: Iterable[Document]) -> Iterable[Document]:
    for document in documents:
        yield Document(url=document.url, content=postprocess(markdownify(document.content)))


def postprocess(markdown: str) -> str:
    # Remove successive new lines
    return re.sub(r"\n\n+", "\n\n", markdown)