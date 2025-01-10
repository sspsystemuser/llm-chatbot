
import pandas as pd
from langchain.docstore.document import Document

df = pd.read_csv("movies.csv")
documents = [Document(page_content=row.to_string(), metadata={}) for _, row in df.iterrows()]

# Use documents with LangChain
print(documents)
