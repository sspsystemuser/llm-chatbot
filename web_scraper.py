import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
import langchain
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai
langchain.debug = True

llm = OpenAI()

# first step in chain

first_prompt = PromptTemplate(
input_variables=["project"],
template="Find the information about {project}. return list of grants provided with eligibility criteria")
chain_one = LLMChain(llm = llm, prompt = first_prompt)

# second step in chain
second_prompt = PromptTemplate(
input_variables=["data"],
template="Is there any grant available from sweden from given data {data}. provide result with website url",)
chain_two = LLMChain(llm=llm, prompt=second_prompt)
# Combine the first and the second chain

overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)

main_placeholder = st.empty()
st.title("Result:")
query = main_placeholder.text_input("Query: ")
if query:
    final_answer = overall_chain.run(query)
    st.write("Final answer, after merging both chains")
    st.write(final_answer)