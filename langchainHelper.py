from secret_key import openapi_key
import os
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
os.environ['OPENAI_API_KEY'] = openapi_key
llm = OpenAI(temperature=0.6)


def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest a fency name for this")

    name_chain = LLMChain(llm=llm,prompt=prompt_template_name,output_key="restaurant_name")
    prompt_template_items = PromptTemplate(
    input_variables=["restaurant_name"],
    template="Suggest some menu items for {restaurant_name} Return it as a comma separated list")
    food_item_chain = LLMChain(llm=llm,prompt=prompt_template_items,output_key="menu_items")
    chain = SequentialChain(
    chains=[name_chain,food_item_chain],
    input_variables=["cuisine"],
    output_variables=["restaurant_name","menu_items"])
    response = chain({'cuisine':cuisine})

    return response





