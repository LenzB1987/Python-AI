import os
from myopenaikey import apikey
import streamlit as sets
from langchain.llms import OpenAI
from langchain.prompts  import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
os.environ['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'] =apikey
#app framework
sets.title ('Lenz Chat Gpt similar app')
prompt=sets.text_input('ask me anything')
#prompt templates
titles=PromptTemplate(
    input_variables=['topics or essays about'],
    template='write for me anything {topic}'
)
essay_template=PromptTemplate(
    input_variables=['titles','from wiki'],
    template="explain for me anything based on this title"
)
#memory
title_memory=ConversationBufferMemory(input_key='topics or essays about', memory_key='chat_history')
script_memory=ConversationBufferMemory(input_key='titles', memory_key='chat_history')
#llms
learningmodel=OpenAI(temperature=0.9)
title_chain=LLMChain(learningmodel=learningmodel,prompt=titles, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(learningmodel=learningmodel, prompt=essay_template, verbose=True, output_key='essay', memory=script_memory)
wiki=WikipediaAPIWrapper()
if prompt:
    title=title_chain.run(prompt)
    wiki=wiki.run(prompt)
    essay=script_chain.run(titles=title,wiki=wiki)
    sets.write(title)
    sets.write(essay)
    with sets.expander('Topic/ Title history'):
        sets.info(title_memory.buffer)
    with sets.expander('essay history'):
        sets.info (script_memory.buffer)
    with sets.expander('wiki info'):
        sets.info(wiki)