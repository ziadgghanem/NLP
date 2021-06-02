
import streamlit as st

from transformers import BertForQuestionAnswering, AutoTokenizer
modelname = 'deepset/bert-base-cased-squad2'

model = BertForQuestionAnswering.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)

from transformers import pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)



st.title("this my streamlit q&a page title")

st.write("""
# this is my heading
and this is just normal text
""")

user_context = st.text_area("give the model some context here", height = 100)
user_question = st.text_input("ask your question related to given context")
submit = st.button('Generate Answer')




if submit:
    st.subheader("good question! my answer would be...")
    output = nlp({
    'question': user_question,
    'context': user_context
    })
    st.write(output["answer"])