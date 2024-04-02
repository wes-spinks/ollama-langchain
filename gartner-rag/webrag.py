import streamlit as st
from run_rag import ChatAi

st.title("Gartner Markets Bot")
st.caption("CSV sourced from: https://www.gartner.com/reviews/markets")

bot = ChatAi()
bot.load()

def _resp(textinput):
    answ = bot.ask(textinput)
    st.info(answ)


with st.form("my_form"):
    text = st.text_area('Submit your inquiry:', 'Describe the Cloud AI Developer Services market. List the products in that market.')
    submitted = st.form_submit_button('Ask AI')
    if submitted:
        _resp(text)