import streamlit as st
import langchainHelper
	 
st.title('Restaurant Name Generator')
cuisine = st.sidebar.selectbox("Pick a cuisine",("Indian","Mexican","American","Italian"))
if cuisine:
    response = langchainHelper.generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'].strip())
    menu_items = response['menu_items'].split(",")
    st.write("Menu Items")
    for item in menu_items:
        st.write(item)

