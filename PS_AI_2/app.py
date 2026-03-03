import streamlit as st
from qa_engine import load_documents, prepare_data, get_answer


st.set_page_config(page_title="Policy Q&A Bot")

st.title("Company Policy Q&A Bot")
st.write("Ask questions related to Leave, IT, or Travel policies.")


documents = load_documents()
sentences, sources, vectorizer, tfidf_matrix = prepare_data(documents)


query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        answer, source = get_answer(query, sentences, sources, vectorizer, tfidf_matrix)

        if source:
            st.success("Answer Found")
            st.write("### 📌 Answer:")
            st.write(answer)

            st.write("### Document Used:")
            st.info(source)
        else:
            st.error(answer)