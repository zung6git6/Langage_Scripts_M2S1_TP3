#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Modules to import
from Partie2 import creer_BDD_et_purger, traiter_fichier_en_parallele
from collections import defaultdict
from rake_nltk import Rake
import asyncio
import streamlit as st
import argparse
from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def documents_retriever_par_mots_cles(mots_cles: list, collection, nb_doc: int = 1):
    """
    Recherche les documents les plus pertinents dans une collection en fonction de mots-clés donnés.

    Args:
        mots_cles: Liste de mots-clés à rechercher.
        collection: Instance de la collection ChromaDB où effectuer la recherche.
        nb_doc: Nombre de documents les plus pertinents à retourner (par défaut 1).
    Return:
        Un dictionnaire associant les identifiants des documents aux contenus pertinents.
    """
    chunks_modifies = []
    pertinents_documents_scores = defaultdict(int)
    
    # Conversion des mots-clés en minuscules pour une recherche insensible à la casse
    mots_cles = list(map(lambda x: x.lower(), mots_cles))
    
    # Récupération des chunks, de leurs identifiants et de liens URL
    chunks = collection.get()['documents']
    ids = collection.get()['ids']
    metadatas = collection.get()['metadatas']
    liens = [meta['lien'] for meta in metadatas]
    doc2text = {}
    
    # Calcul des scores de pertinence pour chaque chunk
    for chunk, ID, url in zip(chunks, ids, liens):
        document_id = ID.split('chunk')[0]
        doc2text[document_id] = [url]
        score = sum(chunk.lower().count(mot) for mot in mots_cles)
        chunk_modifie = chunk
        if score > 0:
            pertinents_documents_scores[document_id] += score
            chunk_modifie = "<mark>" + chunk + "</mark>"  # Mise en surbrillance des chunks pertinents
        chunks_modifies.append(chunk_modifie)
            
    # Trier les documents par score décroissant et sélectionner les meilleurs
    best_documents = list(dict(sorted(pertinents_documents_scores.items(), key=lambda x: x[1], reverse=True)).keys())
    
    if nb_doc <= len(best_documents):
        best_documents = best_documents[:nb_doc]

    # Regrouper les chunks pertinents par document
    for best_document in best_documents:
        pertinent_chunks = []
        for chunk_modifie, ID in zip(chunks_modifies, ids):
            document_id = ID.split('chunk')[0]
            if document_id == best_document:
                pertinent_chunks.append(chunk_modifie)
        doc2text[best_document].append("\n".join("[SEP]".join(pertinent_chunks).split("[SEP]")))

    return doc2text

def mots_cles_retriever(question:str):
    # Uses stopwords for english from NLTK, and all puntuation characters by
    # default
    r = Rake(language="french", max_length=1)

    # Extraction given the text.
    r.extract_keywords_from_text(question)

    # To get keyword phrases ranked highest to lowest.
    mots_cles = r.get_ranked_phrases()

    return mots_cles

def streamlit_constructeur(collection):
    # Titre de l'application
    st.title("Recherche de Documents")

    # Initialisation de la session pour la pagination et les résultats
    if 'page' not in st.session_state:
        st.session_state.page = 1
    if 'doc2text' not in st.session_state:
        st.session_state.doc2text = {}

    # Recherche et sauvegarde des résultats dans la session
    with st.form(key='my_form'):
        # Champ de texte pour que l'utilisateur puisse poser sa question
        question = st.text_input("Entrez votre question ou mots-clés :", "")
        # Sélecteur pour choisir le nombre de documents maximum à retourner
        nb_doc = st.slider("Nombre maximum de documents à retourner :", 1, 20, 1)
        submit_button = st.form_submit_button(label='Recherche')

        # Si le formulaire est soumis, effectuer la recherche et enregistrer les résultats dans la session
        if submit_button:
            if question:
                mots_cles = mots_cles_retriever(question)
                # Appel de la fonction de recherche
                st.session_state.doc2text = documents_retriever_par_mots_cles(mots_cles, collection, nb_doc)
                st.session_state.page = 1  # Réinitialiser à la première page après une nouvelle recherche

    # Récupérer les résultats de la session
    doc2text = st.session_state.doc2text
    if doc2text:
        # Nombre total de documents trouvés
        total_docs = len(doc2text)

        # Nombre total de pages (1 page = 1 document)
        num_pages = total_docs

        # Afficher le nombre total de documents
        st.markdown(f"**{total_docs} documents trouvés.**")

        # Utiliser un selectbox pour naviguer entre les pages (ou un slider si vous préférez)
        current_page = st.selectbox("Sélectionnez la page", range(1, num_pages + 1), key="page_selector")
        st.session_state.page = current_page

        # Afficher le document courant
        doc_id, url_contenu = list(doc2text.items())[current_page - 1]
        url, contenu = url_contenu

        # Afficher le titre ou l'ID du document
        st.subheader(f"Document ID: {doc_id[1:-1]}")

        # Mettre en avant les mots-clés dans le texte du document
        contenu_html = contenu

        # Afficher le lien URL
        if url:
            st.markdown(f"***URL : {url}***", unsafe_allow_html=True)
        else:
            st.markdown(f"***URL : URL manquant***", unsafe_allow_html=True)

        # Afficher le contenu du document avec la mise en forme HTML
        st.markdown(contenu_html, unsafe_allow_html=True)

        # Afficher le numéro de la page actuelle
        st.markdown(f"Page {current_page} sur {num_pages}")

    else:
        st.warning("Aucun document pertinent trouvé pour cette question.")

def main():
    collection = creer_BDD_et_purger("Inalco")

    parser = argparse.ArgumentParser(prog="Data Prep")
    parser.add_argument("nom_repertoire")
    args = parser.parse_args()
    dossier = Path(args.nom_repertoire)

    fichiers = [file for file in dossier.rglob("*") if file.suffix in [".pdf", ".html"]]
    # total_iterations = len(fichiers)
    # progress_bar = tqdm(total=total_iterations, desc="Lecture des fichiers PDF ou HTML")
    asyncio.run(traiter_fichier_en_parallele(fichiers, collection))

    streamlit_constructeur(collection)

    return 0

# Main procedure
if __name__ == "__main__":
    main()