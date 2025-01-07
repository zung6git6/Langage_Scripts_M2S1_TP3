#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Modules to import
from unstructured.partition.auto import partition
from collections import defaultdict
import functools
import chromadb
from rake_nltk import Rake
from chromadb.config import Settings
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm
import asyncio
import streamlit as st
import re
from math import ceil
import argparse
from typing import List
from pathlib import Path
from datetime import date
import string
import time
import logging
import statistics
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Decorateur
def add_method_index(cls):
    """
    Ajoute une méthode `id_index` à une classe donnée, qui attribue un index unique à une instance.

    Args:
        cls (type): La classe à laquelle la méthode `id_index` sera ajoutée.

    Returns:
        type: La classe modifiée avec la méthode `id_index`.
    """
    index = 1  # Initialise un index global

    # Définition de la méthode qui sera ajoutée à la classe
    def method_index(self):
        nonlocal index
        self.index = index  # Associe l'index actuel à l'instance
        index += 1  # Incrémente l'index pour la prochaine instance

    cls.id_index = method_index  # Ajoute la méthode à la classe
    return cls  # Retourne la classe modifiée


# Dictionnaire global pour suivre les temps d'exécution des fichiers
temps_fichier = defaultdict(float)

def monitoring(func):
    """
    Décorateur pour surveiller les performances et gérer les erreurs lors de l'exécution d'une fonction.

    Fonctionnalités :
    - Gère les journaux d'erreurs et d'informations dans un fichier `Rapport_Logging.log`.
    - Mesure le temps d'exécution d'une fonction.
    - Génère un rapport de performance lorsque la fonction décorée est `main`.

    Args:
        func (Callable): La fonction à décorer.

    Returns:
        Callable: La fonction décorée.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global temps_fichier

        # Configure le logger
        logger = logging.getLogger("ErrorLogger")
        logger.setLevel(logging.INFO)

        # S'assurer que le handler est ajouté une seule fois
        if not logger.handlers:
            handler = logging.FileHandler("Rapport_Logging.log", "w")  # Mode écriture
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Si la fonction décorée est `main`, génère un rapport
        if func.__name__ == "main":
            result = func(*args, **kwargs)

            # Calcul des statistiques à partir des temps enregistrés
            TempsList = list(temps_fichier.values())
            print(TempsList)

            TempsMoyenTraitement = round(sum(TempsList) / len(TempsList), 3)
            TempsTotalTraitement = round(sum(TempsList), 3)
            TempsMedianTraitement = round(statistics.median(TempsList), 3)

            # Affichage et enregistrement des statistiques
            print("Le rapport du monitoring : ")
            logger.info("Le rapport du monitoring : \n")
            print(f"Le temps total de l'exécution : {TempsTotalTraitement}")
            logger.info(f"Le temps total de l'exécution : {TempsTotalTraitement}\n")
            print(f"Le temps moyen de l'exécution : {TempsMoyenTraitement}")
            logger.info(f"Le temps moyen de l'exécution : {TempsMoyenTraitement}\n")
            print(f"Le temps médian de l'exécution : {TempsMedianTraitement}")
            logger.info(f"Le temps médian de l'exécution : {TempsMedianTraitement}\n")

        else:
            # Récupère le fichier à partir des arguments ou des mots-clés
            fichier = kwargs.get('fichier') or (args[0] if args else None)

            # Vérifie si le fichier est un chemin valide
            if isinstance(fichier, (str, Path)):
                fichier = Path(fichier)
                
                if not fichier.exists():
                    logger.error(f"Le fichier {fichier} n'existe pas")
                    raise ValueError(f"Le fichier {fichier} n'existe pas")
            else:
                logger.error("Un fichier (chemin en tant que chaîne ou objet Path) doit être fourni.")
                raise ValueError("Un fichier (chemin en tant que chaîne ou objet Path) doit être fourni.")
            
            # Mesure le temps d'exécution de la fonction
            debut = time.perf_counter()
            result = func(*args, **kwargs)
            fin = time.perf_counter()

            temps_execution = fin - debut  # Temps d'exécution

            # Journalise les résultats ou les avertissements
            if not result:
                logger.warning(f"Aucun texte n'est extrait à partir du fichier {fichier}")
            else:
                logger.info(f"Le fichier {fichier.name} est traité avec un temps de {temps_execution:.6f} secondes")

            # Enregistre le temps d'exécution pour le fichier
            temps_fichier[str(fichier.name)] = temps_execution

        return result
    
    return wrapper

# Class
@add_method_index
class Document:
    """
    Classe représentant un document avec un contenu, une date de collecte et un identifiant unique.

    Attributs:
        Content (str): Le contenu du document.
        CollectDate (str): La date de collecte du document au format "JJ/MM/AAAA".
        Name (str): Le nom du fichier sans ponctuation ni extension.
        ID (str): Identifiant unique généré pour le document.
    """

    def __init__(self, file_name: str, content: str, Url:str=None):
        """
        Initialise un objet Document avec un nom de fichier et un contenu.

        Args:
            file_name (str): Le chemin ou nom du fichier source.
            content (str): Le contenu du document.
        """
        self.Content = content
        self.CollectDate = date.today().strftime("%d/%m/%Y")  # Date de collecte actuelle
        self.Name = ""
        self.Url = Url

        # Nettoyage du nom du fichier pour exclure la ponctuation
        for c in Path(file_name).stem:
            if c not in string.punctuation:
                self.Name += c

        # Génération de l'index via la méthode ajoutée par `add_method_index`
        self.id_index()

        # Génération d'un identifiant unique pour le document
        self.ID = f"inalco{{{self.index}}}{{{self.CollectDate}}}{{{self.Name[:10]}}}"

    def __str__(self) -> str:
        """
        Représentation sous forme de chaîne de caractères de l'objet Document.

        Returns:
            str: Le contenu du document.
        """
        return self.Content

    def __add__(self, field_info: tuple) -> None:
        """
        Permet d'ajouter dynamiquement un champ personnalisé au document via l'opérateur `+`.

        Args:
            field_info (tuple): Un tuple contenant le nom du champ et son contenu.

        Raises:
            ValueError: Si l'argument n'est pas un tuple ou n'a pas exactement deux éléments.
        """
        if isinstance(field_info, tuple) and len(field_info) == 2:
            field_name, field_content = field_info
        else:
            raise ValueError("La variable à concaténer avec l'objet Document n'est pas un tuple.")

        # Ajout dynamique de l'attribut au document
        setattr(self, field_name, field_content)

# User functions
def extraire_PDFtexte(fichier: str) -> str:
    """
    Extrait le texte d'un fichier PDF en utilisant la bibliothèque Unstructured.

    Args:
        fichier (str): Chemin vers le fichier PDF.

    Returns:
        str: Texte extrait du fichier sous forme de chaîne de caractères, 
             ou un message d'erreur en cas d'échec.
    """
    try:
        # Partition automatique selon le type de contenu dans le fichier
        documents = partition(filename=fichier)
        
        # Combine le contenu textuel des différents blocs en une seule chaîne
        texte = "[SEP]".join([doc.text for doc in documents if doc.text])
        return texte
    
    except Exception as e:
        return f"Erreur lors de l'extraction : {str(e)}"

def extraire_HTMLtexte(fichier: str) -> str:
    """
    Extrait le texte d'un fichier HTML en nettoyant le contenu avec BeautifulSoup.

    Args:
        fichier (str): Chemin vers le fichier HTML.

    Returns:
        str: Texte extrait du fichier sous forme de chaîne de caractères, avec des
             séparateurs `[SEP]` pour les sections.
    """
    # Lire le fichier HTML
    with open(fichier, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Parser le contenu HTML
    soup = BeautifulSoup(content, 'html.parser')

    # Extraire l'URL
    link_tag = soup.find('link', rel='canonical')
    Url = link_tag['href'] if link_tag else None 
    
    # Extraire tout le texte et nettoyer les espaces multiples ou sauts de ligne
    text = soup.get_text()
    text = re.sub(r"\n{2,}", "\n", text)  # Réduit les sauts de ligne multiples
    text = re.sub(r"\s{2,}", " ", text)   # Réduit les espaces multiples
    text = re.sub(r"\n", "[SEP]", text)   # Remplace les sauts de ligne par `[SEP]`

    return text, Url 

def extraire_texte(fichier: str) -> str:
    """
    Détermine le type de fichier (PDF ou HTML) et extrait le texte en conséquence.

    Args:
        fichier (str): Chemin vers le fichier à traiter.

    Returns:
        str: Texte extrait du fichier, ou `None` si le type n'est pas supporté.
    """
    fichier_suffixe = Path(fichier).suffix
    if fichier_suffixe == ".pdf":
        return extraire_PDFtexte(fichier)
    elif fichier_suffixe == ".html":
        return extraire_HTMLtexte(fichier)
    print("Merci de passer un fichier PDF ou HTML")
    return None

async def creer_Document(fichier: str):
    """
    Crée un objet Document à partir d'un fichier donné.

    Args:
        fichier (str): Chemin vers le fichier source (PDF ou HTML).

    Returns:
        Document: Instance de la classe Document contenant le texte extrait.
    """
    Url = ""

    if str(fichier).endswith("html"):
        texte, Url = extraire_texte(fichier)
    else:
        texte = extraire_texte(fichier)
    
    # Si le texte extrait est vide, créer un document avec un contenu par défaut
    if not texte:
        texte = "[[[Document Vide]]]"

    return Document(fichier, texte, Url)

def texte_chunkeur(texte: str, taille: int = 500) -> list:
    """
    Divise un texte en morceaux (`chunks`) d'une taille maximale donnée.

    Args:
        texte (str): Texte à diviser, avec des sections séparées par `[SEP]`.
        taille (int, optional): Taille maximale d'un chunk. Par défaut, 500 caractères.

    Returns:
        list: Liste de morceaux (`chunks`) de texte divisés en respectant la taille limite.
    """
    phrases = texte.split('[SEP]')  # Divise le texte en phrases à partir du séparateur
    chunks = []
    chunk_actuel = ""
    
    for phrase in phrases:
        # Vérifier si ajouter la phrase dépasse la taille limite
        if len(chunk_actuel) + len(phrase) + len('[SEP]') > taille:
            # Si le chunk actuel n'est pas vide, l'ajouter aux chunks
            if chunk_actuel:
                chunks.append(chunk_actuel.strip('[SEP]'))
                chunk_actuel = ""
            
            # Si la phrase seule dépasse la taille limite, la scinder
            while len(phrase) > taille:
                chunks.append(phrase[:taille])
                phrase = phrase[taille:]
        
        # Ajouter la phrase au chunk actuel
        chunk_actuel += phrase + '[SEP]'
    
    # Ajouter le dernier chunk s'il reste du contenu
    if chunk_actuel:
        chunks.append(chunk_actuel.strip('[SEP]'))
    
    return chunks

@add_method_index
class Chunk:
    """
    Classe représentant un morceau (chunk) de texte extrait d'un document.

    Attributs :
        Content (str) : Le contenu textuel du chunk.
        ID (str) : Identifiant unique du chunk, basé sur l'identifiant du document source.
    """
    def __init__(self, content: str, ID_doc: str, Url_doc):
        """
        Initialise un objet Chunk avec son contenu et son identifiant basé sur le document source.

        Args:
            content (str): Le contenu textuel du chunk.
            ID_doc (str): L'identifiant unique du document source auquel ce chunk appartient.
        """
        self.Content = content
        self.id_index()  # Ajoute un index unique au chunk
        self.ID = f"{{{ID_doc}}}chunk{{{self.index}}}"
        self.Url = Url_doc

    def __str__(self) -> str:
        """
        Retourne une représentation textuelle du chunk.

        Returns:
            str: Le contenu textuel du chunk.
        """
        return self.Content


async def Document2Chunks(document: Document) -> List[Chunk]:
    """
    Divise un document en morceaux (chunks) en utilisant une fonction de découpage (`texte_chunkeur`),
    et crée des instances de la classe Chunk pour chaque morceau.

    Args:
        document (Document): Le document à diviser en chunks.

    Returns:
        List[Chunk]: Une liste d'objets Chunk, chacun représentant un morceau de texte du document.
    """
    # Récupérer l'identifiant unique du document
    ID_doc = document.ID

    # Récupérer l'URL du document
    Url_doc = document.Url

    # Récupérer le contenu textuel du document
    texte_doc = document.Content

    # Diviser le texte du document en chunks
    chunks: List[str] = texte_chunkeur(texte_doc)

    # Créer des objets Chunk pour chaque morceau de texte
    Chunks: List[Chunk] = [Chunk(chunk, ID_doc, Url_doc) for chunk in chunks]

    return Chunks

def creer_BDD_et_purger(nom_collection: str, purger: bool = False):
    """
    Crée une base de données ou une collection spécifique dans ChromaDB. Si la collection existe déjà, 
    elle peut être purgée avant de continuer.

    Args:
        nom_collection: Nom de la collection à créer ou à récupérer.
        purger: Indique si la collection existante doit être purgée (par défaut False).
    Return: 
        Une instance de la collection prête à être utilisée.
    """
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Initialisation du client ChromaDB
    client = chromadb.Client(tenant="default_tenant")
    
    # Vérification si la collection existe déjà
    if nom_collection in [coll.name for coll in client.list_collections()]:
        collection = client.get_collection(nom_collection)
        if purger:
            collection.delete()  # Suppression des données existantes
    else:
        # Création d'une nouvelle collection
        collection = client.create_collection(nom_collection)
    
    return collection


async def stocker_chunk(collection, chunk_id: str, chunk_content: str, document_id: str, position: int, link:str, metadata: dict = None):
    """
    Ajoute un chunk à une collection ChromaDB, accompagné de métadonnées pour faciliter la recherche et la reconstruction.

    Args:
        collection: Instance de la collection ChromaDB à laquelle ajouter le chunk.
        chunk_id: Identifiant unique du chunk.
        chunk_content: Contenu textuel du chunk.
        document_id: Identifiant du document source auquel le chunk appartient.
        position: Position du chunk dans le document.
        links : Lien URL potentiel du chunk.
        metadata: Métadonnées additionnelles associées au chunk (facultatif).
    """
    # Ajout des métadonnées par défaut
    if metadata is None:
        metadata = {}

    metadata['document_id'] = document_id
    metadata['position'] = position
    metadata['lien'] = link

    # Ajout du chunk à la collection
    collection.add(
        ids=[chunk_id],         # Liste contenant l'ID du chunk
        documents=[chunk_content],  # Liste contenant le contenu textuel du chunk
        metadatas=[metadata]    # Liste contenant les métadonnées associées
    )


@monitoring
async def CreerDocEtChunksIndex(nom_fichier: str, collection):
    """
    Crée un document à partir d'un fichier, découpe son contenu en chunks, et stocke les chunks dans une collection.

    Args:
        nom_fichier: Chemin du fichier source (PDF ou HTML).
        collection: Instance de la collection ChromaDB où stocker les chunks.
    Return:
        1 si le processus est réussi, None si le document est vide.
    """
    # Créer un document à partir du fichier
    document = await creer_Document(nom_fichier)
    
    if document.Content == "[[[Document Vide]]]":
        return None

    # Découper le document en chunks
    chunks = await Document2Chunks(document)
    
    # Stocker chaque chunk dans la collection
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.ID
        chunk_content = chunk.Content
        chunk_url = chunk.Url
        await stocker_chunk(collection, chunk_id, chunk_content, document.ID, i, chunk_url)
    
    return 1


async def traiter_fichier_en_parallele(fichiers: list, collection):
    """
    Traite plusieurs fichiers en parallèle en les convertissant en documents, puis en chunks, et les stocke dans une collection.
    
    Args:
        fichiers: Liste de chemins de fichiers à traiter.
        collection: Instance de la collection ChromaDB où stocker les chunks.
    Return:
        Une liste des résultats des tâches parallèles (1 pour succès, None pour documents vides).
    """
    return await tqdm.gather(*[CreerDocEtChunksIndex(fichier, collection) for fichier in fichiers])

@monitoring
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

    return collection

# Main procedure
if __name__ == "__main__":

    collection = main()
    print(f"Nom de la collection: {collection.name}")
    documents = collection.get()['documents']  # Récupérer les documents
    ids = collection.get()['ids'] # Récupérer les ids
    metadatas = collection.get()['metadatas']
    liens = [meta['lien'] for meta in metadatas]

    # Afficher les 5 premiers documents
    for id, chunk, lien in zip(ids, documents, liens):
        if lien:
            print("URL :", lien)
        else:
            print("URL : Lien manquant")
        print(f"{id}: {chunk}")
        print("\n"*2)