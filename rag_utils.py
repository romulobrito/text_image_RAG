import streamlit as st

from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from langchain_core.runnables import chain
from typing import List, Any, Dict, Optional
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from collections import defaultdict

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain_core.runnables import RunnableLambda
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from typing import Literal, List
import os


import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

api_key = openai_key = "xxxxxxxxxxxx"

# ====================================================================
# Load databases and initialize the retrievers
# ====================================================================
# Instantiate the OpenAIEmbeddings class
openai_embeddings = OpenAIEmbeddings(api_key=api_key)
# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# https://python.langchain.com/v0.2/docs/how_to/add_scores_retriever/

import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

# def process_and_add_pdf(uploaded_file=None, pdf_path=None):
#     # Extrair texto do PDF
#     logger.info("🔄 Iniciando processamento do PDF...")
#     try:
#         if uploaded_file:
#             logger.info(f"📤 Processando arquivo enviado pelo usuário")
#             pdf_reader = PyPDF2.PdfReader(uploaded_file)
#         elif pdf_path:
#             logger.info(f"📂 Processando arquivo do caminho: {pdf_path}")
#             with open(pdf_path, "rb") as f:
#                 pdf_reader = PyPDF2.PdfReader(f)
#                 text = ""
#                 for page in pdf_reader.pages:
#                     text += page.extract_text()
#         else:
#             raise ValueError("No PDF file provided.")
#         logger.info(f"📄 Texto extraído com sucesso. Tamanho: {len(text)} caracteres")

#         # Dividir o texto em chunks e resumos
#         parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
#         child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

#         parent_chunks = parent_splitter.split_text(text)
#         child_chunks = child_splitter.split_text(text)

#         # Criar documentos para indexação
#         parent_docs = [Document(page_content=chunk) for chunk in parent_chunks]
#         child_docs = [Document(page_content=chunk) for chunk in child_chunks]

#         # Indexar os documentos usando FAISS
#         vectorstore = FAISS.from_documents(parent_docs, openai_embeddings)
#         vectorstore.save_local("/home/romulobrito/projetos/image-RAG/text_image_rag/indices/chunks/chunk_vectorstore_marco")

#         summary_vectorstore = FAISS.from_documents(child_docs, openai_embeddings)
#         summary_vectorstore.save_local("/home/romulobrito/projetos/image-RAG/text_image_rag/indices/summaries/whole_doc_summary_vectorstore_marco")

#         logger.info(f"✂️ Chunks gerados: {len(parent_chunks)} parent, {len(child_chunks)} child")
#         return vectorstore, summary_vectorstore
#     except Exception as e:
#         logger.error(f"❌ Erro no processamento do PDF: {e}")
#         raise

def process_and_add_pdf(uploaded_file=None, pdf_path=None):
    # Definir caminhos dos índices
    base_path = "/home/romulobrito/projetos/image-RAG/text_image_rag/indices"
    chunks_path = f"{base_path}/chunks/chunk_vectorstore_marco"
    summaries_path = f"{base_path}/summaries/whole_doc_summary_vectorstore_marco"
    
    # Criar diretórios se não existirem
    for path in [f"{base_path}/chunks", f"{base_path}/summaries"]:
        os.makedirs(path, exist_ok=True)
    
    logger.info("🔄 Iniciando processamento do PDF...")
    try:
        if uploaded_file:
            logger.info(f"📤 Processando arquivo enviado pelo usuário")
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
        elif pdf_path:
            logger.info(f"📂 Processando arquivo do caminho: {pdf_path}")
            with open(pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
        else:
            raise ValueError("No PDF file provided.")
        
        logger.info(f"📄 Texto extraído com sucesso. Tamanho: {len(text)} caracteres")

        # Dividir o texto em chunks e resumos
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

        parent_chunks = parent_splitter.split_text(text)
        child_chunks = child_splitter.split_text(text)

        # Criar documentos para indexação
        parent_docs = [Document(page_content=chunk) for chunk in parent_chunks]
        child_docs = [Document(page_content=chunk) for chunk in child_chunks]

        logger.info(f"✂️ Chunks gerados: {len(parent_chunks)} parent, {len(child_chunks)} child")

        # Indexar os documentos usando FAISS
        vectorstore = FAISS.from_documents(parent_docs, openai_embeddings)
        vectorstore.save_local(chunks_path)
        logger.info(f"💾 Índice de chunks salvo em: {chunks_path}")

        summary_vectorstore = FAISS.from_documents(child_docs, openai_embeddings)
        summary_vectorstore.save_local(summaries_path)
        logger.info(f"💾 Índice de resumos salvo em: {summaries_path}")

        return vectorstore, summary_vectorstore
    except Exception as e:
        logger.error(f"❌ Erro no processamento do PDF: {e}")
        raise

class CustomParentDocumentRetriever:
    def __init__(self, vectorstore, docstore, child_splitter, parent_splitter, search_kwargs):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.child_splitter = child_splitter
        self.parent_splitter = parent_splitter
        self.search_kwargs = search_kwargs

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents based on the query."""
        results = self.vectorstore.similarity_search_with_score(query, **self.search_kwargs)
        id_to_doc = defaultdict(list)
        for doc, score in results:
            doc_id = doc.metadata.get("doc_id")
            if doc_id:
                doc.metadata["score"] = score
                id_to_doc[doc_id].append(doc)

        docs = []
        for _id, sub_docs in id_to_doc.items():
            docstore_docs = self.docstore.mget([_id])
            if docstore_docs:
                if doc := docstore_docs[0]:
                    doc.metadata["sub_docs"] = sub_docs
                    docs.append(doc)

        return docs



# def load_chunk_retriever() -> CustomParentDocumentRetriever:
#     vectorstore = FAISS.load_local(
#         "/home/romulobrito/projetos/image-RAG/text_image_rag/indices/chunks/chunk_vectorstore_marco",
#         openai_embeddings,
#         allow_dangerous_deserialization=True
#     )
#     fs = LocalFileStore("/home/romulobrito/projetos/image-RAG/text_image_rag/indices/chunks/chunk_docstore_marco")
#     store = create_kv_docstore(fs)
#     return CustomParentDocumentRetriever(
#         vectorstore=vectorstore,
#         docstore=store,
#         child_splitter=child_splitter,
#         parent_splitter=parent_splitter,
#         search_kwargs={'score_threshold': 0.5, 'k': 4}
#     )


def load_chunk_retriever():
    logger.info("🔄 Carregando/Inicializando retrievers...")
    
    base_path = "/home/romulobrito/projetos/image-RAG/text_image_rag/indices"
    chunks_path = f"{base_path}/chunks/chunk_vectorstore_marco"
    index_file = os.path.join(chunks_path, "index.faiss")
    
    # Criar diretórios se não existirem
    os.makedirs(chunks_path, exist_ok=True)
    
    # Verificar se o arquivo de índice existe
    if not os.path.exists(index_file):
        logger.info("🔄 Índice não encontrado. Criando novo índice...")
        
        # Criar o índice a partir do PDF padrão
        pdf_path = "/home/romulobrito/projetos/image-RAG/text_image_rag/Injection Mold Design Handbook - preview-9781569908167_A42563111.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF não encontrado em: {pdf_path}")
            
        vectorstore, _ = process_and_add_pdf(pdf_path=pdf_path)
        logger.info("✅ Novo índice criado com sucesso")
    else:
        # Carregar índice existente com allow_dangerous_deserialization=True
        vectorstore = FAISS.load_local(
            folder_path=chunks_path,
            embeddings=openai_embeddings,
            index_name="index",
            allow_dangerous_deserialization=True  # Adicionado este parâmetro
        )
        logger.info("✅ Índice FAISS carregado com sucesso")

    # Configurar o retriever
    search_kwargs = {"k": 5}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)

def load_summary_retriever() -> chain:
    vectorstore = FAISS.load_local(
        "/home/romulobrito/projetos/image-RAG/text_image_rag/indices/summaries/whole_doc_summary_vectorstore_marco",
        openai_embeddings,
        allow_dangerous_deserialization=True
    )

    @chain
    def whole_doc_summary_retriever(query: str) -> List[Document]:
        docs, scores = zip(*vectorstore.similarity_search_with_score(query=query, k=2))
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score
        return docs

    return whole_doc_summary_retriever

# No início do arquivo, junto com os outros imports
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from typing import List, Tuple, Any, Dict
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from pathlib import Path

# Cache do modelo de similaridade para evitar recarregamento
@st.cache_resource
def get_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def describe_and_rank_images(query: str, answer: Any, image_dir: str, n: int = 3) -> List[Tuple[str, str, float]]:
    """
    Retorna os caminhos das imagens mais relevantes junto com suas descrições e relevância.
    
    Args:
        query (str): A pergunta do usuário
        answer (Any): A resposta gerada pelo sistema (pode ser string ou AIMessage)
        image_dir (str): Diretório onde estão as imagens
        n (int): Número máximo de imagens para retornar
        
    Returns:
        List[Tuple[str, str, float]]: Lista de tuplas (caminho_imagem, descrição, relevância)
    """
    logger.info(f"🖼️ Iniciando processamento de imagens para query: {query[:50]}...")

    try:
        # Extrair texto da resposta se for AIMessage
        if hasattr(answer, 'content'):
            answer_text = answer.content
        else:
            answer_text = str(answer)

        # Verificar se o diretório existe
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"Diretório de imagens não encontrado: {image_dir}")
            return []
        
        # Inicializar o modelo para descrição de imagens
        description_llm = ChatOpenAI(temperature=0.3, model='gpt-4', openai_api_key=api_key)
        
        # Obter modelo de similaridade do cache
        similarity_model = get_similarity_model()
        
        # Template para gerar descrições técnicas das imagens
        description_template = """Você é um especialista em moldes de injeção plástica.
        Analise a imagem e gere uma descrição técnica detalhada do que ela provavelmente representa dentro do contexto da resposta e da query fornecida.
        
        Nome do arquivo: {image_name}
        
        Gere uma descrição técnica focando em:
        1. Aspectos técnicos do molde ou peça
        2. Princípios de design mostrados
        3. Boas práticas ilustradas
        4. Possíveis aplicações e contextos de uso
        
        Descrição:"""
        
        # Obter caminhos das imagens válidas
        image_paths = []
        for file_path in image_dir.rglob("*"):
            if file_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                try:
                    # Verificar se a imagem é válida
                    with Image.open(file_path) as img:
                        img.verify()
                    image_paths.append(str(file_path))
                except Exception as e:
                    print(f"Imagem inválida ignorada {file_path}: {e}")
                    continue
        
        if not image_paths:
            print("Nenhuma imagem válida encontrada")
            return []
            
        # Gerar descrições técnicas para cada imagem
        image_descriptions = []
        for image_path in image_paths:
            try:
                file_name = Path(image_path).name
                # Gerar descrição técnica usando o LLM
                description_prompt = description_template.format(image_name=file_name)
                description = description_llm.invoke(description_prompt)
                if isinstance(description, str):
                    image_descriptions.append((image_path, description))
                else:
                    # Se a resposta não for string, tentar extrair o conteúdo
                    description_text = getattr(description, 'content', str(description))
                    image_descriptions.append((image_path, description_text))
            except Exception as e:
                print(f"Erro ao gerar descrição para {image_path}: {e}")
                continue
        
        # Preparar o contexto de busca
        search_context = f"""
        Contexto técnico de moldes de injeção:
        Pergunta do usuário: {query}
        Resposta técnica: {answer_text}
        """
        
        # Calcular embeddings usando sentence-transformers
        context_embedding = similarity_model.encode(search_context, convert_to_tensor=True)
        
        # Calcular similaridade e relevância
        ranked_images = []
        for path, desc in image_descriptions:
            try:
                # Gerar embedding da descrição
                desc_embedding = similarity_model.encode(desc, convert_to_tensor=True)
                
                # Calcular similaridade coseno usando torch
                similarity = util.pytorch_cos_sim(context_embedding, desc_embedding).item()
                
                # Normalizar similaridade para [0,1]
                normalized_similarity = float((similarity + 1) / 2)
                
                # Adicionar à lista apenas se a similaridade for relevante
                if normalized_similarity > 0.3:  # Limiar de relevância
                    ranked_images.append((path, desc, normalized_similarity))
            except Exception as e:
                print(f"Erro ao calcular similaridade para {path}: {e}")
                continue
        
        # Ordenar por similaridade e pegar os top N
        ranked_images.sort(key=lambda x: x[2], reverse=True)
        top_images = ranked_images[:n]
        
        # Formatar as descrições finais
        formatted_results = []
        for path, desc, score in top_images:
            formatted_desc = f"""
            **Descrição Técnica do Componente**:
            {desc}
            
            **Contexto de Aplicação**:
            - Relevância para a consulta: {score:.2%}
            - Tipo de componente: {Path(path).stem}
            """
            formatted_results.append((path, formatted_desc, score))
        
        return formatted_results
        
    except Exception as e:
        print(f"Erro geral na função describe_and_rank_images: {e}")
        return []



# Sem utilizar sentence treansformers
# from PIL import Image
# import numpy as np
# from typing import List, Tuple, Any, Dict
# import os
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_core.documents import Document
# from pathlib import Path

# def describe_and_rank_images(query: str, answer: Any, image_dir: str, n: int = 3) -> List[Tuple[str, str, float]]:
#     """
#     Retorna os caminhos das imagens mais relevantes junto com suas descrições e relevância.
    
#     Args:
#         query (str): A pergunta do usuário
#         answer (Any): A resposta gerada pelo sistema (pode ser string ou AIMessage)
#         image_dir (str): Diretório onde estão as imagens
#         n (int): Número máximo de imagens para retornar
        
#     Returns:
#         List[Tuple[str, str, float]]: Lista de tuplas (caminho_imagem, descrição, relevância)
#     """
#     try:
#         # Extrair texto da resposta se for AIMessage
#         if hasattr(answer, 'content'):
#             answer_text = answer.content
#         else:
#             answer_text = str(answer)

#         # Verificar se o diretório existe
#         image_dir = Path(image_dir)
#         if not image_dir.exists():
#             print(f"Diretório de imagens não encontrado: {image_dir}")
#             return []
        
#         # Inicializar modelos
#         description_llm = ChatOpenAI(temperature=0.3, model='gpt-4', openai_api_key=api_key)
#         embeddings_model = OpenAIEmbeddings(api_key=api_key)
        
#         # Template para gerar descrições técnicas das imagens
#         description_template = """Você é um especialista em moldes de injeção plástica.
#         Analise o nome do arquivo da imagem e gere uma descrição técnica detalhada do que ela provavelmente representa.
        
#         Nome do arquivo: {image_name}
        
#         Gere uma descrição técnica focando em:
#         1. Aspectos técnicos do molde ou peça
#         2. Princípios de design mostrados
#         3. Boas práticas ilustradas
#         4. Possíveis aplicações e contextos de uso
        
#         Descrição:"""
        
#         # Obter caminhos das imagens válidas
#         image_paths = []
#         for file_path in image_dir.rglob("*"):
#             if file_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
#                 try:
#                     # Verificar se a imagem é válida
#                     with Image.open(file_path) as img:
#                         img.verify()
#                     image_paths.append(str(file_path))
#                 except Exception as e:
#                     print(f"Imagem inválida ignorada {file_path}: {e}")
#                     continue
        
#         if not image_paths:
#             print("Nenhuma imagem válida encontrada")
#             return []
            
#         # Gerar descrições técnicas para cada imagem
#         image_descriptions = []
#         for image_path in image_paths:
#             try:
#                 file_name = Path(image_path).name
#                 # Gerar descrição técnica usando o LLM
#                 description_prompt = description_template.format(image_name=file_name)
#                 description = description_llm.invoke(description_prompt)
#                 if isinstance(description, str):
#                     image_descriptions.append((image_path, description))
#                 else:
#                     # Se a resposta não for string, tentar extrair o conteúdo
#                     description_text = getattr(description, 'content', str(description))
#                     image_descriptions.append((image_path, description_text))
#             except Exception as e:
#                 print(f"Erro ao gerar descrição para {image_path}: {e}")
#                 continue
        
#         # Preparar o contexto de busca
#         search_context = f"""
#         Contexto técnico de moldes de injeção:
#         Pergunta do usuário: {query}
#         Resposta técnica: {answer_text}
#         """
        
#         # Calcular embedding do contexto
#         context_embedding = embeddings_model.embed_query(search_context)
        
#         # Calcular similaridade e relevância
#         ranked_images = []
#         for path, desc in image_descriptions:
#             try:
#                 # Gerar embedding da descrição
#                 desc_embedding = embeddings_model.embed_query(desc)
                
#                 # Calcular similaridade coseno
#                 similarity = np.dot(context_embedding, desc_embedding) / (
#                     np.linalg.norm(context_embedding) * np.linalg.norm(desc_embedding)
#                 )
                
#                 # Normalizar similaridade para [0,1]
#                 normalized_similarity = float((similarity + 1) / 2)
                
#                 # Adicionar à lista apenas se a similaridade for relevante
#                 if normalized_similarity > 0.3:  # Limiar de relevância
#                     ranked_images.append((path, desc, normalized_similarity))
#             except Exception as e:
#                 print(f"Erro ao calcular similaridade para {path}: {e}")
#                 continue
        
#         # Ordenar por similaridade e pegar os top N
#         ranked_images.sort(key=lambda x: x[2], reverse=True)
#         top_images = ranked_images[:n]
        
#         # Formatar as descrições finais
#         formatted_results = []
#         for path, desc, score in top_images:
#             formatted_desc = f"""
#             **Descrição Técnica do Componente**:
#             {desc}
            
#             **Contexto de Aplicação**:
#             - Relevância para a consulta: {score:.2%}
#             - Tipo de componente: {Path(path).stem}
#             """
#             formatted_results.append((path, formatted_desc, score))
        
#         return formatted_results
        
#     except Exception as e:
#         print(f"Erro geral na função describe_and_rank_images: {e}")
#         return []





# Inicialização dos retrievers
chunk_retriever = load_chunk_retriever()
summary_retriever = load_summary_retriever()

# Função para obter caminhos de imagens
def get_image_paths(doc_id: str) -> List[str]:
    logger.info(f"🔍 Buscando imagens para documento {doc_id}")
    image_dir = r"/home/romulobrito/projetos/image-RAG/text_image_rag/extracted_images/marco"
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.startswith(doc_id) and file.endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(root, file))
    logger.info(f"📸 Encontradas {len(image_paths)} imagens para {doc_id}")
    return image_paths

# Função para combinar texto e imagens
def combine_text_and_images(context: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
    for key, docs in context.items():
        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            if doc_id:
                image_paths = get_image_paths(doc_id)
                doc.metadata["image_paths"] = image_paths
                print(f"Added image paths to doc {doc_id}: {image_paths}")  # Adicione este print para depuração
    return context


# ====================================================================
# Preprocessing chain
# ====================================================================
class Condense(BaseModel):
    """Sumariza o histórico do chat e cria uma pergunta independente para recuperação RAG sobre moldes de injeção"""
    condensed_history: str = Field(description="histórico do chat sumarizado focado em aspectos técnicos de moldes")
    standalone_question: str = Field(description="pergunta independente condensada do histórico e pergunta de acompanhamento sobre moldes de injeção")

condense_functions = [convert_to_openai_function(Condense)]

condense_template = """Sumarize o histórico do chat de forma concisa,
e combine-o com a pergunta/solicitação de acompanhamento para criar uma pergunta independente para recuperação RAG.
Foque apenas em informações técnicas sobre moldes de injeção plástica que sejam relevantes para entender a pergunta.
Não inclua detalhes desnecessários ou perguntas anteriores que não agreguem ao contexto técnico.
Certifique-se de que a pergunta independente não repita informações já respondidas.

Exemplo 1:
Chat History:
User: Quais são os principais tipos de canais de refrigeração em moldes?
Assistant: Os principais tipos são canais retos, em cascata e conformais, cada um com suas vantagens específicas.
User: Como funcionam os canais conformais?

Condensed History: Usuário perguntou sobre tipos de canais de refrigeração, e agora quer entender especificamente os canais conformais.

Standalone Question: Qual é o princípio de funcionamento e as vantagens dos canais de refrigeração conformais em moldes de injeção?

Exemplo 2:
Chat History:
User: Quais são as melhores práticas para design de extratores?
Assistant: Os extratores devem ser posicionados estrategicamente para evitar marcas e garantir extração uniforme.
User: Como evitar marcas de extrator na peça?

Condensed History: Após discussão sobre design de extratores, usuário busca informações específicas sobre prevenção de marcas.

Standalone Question: Quais são as técnicas e considerações de design para minimizar marcas de extrator em peças injetadas?

Agora, processe o seguinte:

Chat History:
{chat_history}

Follow Up Question/Request: {question}

Forneça tanto o histórico condensado quanto a pergunta independente, mantendo o foco em aspectos técnicos de moldes de injeção.
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', openai_api_key=api_key)

llm_to_condense = llm.bind(
    functions=condense_functions,
    function_call={"name": "Condense"}
)

# turns nested dict into a flat dict
def format_query(input_dict):
    condensed_info_dict = input_dict.pop('condensed_info')
    input_dict['condensed_history'] = condensed_info_dict['condensed_history']
    input_dict['standalone_question'] = condensed_info_dict['standalone_question']
    return input_dict

preprocess_query_chain = RunnablePassthrough.assign(condensed_info = CONDENSE_QUESTION_PROMPT                                   
                                                    | llm_to_condense 
                                                    | JsonOutputFunctionsParser()) | format_query


# ====================================================================
# Router chain and retrieval chain
# ====================================================================
class RouteQuery1(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasources: List[Literal["summary_store", "vector_store"]] = Field(
        ...,
        description="Given a user question choose which datasources would be most relevant for answering their question",
    )

router_system_prompt1 = """You are an expert at routing user questions about plastic injection mold design to the appropriate data source.
There are two possible destinations:
1. vector_store: This store contains detailed chunks of text about injection mold design.
2. summary_store: This store contains summaries of injection mold design documents.

When deciding where to route a query, consider the following:
- If the query asks for detailed information, specific techniques, or in-depth content about plastic injection mold design, route it to the vector_store.
- If the query asks for an overview, summary, or general information about plastic injection mold design concepts, route it to the summary_store.
- If the query involves both types of information, route it to both stores.

Examples:
1. User query: "What are the basic principles of injection mold design?"
   Routing: summary_store

2. User query: "What are the specific requirements for designing cooling channels in an injection mold?"
   Routing: vector_store

3. User query: "Can you explain the importance of draft angles and provide detailed guidelines for different materials?"
   Routing: summary_store, vector_store
"""

router_prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", router_system_prompt1),
        ("human", "{standalone_question}"),
    ]
)

router_llm1 = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=api_key)
#router_llm1 = ChatOpenAI(temperature=0, model='gpt-4o-mini', openai_api_key=api_key)
structured_llm1 = router_llm1.with_structured_output(RouteQuery1)
router_chain1 = RunnablePassthrough.assign(classification1 = (lambda x: x['standalone_question']) | router_prompt1 | structured_llm1)


class RouteQuery2(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasources: List[Literal["marco"]] = Field(
        ...,
        description="Given a user question choose which datasources would be most relevant for answering their question about injection mold design",
    )

router_system_prompt2 = """You are an expert at routing user questions about plastic injection mold design.
The only destination available is:

1. marco: Contains information and resources related to the design and manufacturing of injection molds.

When deciding where to route a query, consider the following:
- If the query is about injection mold design, manufacturing, best practices, or any related topics, route it to "marco".
- All queries related to plastic injection molding should be routed to "marco".

Examples:

1. User query: "What are the best practices in Injection Mold Design?"
    Routing: marco

2. User query: "How to design cooling channels for injection molds?"
    Routing: marco

3. User query: "What are the key considerations for plastic part design?"
    Routing: marco

4. User query: "Can you show me examples of good mold design practices?"
    Routing: marco
"""

router_prompt2 = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt2),
    ("human", "{standalone_question}"),
])

router_llm2 = ChatOpenAI(temperature=0, model='gpt-4o-mini', openai_api_key=api_key)
#router_llm2 = ChatOpenAI(temperature=0, model='gpt-4o', openai_api_key=api_key)
structured_llm2 = router_llm2.with_structured_output(RouteQuery2)
router_chain2 = RunnablePassthrough.assign(classification2 = (lambda x: x['standalone_question']) | router_prompt2 | structured_llm2)

def remove_sub_docs(doc_dict: Dict[str, List[Document]]) -> Dict[str, List[Document]]:

    """Removes sub_docs from the context to save on LLM context window"""

    new_doc_dict = {}

    for key, docs in doc_dict.items():
        new_docs = []
        for doc in docs:
            if 'sub_docs' in doc.metadata:
                # Create a copy of the metadata without sub_docs
                new_metadata = {k: v for k, v in doc.metadata.items() if k != 'sub_docs'}
                new_doc = Document(page_content=doc.page_content, metadata=new_metadata)
                new_docs.append(new_doc)
            else:
                new_docs.append(doc)
        new_doc_dict[key] = new_docs

    return new_doc_dict


def route_query(output):
    """Route the query to the appropriate retrievers based on the classification"""

    # Extract classifications from the output
    classification1 = output.get('classification1')
    classification2 = output.get('classification2')

    # Initialize a list to store selected retrievers
    selected_retrievers = []

    # Map the datasources to retrievers
    datasource_map = {
        'vector_store': chunk_retriever,  # Certifique-se de que isso é uma instância de CustomParentDocumentRetriever
        'summary_store': summary_retriever  # Certifique-se de que isso é uma instância de CustomParentDocumentRetriever
    }

    # Helper function to add retrievers based on classification
    def add_retrievers(location, datasources):
        for datasource in datasources:
            retriever = datasource_map.get(datasource)
            if retriever and hasattr(retriever, 'retrieve'):
                selected_retrievers.append((datasource, retriever))

    # Extract and combine the datasources from both classifications
    if classification1:
        datasources = classification1.datasources
    else:
        datasources = []

    if classification2:
        locations = classification2.datasources
    else:
        locations = []

    # Combine the datasources with the specific locations
    for location in locations:
        add_retrievers(location, datasources)

    # Use each selected retriever to get context
    context = {}
    standalone_question = output.get('standalone_question')
    for datasource, retriever in selected_retrievers:
        retrieved_docs = retriever.retrieve(standalone_question)  # Use o método retrieve
        context[f"{datasource}"] = retrieved_docs

    context = remove_sub_docs(context)

    return context
retrieval_chain = RunnablePassthrough.assign(context = route_query)


# ====================================================================
# QA chain
# ====================================================================
class QAFormat(BaseModel):
    """You are a knowledgeable AI assistant specializing in plastic injection mold design. 
                   Provide detailed and accurate information about best practices, design considerations, and troubleshooting tips for plastic injection molding."""
    answer: str = Field(description="Answer to the user question/request")
    image_paths: List[str] = Field(default_factory=list, description="Relevant image paths extracted from the context. Leave empty if no relevant images are found.")

qaformat_functions = [convert_to_openai_function(QAFormat)]

qa_template = """
Você é um especialista em moldes de injeção plástica. Responda à pergunta/solicitação baseando-se apenas no contexto fornecido e no histórico do chat.
Foque em fornecer insights práticos e melhores práticas relacionadas ao design de moldes de injeção plástica.

Use formatação estruturada como pontos ou listas numeradas quando apropriado.

Se houver imagens relevantes mencionadas no contexto, inclua seus caminhos na resposta usando o formato:
[IMAGE_PATH: caminho/para/imagem]

Contexto:
{context}

Histórico do Chat:
{chat_history}

Pergunta:
{question}

Responda detalhadamente e inclua os caminhos das imagens relevantes encontradas no contexto.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("user", qa_template)
])

qa_llm = ChatOpenAI(temperature=0.5, model='gpt-4', openai_api_key=api_key)
#qa_llm = ChatOpenAI(temperature=0.5, model='gpt-4o', openai_api_key=api_key)
qa_llm_structured = llm.bind(
    functions=qaformat_functions,
    function_call={"name": "QAFormat"}
)

# turns nested dict into a flat dict
def format_qa_out(input_dict):
    """
    Formats the dictionary to move 'answer' and 'image_paths' to the top level.
    
    Args:
        data (dict): The input dictionary containing the nested 'initial_answer' dictionary.
    
    Returns:
        dict: The formatted dictionary with 'answer' renamed to 'initial_answer' and moved to the top level,
              along with 'image_paths'.
    """
    # Extract the nested dictionary
    initial_answer_dict = input_dict.pop('initial_answer')
    
    # Update the original dictionary with the extracted keys
    input_dict['initial_answer'] = initial_answer_dict['answer']
    input_dict['image_paths'] = initial_answer_dict['image_paths']
    
    return input_dict

qa_chain = RunnablePassthrough.assign(initial_answer = qa_prompt | qa_llm_structured | JsonOutputFunctionsParser()) | format_qa_out


# ====================================================================
# Evaluation chain
# ====================================================================
class Evaluate(BaseModel):
    """Avalia se a resposta fornecida sobre moldes de injeção é satisfatória"""
    eval_result: str = Field(description="Resultado da avaliação (Y ou N)")

eval_functions = [convert_to_openai_function(Evaluate)]

eval_template = """Você é um avaliador especializado em moldes de injeção plástica, responsável por determinar se a resposta fornecida é satisfatória.
O sistema responde a consultas técnicas sobre design, manufatura e boas práticas em moldes de injeção.
Considere o histórico do chat e a pergunta do usuário ao avaliar a resposta.

Uma resposta satisfatória **não deve** incluir respostas como "Não posso responder" ou "Não tenho informações suficientes." 
Além disso, a resposta deve ser tecnicamente precisa e relevante para moldes de injeção.

Exemplo 1:
User Question:
Quais são os principais tipos de sistemas de refrigeração em moldes?

Answer Provided:
Os principais sistemas de refrigeração em moldes de injeção incluem:
1. Canais convencionais (lineares)
2. Circuitos em série
3. Canais conformais
4. Sistema baffle
5. Sistema bubbler
Cada sistema tem suas aplicações específicas dependendo da geometria da peça e requisitos de resfriamento.

Evaluation: Y

Exemplo 2:
User Question:
Como dimensionar corretamente os canais de alimentação?

Answer Provided:
Desculpe, não tenho informações suficientes sobre dimensionamento de canais de alimentação.

Evaluation: N

Exemplo 3:
User Question:
Mostre exemplos de boas práticas em extração de peças.

Answer Provided:
Aqui estão algumas boas práticas para extração de peças em moldes de injeção:
1. Posicionamento adequado dos extratores
2. Uso de ângulos de saída apropriados
3. Acabamento superficial adequado
[IMAGE_PATH: extratores_exemplo.jpg]

Evaluation: Y

Chat History:
{condensed_history}

User Question:
{question}

Answer Provided:
{initial_answer}

Responda com "Y" se a resposta for satisfatória (tecnicamente precisa e completa) ou "N" se não for adequada.
"""

EVAL_QUESTION_PROMPT = PromptTemplate.from_template(eval_template)

llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', openai_api_key=api_key)
#llm = ChatOpenAI(temperature=0, model='gpt-4o', openai_api_key=api_key)

llm_to_eval = llm.bind(
    functions=eval_functions,
    function_call={"name": "Evaluate"}
)

eval_chain = RunnablePassthrough.assign(eval_result = EVAL_QUESTION_PROMPT | llm_to_eval | JsonKeyOutputFunctionsParser(key_name="eval_result"))


# ====================================================================
# External search chain
# ====================================================================
search = DuckDuckGoSearchRun()
tools = [search]
llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', openai_api_key=api_key)
functions = [convert_to_openai_function(f) for f in tools]
llm_to_search = llm.bind(functions=functions, function_call="auto")

search_template = """Você é um especialista em moldes de injeção plástica.
Se a resposta não puder ser encontrada no contexto fornecido, use a ferramenta DuckDuckGoSearch para encontrar informações técnicas relevantes.

Foque em:
- Práticas recomendadas de design
- Aspectos técnicos de moldes
- Soluções práticas para problemas comuns

Histórico do Chat:
{chat_history}

Pergunta do Usuário:
{question}

Forneça uma resposta técnica e precisa, evitando informações genéricas.
"""

SEARCH_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("user", search_template),
    #("placeholder", "{agent_scratchpad}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

search_chain = SEARCH_QUESTION_PROMPT | llm_to_search | OpenAIFunctionsAgentOutputParser()
agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | search_chain
agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, max_iterations=2)

def external_search_router(eval_output):
    eval_result = eval_output['eval_result']
    logger.info(f"🔄 Router de busca externa - Avaliação: {eval_result}")
    if eval_result == "N":
        logger.info("🌐 Iniciando busca externa...")
        return agent_executor | (lambda x: x["output"])
    else:
        logger.info("✅ Usando resposta inicial...")
        return eval_output['initial_answer']

external_search_chain = RunnableLambda(external_search_router)


# Determine the width based on the number of images
def get_image_width(num_images):
    if num_images == 1:
        return 400
    elif num_images == 2:
        return 300
    else:
        return 200
