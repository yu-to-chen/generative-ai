# utils.py  (updated for new TruLens APIs)

# !pip install python-dotenv

import os
import warnings
from dotenv import load_dotenv, find_dotenv

import numpy as np
import nest_asyncio

# ---- TruLens (NEW import locations) ----
from trulens.core import Feedback                         # was: from trulens_eval import Feedback
from trulens.providers.openai import OpenAI               # was: from trulens_eval import OpenAI
from trulens.apps.llamaindex import TruLlama              # was: from trulens_eval import TruLlama
# Optional helpers/aggregators:
from trulens.core.feedback import feedback as fb          # for groundedness aggregators/configs

nest_asyncio.apply()

# Optional: hush munch/pkg_resources deprecation warnings if they’re noisy
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


# -------------------- Keys --------------------
def get_openai_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv("HUGGINGFACE_API_KEY")


# Provider instance (reads OPENAI_API_KEY)
openai = OpenAI()

# -------------------- Feedbacks --------------------
# 1) Answer relevance: compare input↔output
qa_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
      .on_input_output()
)

# 2) Context relevance: how relevant are retrieved source nodes to the prompt?
qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Context Relevance")
      .on_input()
      .on(TruLlama.select_source_nodes().node.text)
      .aggregate(np.mean)
)

# 3) Groundedness: is the answer supported by cited context?
#    New API: call provider.groundedness_measure_with_cot_reasons directly.
#    (Optional) You can pass configs via fb.GroundednessConfigs(...)
groundedness = (
    Feedback(openai.groundedness_measure_with_cot_reasons, name="Groundedness")
      .on(TruLlama.select_source_nodes().node.text)   # context chunks
      .on_output()                                    # model answer
      # Optionally aggregate per-statement groundedness into a single score:
      .aggregate(np.mean)
      #.aggregate(fb.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance, groundedness]


# -------------------- TruLlama recorders --------------------
def get_trulens_recorder(query_engine, feedbacks, app_id):
    return TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )


def get_prebuilt_trulens_recorder(query_engine, app_id):
    return TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )


# -------------------- LlamaIndex utilities --------------------
# OLD
# from llama_index import ServiceContext, VectorStoreIndex, StorageContext

# NEW
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import ServiceContext

# OLD
# from llama_index.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
# from llama_index.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
# from llama_index import load_index_from_storage
# from llama_index.retrievers import AutoMergingRetriever
# from llama_index.query_engine import RetrieverQueryEngine

# NEW
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


def build_sentence_window_index(
    document, llm, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="sentence_index"
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document], service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine

