import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_graph_metrics(df):
    G = nx.from_pandas_edgelist(df, 'user_id', 'business_id')
    return nx.degree_centrality(G)

def get_text_similarity(texts):
    if len(texts) < 2: return 0.0
    vec = TfidfVectorizer(stop_words='english')
    matrix = vec.fit_transform(texts)
    return cosine_similarity(matrix).mean()