import networkx as nx
import json
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph

def visualize_graph(file_path, output_path):
    # Carica il grafo dal file JSON
    with open(file_path) as f:
        data = json.load(f)
        graph = json_graph.node_link_graph(data)

    # Stampa informazioni sui nodi
    print("Nodi del grafo:")
    for node in graph.nodes(data=True):
        print(node)

    # Disegna il grafo
    plt.figure(figsize=(20, 20))
    nx.draw(graph, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')

    # Salva il grafo come immagine
    plt.savefig(output_path)

if __name__ == "__main__":
    file_path = 'inputFilesGraphsage/BLCA2/BLCA2-G0.json'
    output_path = 'output_graph.png'
    visualize_graph(file_path, output_path)