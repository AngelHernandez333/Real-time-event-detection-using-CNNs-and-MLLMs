import networkx as nx
import matplotlib.pyplot as plt

# 1. Definimos la jerarquía como un diccionario
event_hierarchy = {
    "Riding": [],
    "Fighting": [],
    "Playing": [],
    "Running": ["Chasing", "Stealing"],
    "Lying": ["Falling"],
    "Chasing": ["Running"],
    "Jumping": ["Falling"],
    "Falling": [],
    "Guiding": ["Chasing", "Running"],
    "Stealing": ["Running", 'Chasing'],
    "Littering": [],
    "Tripping": ["Falling"],
    "Pickpockering": []
}

# 2. Crear el grafo dirigido
G = nx.DiGraph()

# 3. Añadir nodos y aristas al grafo
for parent, children in event_hierarchy.items():
    for child in children:
        G.add_edge(parent, child)

# 4. Dibujar el grafo
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.8, seed=42)  # Layout con separación

nx.draw(G, pos, with_labels=True, arrows=True, node_color='skyblue', node_size=1500, font_size=10, edge_color='gray')
plt.title("Jerarquía de eventos (grafo dirigido)")
plt.axis('off')
plt.tight_layout()
plt.show()
