import pydot 
import networkx
import matplotlib.pyplot as plt

# Open with pydot.
with open('src/tree.dot', 'r') as f: graph_spec = f.read()
(graph_dot,) = pydot.graph_from_dot_data(graph_spec)
graph_dot.write_png('tree.png') 

# Convert to networkx
graph = networkx.drawing.nx_pydot.from_pydot(graph_dot)
# for node in graph.nodes:
#     print(graph.nodes[node])
pos = networkx.drawing.nx_pydot.graphviz_layout(graph)
networkx.draw(graph, pos)
plt.show()