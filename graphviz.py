import pydot 

with open('src/tree.dot', 'r') as f: graph = f.read()
(graph,) = pydot.graph_from_dot_data(graph) 
graph.write_png('tree.png') 