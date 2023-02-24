class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output

    def add_edge(self, node1, node2, power_min):
        if node1 not in self.nodes:
            self.nodes.append(node1)
            self.graph[node1]=[]
        if node2 not in self.nodes:
            self.nodes.append(node2)
            self.graph[node2]=[]
        self.nb_edges += 1
        self.graph[node1].append((node2,power_min))
        self.graph[node2].append((node1,power_min))



    def connected_components(self):
        val=[]
        for node in self.nodes:
            def recursive(graph, node, visited=None):
                if visited == None:
                    visited = []
                if node not in visited:
                    visited.append(node)
                unvisited = [k[0] for k in graph[node] if k[0] not in visited]
                for node in unvisited:
                    recursive(graph, node, visited)
                return visited
            val.append(recursive(self.graph,node))
        return val



    def connected_components_set(self):
        return set(map(frozenset, self.connected_components()))


    def get_path_with_power(self, src, dest, power):
        return




    def min_power(self, src, dest):
        """
        Should return path, min_power.
        """
        raise NotImplementedError



file="/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.01.in"
def graph_from_file(file):
    f=open(file)
    graphique=Graph()
    nb=f.readline().split()
    for i in range(1,int(nb[1])+1):
        val=f.readline().split()
        if len(val)==3:
            graphique.add_edge(int(val[0]),int(val[1]),int(val[2]))
        elif len(val)==2:
            val.append(1)
            graphique.add_edge(int(val[0]),int(val[1]),int(val[2]))
    return graphique


g=graph_from_file(file)





