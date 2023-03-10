import random
import time

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

    def add_edge(self, node1, node2, power_min=1,dist=1):
        if node1 not in self.nodes:
            self.nodes.append(node1)
            self.graph[node1]=[]
            self.nb_nodes+=1
        if node2 not in self.nodes:
            self.nodes.append(node2)
            self.graph[node2]=[]
            self.nb_nodes+=1
        self.nb_edges += 1
        self.graph[node1].append((node2,power_min,dist))
        self.graph[node2].append((node1,power_min,dist))



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
        def recursive(graph, node, chemin, visited=[]):
            if node==dest:
                return chemin
            if node not in visited:
                visited.append(node)
                unvisited = [k[0] for k in graph[node] if k[0] not in visited and k[1]<=power]
                for node in unvisited:
                    result=recursive(graph,node,chemin+[node],visited)
                    if result is not None:
                        return result
            return None
        return recursive(self.graph, src,[src])


    def min_power(self, src, dest):
        test=False
        for compo in self.connected_components_set():
            if src  in list(compo) and dest in list(compo):
                test=True
        if test==False:
            return None
        elif test==True:
            puis = []
            for noeud in self.graph:
                for traj in self.graph[noeud]:
                    puis.append(traj[1])
            puis.sort()
            while len(puis)>2:
                if len(puis)%2==0:
                    mid=(len(puis)//2)
                else:
                    mid=(len(puis)//2)+1
                power_min=puis[mid-1]
                if self.get_path_with_power(src,dest,power_min)==None:
                    del puis[:mid]
                    if len(puis) % 2 == 0:
                        mid = (len(puis) // 2)
                    else:
                        mid = (len(puis) // 2) + 1
                    power_min = puis[mid-1]
                elif self.get_path_with_power(src,dest,power_min)!=None:
                    del puis[mid:]
                    if len(puis) % 2 == 0:
                        mid = (len(puis) // 2)
                    else:
                        mid = (len(puis) // 2) + 1
                    power_min = puis[mid-1]
            if len(puis)>=2:
                return (self.get_path_with_power(src,dest,puis[1]),puis[1],)
            elif len(puis)==1:
                return (self.get_path_with_power(src,dest,puis[0]),puis[0])
    def kruskal(self):
        edges = []
        for node in self.graph:
            for neighbor in self.graph[node]:
                edges.append((node, neighbor[0], neighbor[1]))

        # Sort edges by increasing weight
        edges.sort(key=lambda x: x[2])

        # Initialize union-find structure
        parents = {node: node for node in self.nodes}

        def find(node):
            if parents[node] != node:
                parents[node] = find(parents[node])
            return parents[node]

        # Loop over edges and add them to the tree
        tree = Graph()
        for edge in edges:
            parent1 = find(edge[0])
            parent2 = find(edge[1])
            if parent1 != parent2:
                tree.add_edge(edge[0], edge[1], power_min=edge[2])
                parents[parent1] = parent2

        return tree          
        
        
#filename ="/Users/oktay/OneDrive/Bureau/ENSAE/S2/projet info/ensae-prog23-main/ensae-prog23-main/input/network.5.in"
#file="/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.01.in"
filename="/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.2.in"

def graph_from_file(file):
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())
        g = Graph(range(1, n+1))
        for _ in range(m):
            edge = list(map(int, file.readline().split()))
            if len(edge) == 3:
                node1, node2, power_min = edge
                g.add_edge(node1, node2, power_min) # will add dist=1 by default
            elif len(edge) == 4:
                node1, node2, power_min, dist = edge
                g.add_edge(node1, node2, power_min, dist)
            else:
                raise Exception("Format incorrect")
    return g


#def graph_from_file(file):
    #f=open(file)
    #graphique=Graph()
    #nb=f.readline().split()
    #if len(nb)==2:
        #for i in range(1,int(nb[1])+1):
            #val=f.readline().split()
            #if len(val)==3:
                #graphique.add_edge(int(val[0]),int(val[1]),int(val[2]))
            #elif len(val)==2:
                #graphique.add_edge(int(val[0]),int(val[1]))
            #elif len(val)==4:
                #graphique.add_edge(int(val[0]),int(val[1]),int(val[2]),int(val[3]))
        #return graphique
    #elif len(nb)==1:
        #"for i in range(1,int(nb[0])+1):
            #val=f.readline().split()
            #if len(val)==3:
                #graphique.add_edge(int(val[0]),int(val[1]),int(val[2]))
            #elif len(val)==2:
                #graphique.add_edge(int(val[0]),int(val[1]))
            #elif len(val)==4:
                #graphique.add_edge(int(val[0]),int(val[1]),int(val[2]),int(val[3]))
        #return graphique


def question_11():
    g=graph_from_file(file)

    #Nombre de trajets possibles = nb_nodes**2
    start = time.time()
    N=1
    for i in range(N):
        dep=random.randint(1,g.nb_nodes)
        fin=random.randint(1,g.nb_nodes)
        g.min_power(dep,fin)

    end = time.time()
    elapsed = end - start
    moyen = elapsed / N
    total = moyen * (g.nb_nodes**2)

    print(f'Temps d\'exécution : {elapsed:.2}ms')
    print(f' Temps total : {moyen:.2}ms')
    return

import graphviz

def draw_graph(graph):
    dot = graphviz.Graph(format='png')
    dot.attr(rankdir='LR')
    for node in graph.nodes:
        dot.node(str(node))
        for neighbor in graph.graph[node]:
            dot.edge(str(node), str(neighbor[0]), label=f"Power={neighbor[1]}, Dist={neighbor[2]}")
    dot.render(view=True)
    
#A reprendre à partir de là
class EnsembleDisjoint:
    parent={}
    def __init__(self,N):
        for i in range(1,N+1):
            self.parent[i]=i

    def get_parent(self,k):
        if self.parent[k]==k:
            return k
        return self.get_parent(self.parent[k])

    def Union(self,a,b):
        x = self.get_parent(a)
        y = self.get_parent(b)

        self.parent[x] = y

def Krustal(arcs, nb_sommets):
    Arbre_minimal=[]
    ed = EnsembleDisjoint(nb_sommets)
    index=0

    while len(Arbre_minimal)!=nb_sommets-1:
        src,dest,weight=arcs[index]
        index += 1

        x = ed.get_parent(src)
        y = ed.get_parent(dest)

        if x!=y:
            Arbre_minimal.append((src,dest,weight))
            ed.Union(x, y)

    return Arbre_minimal

g=graph_from_file(file)
graph=g.graph

arcs=[]
for dep in graph:
    for dest in graph[dep]:
        arr,puissance =dest[0],dest[1]
        arcs.append((dep,arr,puissance))

arcs.sort(key=lambda x:x[2])
nb_sommets=g.nb_nodes

a=Krustal(arcs,nb_sommets)

g_min=Graph()
for val in a :
    g_min.add_edge(val[0],val[1],val[2])
print(g_min)







