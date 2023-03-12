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
            # On va réaliser un algorithme de parcours en largeur, qui est plus intéressant à utiliser qu'un algorithme de parcours en profondeur car nous n'allons pas avoir
# de problème de maximum recursion
        visited = set()
        components = []
        for node in self.nodes:
            if node not in visited:
                component = []
                queue = deque([node]) #nous utilisons les structures de piles afin de gérer les différents voisins
                while queue:
                    current = queue.popleft()
                    if current not in visited: #si l'élément n'a pas été déjà visité, on le marqué comme visité et on rajoute ses voisins dans la pile
                        component.append(current)
                        visited.add(current)
                        queue.extend(neighbour[0] for neighbour in self.graph[current] if neighbour[0] not in visited)
                components.append(component)
        return components #on renvoie les composantes connexes du graphe
# Complexité en 0(nb_edges + nb_nodes)
        
        
        
   # def connected_components(self):
       # val=[]
    #    for node in self.nodes:
     #       def recursive(graph, node, visited=None):
      #          if visited == None:
       #             visited = []
        #        if node not in visited:
         #           visited.append(node)
          #      unvisited = [k[0] for k in graph[node] if k[0] not in visited]
           #     for node in unvisited:
            #        recursive(graph, node, visited)
             #   return visited
            #val.append(recursive(self.graph,node))
        #return val



    def connected_components_set(self):
        return set(map(frozenset, self.connected_components()))


    def get_path_with_power(self,src,dest,power):
        #De la même manière que précédemment, nous allons avoir recourt à un algorithme de parcours en largeur afin de résoudre notre problème, de type Djikstra.
        queue = deque([(src, [src])]) #on met en place une pile afin de gérer les noeuds. L'intérêt de la pile est que l'on garde l'odre d'arrivée
        visited = set([src])
        while queue:
            node, path = queue.popleft()
            if node == dest:
                return path #dès que nous sommes arrivés à destination, nous renvoyons le parcours qui nous a permis
            for neighbor, neighbor_power, dist in self.graph[node]: #on regarde chaque voisin du point
                if neighbor not in visited and neighbor_power <= power: #on vérifie que le voisin a une puissance <= à notre puissance pour continuer sur cette branche
                    visited.add(neighbor) #on le marque comme visité
                    queue.append((neighbor, path + [neighbor])) #on rajoute à la pile ce noeud et on actualise le parcours en rajoutant le noeud
        return None #on renvoie None seulement si après avoir tout parcouru on ne trouve pas de chemin de puissance minimal, ou que les éléments sont dans 2 compo connexes
#La complexité est en O(nb_edges)
    
    #def get_path_with_power(self, src, dest, power):
     #   def recursive(graph, node, chemin, visited=[]):
      #      if node==dest:
       #        return chemin
        #    if node not in visited:
         #       visited.append(node)
          #      unvisited = [k[0] for k in graph[node] if k[0] not in visited and k[1]<=power]
           #     for node in unvisited:
            #        result=recursive(graph,node,chemin+[node],visited)
             #       if result is not None:
           #             return result
           # return None
        #return recursive(self.graph, src,[src])


    #Nous allons utiliser notre fonction get_path_with_power avec une recherche par dichotomie afin de trouver la puissance minimale.
    def min_power(self, src, dest):
        test=False
        for compo in self.connected_components_set():
            if src  in list(compo) and dest in list(compo):
                test=True #on vérifie tout d'abord que les deux éléments sont dans la même composante connexe
        if test==False:
            return None
        elif test==True:
            puis = []
            for noeud in self.graph:
                for traj in self.graph[noeud]:
                    puis.append(traj[1])
            puis.sort() #on récupère la liste des puissances de tous les trajets, que l'on trie par ordre croissant
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
            if len(puis)>=2: #on réalise la dichotomie en fonction de si la puissance donnée permet de réaliser le chemin.
                return (self.get_path_with_power(src,dest,puis[1]),puis[1],)
            elif len(puis)==1:
                return (self.get_path_with_power(src,dest,puis[0]),puis[0])
#La complexité est de 0((nb_nodes+nb_egdes)*ln(nb_edges))
            
    
# Nous allons implémenter l'algorithme de Kruskal
    def kruskal(self):
        edges = []
        for node in self.graph:
            for neighbor in self.graph[node]:
                edges.append((node, neighbor[0], neighbor[1]))
        edges.sort(key=lambda x: x[2]) #on trie les chemins par ordre croissant de puissance
        parents = {node: node for node in self.nodes} #on initialise les parents de chaque noeud.
        def find(node):
            if parents[node] != node:
                parents[node] = find(parents[node])
            return parents[node] #la fonction find est une fonction récursive qui permet de renvoyer le parent d'un noeud
        tree = Graph() #on crée l'arbre que l'on va renvoyer.
        for edge in edges:
            parent1 = find(edge[0])
            parent2 = find(edge[1]) #pour chaque chemin, on détermine le parent du départ et de l'arrivée
            if parent1 != parent2:
                tree.add_edge(edge[0], edge[1], power_min=edge[2])
                parents[parent1] = parent2 #si les parents sont différents, on peut ajouter ce chemin à notre arbre, s'assurant donc bien qu'il n'y aura pas de
                # boucles dans notre nouvel arbre
        return tree      
        
        
#filename ="/Users/oktay/OneDrive/Bureau/ENSAE/S2/projet info/ensae-prog23-main/ensae-prog23-main/input/network.5.in"
#file="/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.01.in"
filename="/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.2.in"

def graph_from_file(filename):
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


filename1="/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.1.in"
filename2="/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.2.in"
filename3="/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.3.in"
file=[filename1,filename2,filename3]
def question_10(g): # on cherche à estimer grossièrement le temps de calcul de plusieurs graphes. L'on remarque que ce temps est considérable.
    #Nombre de trajets possibles = nb_nodes**2
    start = time.time()
    N=5
    for i in range(N):
        dep=random.randint(1,g.nb_nodes)
        fin=random.randint(1,g.nb_nodes)
        g.min_power(dep,fin)

    end = time.time()
    elapsed = end - start
    moyen = elapsed / N
    total = moyen * (g.nb_nodes**2)

    print(f'Temps d\'exécution : {elapsed:.2}ms')
    print(f' Temps total : {total:.2}ms')
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







