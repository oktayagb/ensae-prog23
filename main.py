import numpy as np
from queue import PriorityQueue
import heapq
import random
import time
from collections import deque
import graphviz


class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.edges = []

        
    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output

    
    def add_edge(self, node1, node2, power_min=1, dist=1):
        if node1 not in self.nodes:
            self.nodes.append(node1)
            self.graph[node1] = []
            self.nb_nodes += 1
        if node2 not in self.nodes:
            self.nodes.append(node2)
            self.graph[node2] = []
            self.nb_nodes += 1
        self.nb_edges += 1
        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        self.edges.append((node1, node2, power_min)) #on rajoute un élément 'edges' qui est une liste de tuple qui contient chaque chemin, avec leur 
        # puissance associée


# On va réaliser un algorithme de parcours en largeur, qui est plus intéressant à utiliser qu'un algorithme de parcours en profondeur car nous n'allons pas avoir
# de problème de maximum recursion
    def connected_components(self):
        visited = set()
        components = []
        for node in self.nodes:
            if node not in visited:
                component = []
                queue = deque([node])  # nous utilisons les structures de piles afin de gérer les différents voisins
                while queue:
                    current = queue.popleft()
                    if current not in visited:  # si l'élément n'a pas été déjà visité, on le marqué comme visité et on rajoute ses voisins dans la pile
                        component.append(current)
                        visited.add(current)
                        queue.extend(neighbour[0] for neighbour in self.graph[current] if neighbour[0] not in visited)
                components.append(component)
        return components  # on renvoie les composantes connexes du graphe
    # Complexité en 0(nb_edges + nb_nodes)

    
    
    def connected_components_set(self):
        return set(map(frozenset,
                       self.connected_components()))  # on fait juste un frozenset de la fonction connected_components


    
 # De la même manière que précédemment, nous allons avoir recourt à un algorithme de parcours en largeur afin de résoudre notre problème, de type Djikstra.
    def get_path_with_power(self, src, dest, power):
        queue = deque([(src, [src])])  # on met en place une pile afin de gérer les noeuds. L'intérêt de la pile est que l'on garde l'odre d'arrivée
        visited = set([src])
        while queue:
            node, path = queue.popleft()
            if node == dest:
                return path  # dès que nous sommes arrivés à destination, nous renvoyons le parcours qui nous a permis
            for neighbor, neighbor_power, dist in self.graph[node]:  # on regarde chaque voisin du point
                if neighbor not in visited and neighbor_power <= power:  # on vérifie que le voisin a une puissance <= à notre puissance pour continuer sur cette branche
                    visited.add(neighbor)  # on le marque comme visité
                    queue.append((neighbor, path + [neighbor]))  # on rajoute à la pile ce noeud et on actualise le parcours en rajoutant le noeud
        return None  # on renvoie None seulement si après avoir tout parcouru on ne trouve pas de chemin de puissance minimal, ou que les éléments sont dans 2 compo connexes
    # La complexité est en O(nb_edges)

    
    
    # Nous allons utiliser notre fonction get_path_with_power avec une recherche par dichotomie afin de trouver la puissance minimale.
    def min_power(self, src, dest):
        test = False
        for compo in self.connected_components_set():
            if src in list(compo) and dest in list(compo):
                test = True  # on vérifie tout d'abord que les deux éléments sont dans la même composante connexe
        if test == False:
            return None
        elif test == True:
            if src == dest:
                return ([src, dest], 1)
            else:
                puis = []
                for noeud in self.graph:
                    for traj in self.graph[noeud]:
                        puis.append(traj[1])
                puis.sort()  # on récupère la liste des puissances de tous les trajets, que l'on trie par ordre croissant
                while len(puis) > 2:
                    if len(puis) % 2 == 0:
                        mid = (len(puis) // 2)
                    else:
                        mid = (len(puis) // 2) + 1
                    power_min = puis[mid - 1]
                    if self.get_path_with_power(src, dest, power_min) == None:
                        del puis[:mid]
                        if len(puis) % 2 == 0:
                            mid = (len(puis) // 2)
                        else:
                            mid = (len(puis) // 2) + 1
                        power_min = puis[mid - 1]
                    elif self.get_path_with_power(src, dest, power_min) != None:
                        del puis[mid:]
                        if len(puis) % 2 == 0:
                            mid = (len(puis) // 2)
                        else:
                            mid = (len(puis) // 2) + 1
                        power_min = puis[mid - 1]
                if len(puis) >= 2:  # on réalise la dichotomie en fonction de si la puissance donnée permet de réaliser le chemin.
                    return (self.get_path_with_power(src, dest, puis[1]), puis[1])
                elif len(puis) == 1:
                    return (self.get_path_with_power(src, dest, puis[0]), puis[0])
# La complexité est de 0((nb_nodes+nb_egdes)*ln(nb_edges))


    def kruskal(self):
        edges = self.edges
        edges.sort(key=lambda x: x[2])  # on trie les chemins par ordre croissant de puissance
        parents = {node: node for node in self.nodes}  # on initialise les parents de chaque noeud.
        rang = {node: 0 for node in self.nodes} #on itinialise le rang de chaque noeud
        def find(node):
            if parents[node] != node:
                parents[node] = find(parents[node])
            return parents[node]  # la fonction find est une fonction récursive qui permet de renvoyer le parent d'un noeud
        def union(x, y):
            x_racine = find(x)
            y_racine = find(y)
            if x_racine != y_racine:
                if rang[x_racine] < rang[y_racine]:
                    parents[x_racine] = y_racine
                else:
                    parents[y_racine] = x_racine
                    if rang[x_racine] == rang[y_racine]:
                        rang[x_racine] = rang[x_racine] + 1 #la fonction union permet de lier les noeuds entre eux, l'utilisation de 'racine' 
                        # nous permet de gagner en efficacité
        index = 0
        tree = Graph(self.nodes)  # on crée l'arbre que l'on va renvoyer.
        while tree.nb_edges != self.nb_nodes - 1:
            src, dest, power = edges[index]
            index += 1
            if find(src) != find(dest):
                tree.add_edge(src, dest, power)
                union(src, dest)
        return tree #on rajoute les arêtes si elles n'ont pas le même parent. Dans ce cas on les relie dans la structure union.


    def find_path(self, src, dest,pre_process): #on réalise cette fonction afin d'optimiser le temps de recherche d'un chemin dans un arbre connexe.
        root=pre_process[0] #on appelle notre pré-processing 
        src_chemin=[src]
        dest_chemin=[dest]
        profondeur, fathers = pre_process[1],pre_process[2]
        src_ligne=profondeur[src]
        dest_ligne=profondeur[dest]
        if src_ligne <= dest_ligne: #on récupère la profondeur des deux noeuds dans l'arbre, et on remonte dans l'arbre jusqu'à trouver le premier 
            # ancêtre commun aux deux noeuds. De plus, on garde en tête le chemin parcouru par chacun des noeuds.
            while src_ligne < dest_ligne:
                dest=fathers[dest]
                dest_ligne=profondeur[dest]
                dest_chemin.append(dest)
            while fathers[dest]!=fathers[src]:
                dest = fathers[dest]
                dest_ligne = profondeur[dest]
                dest_chemin.append(dest)
                src = fathers[src]
                src_ligne = profondeur[src]
                src_chemin.append(src)
            src_chemin.pop()
            return src_chemin + dest_chemin[::-1] # on concatène les deux chemins, afin d'avoir, de manière optimale, le (seul) chemin reliant nos deux noeuds.
        elif src_ligne >= dest_ligne:
            while src_ligne > dest_ligne:
                src = fathers[src]
                src_ligne = profondeur[src]
                src_chemin.append(src)
            while fathers[dest] != fathers[src]:
                dest = fathers[dest]
                dest_ligne = profondeur[dest]
                dest_chemin.append(dest)
                src = fathers[src]
                src_ligne = profondeur[src]
                src_chemin.append(src)
            src_chemin.pop()
            return src_chemin + dest_chemin[::-1]

    def min_power_chemin(self, chemin):  # minimum de puissance nécessaire pour pouvoir faire un trajet donnée
        puissance = 0 #comme nous travaillons sur un arbre, le chemin est unique, donc il suffit juste de parcourir chaque arête de notre chemin,
        # et de garder la plus grande puissance.
        for i in range(len(chemin) - 1):
            depart = self.graph[chemin[i]]
            for voisin in depart :
                if voisin[0] == chemin[i + 1]:
                    if voisin[1] > puissance:
                        puissance = voisin[1]
                    break  # on sort de la boucle sur les voisins dès qu'on a trouvé le bon voisin
        return puissance

    def min_power_tree(self, src, dest, pre_process): # on renvoie notre chemin le plus court; ainsi que sa puissance.
        chemin = self.find_path(src, dest, pre_process)
        return chemin, self.min_power_chemin(chemin)
    # Pour les network.x.in on a qu'une seule composante connexe donc pas besoin d'utiliser les composantes connexes.
    # Comme le chemin est unique, on en determine un peu importe sa puissance et on determine le minimum de puissance nécessaire pour faire ce trajet

def graph_from_file(filename):
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())
        g = Graph(range(1, n + 1))
        for _ in range(m):
            edge = list(map(int, file.readline().split()))
            if len(edge) == 3:
                node1, node2, power_min = edge
                g.add_edge(node1, node2, power_min)
            elif len(edge) == 4:
                node1, node2, power_min, dist = edge
                g.add_edge(node1, node2, power_min, dist)
            else:
                raise Exception("Format incorrect")
    return g


#liste des network, des routes.in et routes.out
network1 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.1.in"
network2 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.2.in"
network3 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.3.in"
network4 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.4.in"
network5 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.5.in"
network6 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.6.in"
network7 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.7.in"
network8 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.8.in"
network9 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.9.in"
network10 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/network.10.in"

route1 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.1.in"
route2 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.2.in"
route3 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.3.in"
route4 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.4.in"
route5 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.5.in"
route6 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.6.in"
route7 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.7.in"
route8 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.8.in"
route9 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.9.in"
route10 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.10.in"

out1 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.1.out"
out2 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.2.out"
out3 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.3.out"
out4 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.4.out"
out5 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.5.out"
out6 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.6.out"
out7 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.7.out"
out8 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.8.out"
out9 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.9.out"
out10 = "/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/routes.10.out"


def draw_graph(graphe,chemin):
    dot = graphviz.Graph() #on crée un graphique à l'aide de la bibliothèque graphviz
    for u, v,w in graphe.edges:
        dot.edge(str(u), str(v))
    for u, v in zip(chemin, chemin[1:]):
        dot.edge(str(u), str(v), color='red') # on ajoute les arêtes que l'on souhaite colorier pour les mettre en évidence.
    return dot.render( format='png')


def question_10(route, network,N):  # on cherche à estimer grossièrement le temps de calcul de plusieurs graphes. L'on remarque que ce temps est considérable.
    g = graph_from_file(network)
    with open(route, "r") as file:
        n = int(file.readline().split()[0])
        start = time.time()
        for i in range(N):
            edge = list(map(int, file.readline().split()))
            dep, arr, utilite = edge
            g.min_power(dep, arr,pre_process)  # on calcule la puissance minimale pour chaque trajet de routes.x.in
        end = time.time()
        elapsed = end - start
        total = (elapsed / N) * n
        return f' Temps total : {total:.5}s'  # on renvoie le temps total
# Nous trouvons pour de l'ordre de 10^6s pour les network.x.in, avec x>1.


# Question 15

def question_15(route, network, out):  # on cherche à estimer le temps de calcul de plusieurs graphes après transformations par Kruskal
    g = graph_from_file(network)
    s = g.kruskal()  # on réalise la transformation de Kruskal
    with open(route, "r") as file:
        n = int(file.readline().split()[0])
        start = time.time()
        fichier = open(out, "a")  # on va stocker les valeurs dans un nouveau dossier route.x.out
        for i in range(n):
            edge = list(map(int, file.readline().split()))
            dep, arr, utilite = edge
            a = s.min_power_tree(dep, arr,pre_process)[1]  # on calcule la puissance minimale pour chaque trajet de routes.x.in
            fichier.write(str(a))
            fichier.write("\n")
        end = time.time()
        total = end - start
        fichier.close()
        return f' Temps total : {total:.5}s'  # on renvoie le temps total
#On remarque que le temps a considérablement diminué, notre fonction crée et rempli notre dossier routes.x.out entre 1sec et 120sec.


def dfs(graph, start, profondeur,fathers,visited=None,index=0): #nous réalisons un dfs de notre graphe afin de récupérer, pour chaque arête 
    # sa profondeur dans le graphe, et son père.
    if visited is None:
        visited = set()  # ensemble de sommets visités
    visited.add(start)  # marquer le sommet comme visité
    for neighbor in graph[start]:
            if neighbor[0] not in visited:  # parcourir les voisins non visités
                index+=1
                profondeur[neighbor[0]] += index #on augmente la profondeur à chaque fois que l'on descend dans le graphe
                fathers[neighbor[0]] = start #on récupère le père de chaque noeud
                dfs(graph, neighbor[0], profondeur,fathers,visited,index)  # appel récursif pour visiter chaque voisin non visité
            else:
                pass
    return profondeur,fathers


#Nous réalisons une étape de pré-processing afin d'améliorer considérablement la vitesse de nos algorithmes. Nous récupérons dans celui-ci : 
# la profondeur de chaque noeud, et le père de chaque noeud.
g = graph_from_file(network9)
s = g.kruskal()
root=s.nodes[0]
prof = {nodes: 0 for nodes in s.nodes}
dads = {nodes: 0 for nodes in s.nodes}
dads[root] = root
profondeur, fathers = dfs(s.graph,root,prof,dads)
pre_process=(root,profondeur,fathers,prof,dads)
