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



    def find_all_paths(self, src, dest): #on cherche à obtenir une liste de listes qui contient tous les chemins possibles
        # pour rejoindre src de dest
        visited = set()
        paths = []
        current_path = [src]

        def dfs(node): #on réalise un dfs pour cela
            visited.add(node)
            if node == dest:
                paths.append(current_path.copy()) #si on est arrivé à la destination, on rajoute ce chemin à notre liste de chemins
            else:
                for neighbor in self.graph[node]:
                    if neighbor[0] not in visited:
                        current_path.append(neighbor[0]) #on rajoute le noeud au trajet
                        dfs(neighbor[0]) #procédé récursif sur chaque noeud
                        current_path.pop()
                visited.remove(node)

        dfs(src)
        return paths

    def min_dist_chemin(self,src,dest):
        all_path = self.find_all_paths(src,dest)
        min_d = np.inf
        min_path = None
        for path in all_path:
            dist = 0
            for k in range(len(path) - 1):
                for (n, p, d) in self.graph[path[k]]:
                    if n == path[k + 1]:
                        dist += d
                        if dist <= min_d:
                            min_d = dist
                            min_path = path
        return min_path,min_d





#########(remarque du prof)on met trop de vas particulier pas besoin de pre_process et pas besoin de faire le egal et elif return fin_path(dest,src)
    def find_path(self, src, dest,profondeur,fathers): #on réalise cette fonction afin d'optimiser le temps de recherche d'un chemin dans un arbre connexe.
        src_chemin=[src]
        dest_chemin=[dest]
        src_ligne=profondeur[src]
        dest_ligne=profondeur[dest]
        if src_ligne < dest_ligne: #on récupère la profondeur des deux noeuds dans l'arbre, et on remonte dans l'arbre jusqu'à trouver le premier 
            # ancêtre commun aux deux noeuds. De plus, on garde en tête le chemin parcouru par chacun des noeuds.
            while src_ligne < dest_ligne:
                dest=fathers[dest]
                dest_ligne=profondeur[dest]
                dest_chemin.append(dest)
            if dest==src:
                return dest_chemin[::-1]
            while fathers[dest]!=fathers[src]:
                dest = fathers[dest]
                dest_ligne = profondeur[dest]
                dest_chemin.append(dest)
                src = fathers[src]
                src_ligne = profondeur[src]
                src_chemin.append(src)
            return src_chemin +[fathers[dest]]+ dest_chemin[::-1] # on concatène les deux chemins, afin d'avoir, de manière optimale, le (seul) chemin reliant nos deux noeuds.
        elif src_ligne > dest_ligne:
            while src_ligne > dest_ligne:
                src = fathers[src]
                src_ligne = profondeur[src]
                src_chemin.append(src)
            if dest==src:
                return src_chemin
            while fathers[dest] != fathers[src]:
                dest = fathers[dest]
                dest_ligne = profondeur[dest]
                dest_chemin.append(dest)
                src = fathers[src]
                src_ligne = profondeur[src]
                src_chemin.append(src)
            return src_chemin +[fathers[dest]]+ dest_chemin[::-1]
        else:
            if fathers[dest]==fathers[src]:
                return [src,fathers[src], dest]
            while fathers[dest] != fathers[src]:
                dest = fathers[dest]
                dest_ligne = profondeur[dest]
                dest_chemin.append(dest)
                src = fathers[src]
                src_ligne = profondeur[src]
                src_chemin.append(src)
            return src_chemin +[fathers[dest]]+ dest_chemin[::-1]

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

    def min_power_tree(self, src, dest, root,profondeur,fathers): # on renvoie notre chemin le plus court; ainsi que sa puissance.
        chemin = self.find_path(src, dest, root,profondeur,fathers)
        return chemin, self.min_power_chemin(chemin)
    # Pour les network.x.in on a qu'une seule composante connexe donc pas besoin d'utiliser les composantes connexes.
    # Comme le chemin est unique, on en determine un peu importe sa puissance et on determine le minimum de puissance nécessaire pour faire ce trajet

def graph_from_file(filename):
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())
        g = Graph(range(1, n + 1))
        for _ in range(m):
            edge = list(map(float, file.readline().split()))
            if len(edge) == 3:
                node1, node2, power_min = int(edge[0]),int(edge[1]),edge[2]
                g.add_edge(node1, node2, power_min)
            elif len(edge) == 4:
                node1, node2, power_min, dist = int(edge[0]),int(edge[1]),edge[2],edge[3]
                g.add_edge(node1, node2, power_min, dist)
            else:
                raise Exception("Format incorrect")
    return g





def kruskal(graph):
        edges = graph.edges
        edges.sort(key=lambda x: x[2])  # on trie les chemins par ordre croissant de puissance
        parents = {node: node for node in graph.nodes}  # on initialise les parents de chaque noeud.
        rang = {node: 0 for node in graph.nodes} #on itinialise le rang de chaque noeud
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
        tree = Graph(graph.nodes)  # on crée l'arbre que l'on va renvoyer.
        while tree.nb_edges != graph.nb_nodes - 1:
            src, dest, power = edges[index]
            index += 1
            if find(src) != find(dest):
                tree.add_edge(src, dest, power)
                union(src, dest)
        return tree #on rajoute les arêtes si elles n'ont pas le même parent. Dans ce cas on les relie dans la structure union.


#liste des network, des routes.in et routes.out
network_files = [f"/network.{i}.in" for i in range(1, 11)]
route_files = [f"/routes.{i}.in" for i in range(1, 11)]
out_files = [f"/routes.{i}.out" for i in range(1, 11)]



def draw_graph(graphe,chemin):
    dot = graphviz.Graph() #on crée un graphique à l'aide de la bibliothèque graphviz
    for u, v,w in graphe.edges:
        dot.edge(str(u), str(v))
    for u, v in zip(chemin, chemin[1:]):
        dot.edge(str(u), str(v), color='red') # on ajoute les arêtes que l'on souhaite colorier pour les mettre en évidence.
    return dot.render( format='png')


def question_10(index,N):  # on cherche à estimer grossièrement le temps de calcul de plusieurs graphes. L'on remarque que ce temps est considérable.
    g = graph_from_file(network_files[index-1])
    with open(route_files[index-1], "r") as file:
        n = int(file.readline().split()[0])
        start = time.time()
        for i in range(N):
            edge = list(map(float, file.readline().split()))
            dep, arr, utilite = int(edge[0]),int(edge[1]), edge[2]
            g.min_power(dep, arr)  # on calcule la puissance minimale pour chaque trajet de routes.x.in
        end = time.time()
        elapsed = end - start
        total = (elapsed / N) * n
        return f' Temps total : {total:.5}s'  # on renvoie le temps total
# Nous trouvons pour de l'ordre de 10^6s pour les network.x.in, avec x>1.


# Question 15

def question_15(index,profondeur,fathers,s):  # on cherche à estimer le temps de calcul de plusieurs graphes après transformations par Kruskal
    with open(route_files[index-1], "r") as file:
        n = int(file.readline().split()[0])
        start = time.time()
        fichier = open(out_files[index-1], "a")  # on va stocker les valeurs dans un nouveau dossier route.x.out
        for i in range(n):
            edge = list(map(float, file.readline().split()))
            dep, arr, utilite = int(edge[0]),int(edge[1]),edge[2]
            a = s.min_power_tree(dep, arr,root,profondeur,fathers)[1]  # on calcule la puissance minimale pour chaque trajet de routes.x.in
            fichier.write(str(a))
            fichier.write("\n")
        end = time.time()
        total = end - start
        fichier.close()
        return f' Temps total : {total:.5}s'  # on renvoie le temps total
#On remarque que le temps a considérablement diminué, notre fonction crée et rempli notre dossier routes.x.out entre 1sec et 120sec.


def dfs(graph, start, prof,dads,visited=None,index=0): #nous réalisons un dfs de notre graphe afin de récupérer, pour chaque arête 
    # sa profondeur dans le graphe, et son père.
    if visited is None:
        visited = set()  # ensemble de sommets visités
    visited.add(start)  # marquer le sommet comme visité
    index+=1
    for neighbor in graph[start]:
            if neighbor[0] not in visited:  # parcourir les voisins non visités
                prof[neighbor[0]] += index #on augmente la profondeur à chaque fois que l'on descend dans le graphe
                dads[neighbor[0]] = start #on récupère le père de chaque noeud
                dfs(graph, neighbor[0], prof,dads,visited,index)  # appel récursif pour visiter chaque voisin non visité
            else:
                pass
    return prof,dads


#Nous réalisons une étape de pré-processing afin d'améliorer considérablement la vitesse de nos algorithmes. Nous récupérons dans celui-ci : 
# la profondeur de chaque noeud, et le père de chaque noeud.
def pre_process(index): #fonction pré-process
    g = graph_from_file(network_files[index-1])
    s = kruskal(g)
    root = s.nodes[0]
    prof = {nodes: 0 for nodes in s.nodes}
    dads = {nodes: 0 for nodes in s.nodes}
    dads[root] = root
    profondeur, fathers = dfs(s.graph, root, prof, dads)
    return profondeur,fathers,s

profondeur, fathers,s = pre_process() #-> manière d'appeler le pré-process


trucks_files = [f"/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/trucks.{i}.in" for i in range(0, 3)]
opti_files = [f"/Users/adrien/Desktop/ENSAE/M1/Cours ENSAE S1/Info/projetS2/input/trucks.{i}.out" for i in range(1, 11)]


#1ere possibilité (force brute)
def maximisation(index_route,index_truck,B):
    list_puissance=[]
    list_dest=[]
    modele=[]
    voyage=[]
    with open(route_files[index_route-1],'r') as file:
        n = int(file.readline().split()[0])
        for i in range(n):
            edge = list(map(float, file.readline().split()))
            dep, arr, utilite = int(edge[0]),int(edge[1]),edge[2]
            list_dest.append((dep,arr,utilite))
    with open(out_files[index_route-1], "r") as file2:
        for i in range(1,n+1):
            puis= float(file2.readline().split()[0])
            list_puissance.append(puis)
    with open(trucks_files[index_truck-1], 'r') as file3:
        nb_modele = int(file3.readline().split()[0])
        for i in range(nb_modele):
            val = list(map(float, file3.readline().split()))
            puis_cam,cout_cam = val[0],val[1]
            modele.append((puis_cam,cout_cam,i))
    modele.sort(key=lambda x:x[1]) #on récupère une liste de tous les modèles de camions (indexé par un entier), triée par ordre croissante de coût
    for i in range(len(list_puissance)):
        voyage.append((list_dest[i][0],list_dest[i][1],list_dest[i][2],list_puissance[i]))  #on recupère une liste avec tous les trajets ainsi que leur utilité et
        #la puissance nécessaire pour faire le trajet

    camion_pour_trajet=[]
    for trajet in voyage:
        for camion in modele:
            if trajet[3]<=camion[0]:
                camion_pour_trajet.append((trajet,camion)) #On associe à chaque trajet le camion le moins cher qui peut réaliser le trajet
                break #on sort de la boucle for lorsque l'on a trouvé le camion le moins cher

    N=len(camion_pour_trajet)
    camion_pour_trajet.sort(key=lambda x:(x[0][2]/x[1][1]),reverse=True) #on trie la nouvelle liste par efficacité (utilité du trajet / prix du camion optimal)
    # décroissante
    S=0
    profit=0
    dernier=0
    for i in range(N):
        if S+camion_pour_trajet[i][1][1]<B: #tant que la contrainte n'est pas saturée, on prend les trajets
            S+=camion_pour_trajet[i][1][1]
            profit+=camion_pour_trajet[i][0][2]
            dernier+=1
    return (camion_pour_trajet[:dernier],profit,B-S) #on renvoie la collection de camions et trajets choisis, ainsi que le profit et le budget restant


def visualisation(index_route,index_truck_out,index_truck_in,B): #permet de présenter dans un fichier les résultats de la fonction maximisation
    dep_arr_cam=[]
    camion_pour_trajet, profit, reste = maximisation(index_route,index_truck_in,B)
    fichier = open(opti_files[index_truck_out-1], "a")  # on va stocker les valeurs dans un nouveau dossier route.x.out
    fichier.write(str((profit,reste)))
    fichier.write("\n")
    fichier.write("profit, budget restant")
    fichier.write("\n")
    fichier.write("départ, arrivée, numéro du camion")
    fichier.write("\n")
    for i in range(len(camion_pour_trajet)):
        dep_arr_cam.append((camion_pour_trajet[i][0][0],camion_pour_trajet[i][0][1],camion_pour_trajet[i][1][2]))
        fichier.write(str(dep_arr_cam[i]))
        fichier.write("\n")
    fichier.close()

budget = 25*10**9

def draw_allocations(graphe,liste_chemin): #forme de liste_chemin=[(chemin,allocation)]
    dot = graphviz.Graph() #on crée un graphique à l'aide de la bibliothèque graphviz
    for chemin in liste_chemin:
        for u, v in zip(chemin[0], chemin[0][1:]):
            dot.edge(str(u), str(v), color='red',label=str(chemin[1])) # on ajoute les arêtes que l'on souhaite colorier pour les mettre en évidence.
    for u, v,w in graphe.edges:
        dot.edge(str(u), str(v))

    return dot.render( format='png')


def draw_allocations_rouge_bleu(graphe,liste_chemin): #forme de liste_chemin=[(chemin,allocation)]
    dot = graphviz.Graph() #on crée un graphique à l'aide de la bibliothèque graphviz

    for u, v in zip(liste_chemin[0][0], liste_chemin[0][0][1:]):
        dot.edge(str(u), str(v), color='red',label=str(liste_chemin[0][1])) # on ajoute les arêtes que l'on souhaite colorier pour les mettre en évidence.
    for u, v in zip(liste_chemin[1][0], liste_chemin[1][0][1:]):
        dot.edge(str(u), str(v), color='blue',label=str(liste_chemin[1][1])) # on ajoute les arêtes que l'on souhaite colorier pour les mettre en évidence.
    for u, v,w in graphe.edges:
        dot.edge(str(u), str(v))

    return dot.render( format='png')
