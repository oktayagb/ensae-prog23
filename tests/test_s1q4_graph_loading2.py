import sys
sys.path.append("ensae-prog23-main/")###peut etre qu'il faudra changer ici mais sur mon ordi ca marche

import unittest
from main import Graph, graph_from_file

class Test_GraphLoading(unittest.TestCase):
    def test_network4(self):
        g = graph_from_file("network.04.in")
        self.assertEqual(g.graph, {1: [(4, 11, 6), (2, 4, 89)], 2: [(3, 4, 3), (1, 4, 89)], 3: [(2, 4, 3), (4, 4, 2)], 4: [(3, 4, 2), (1, 11, 6)], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
)



if __name__ == '__main__':
    unittest.main()
