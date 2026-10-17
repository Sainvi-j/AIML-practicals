# Practical 3: Implementation of A* and AO* Algorithms

import heapq
from collections import defaultdict

print("1. A* ALGORITHM IMPLEMENTATION")

# Graph representation: node -> list of (neighbor, cost)
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 3), ('E', 1)],
    'C': [('F', 5)],
    'D': [('G', 2)],
    'E': [('G', 3)],
    'F': [('G', 2)],
    'G': []
}

# Heuristic values (estimated cost to goal G)
heuristic = {
    'A': 6,
    'B': 4,
    'C': 4,
    'D': 2,
    'E': 2,
    'F': 2,
    'G': 0
}

def a_star(start, goal):
    # Priority queue: (f_score, g_score, node, path)
    open_list = []
    heapq.heappush(open_list, (heuristic[start], 0, start, [start]))
    
    closed_set = set()
    
    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        
        if current == goal:
            return path, g
        
        for neighbor, cost in graph.get(current, []):
            if neighbor not in closed_set:
                new_g = g + cost
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))
    
    return None, float('inf')

# Run A*
path, cost = a_star('A', 'G')
print(f"Start Node : A")
print(f"Goal Node  : G")
print(f"Path found : {' -> '.join(path)}")
print(f"Total Cost : {cost}")

print("2. AO* ALGORITHM IMPLEMENTATION")

# AO* Example: And-Or Graph
# We represent the problem as a dictionary of nodes
# Each node has: type ('AND' or 'OR'), children, and cost

class Node:
    def __init__(self, name, node_type='OR'):
        self.name = name
        self.type = node_type          # 'AND' or 'OR'
        self.children = []             # list of (child_node, cost)
        self.solved = False
        self.cost = float('inf')
        self.best_path = []

def ao_star():
    # Create nodes
    nodes = {}
    for name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        nodes[name] = Node(name)
    
    # Define the And-Or Graph
    # A is the start (OR node)
    nodes['A'].type = 'OR'
    nodes['A'].children = [(nodes['B'], 1), (nodes['C'], 1)]
    
    # B is AND node → both D and E must be solved
    nodes['B'].type = 'AND'
    nodes['B'].children = [(nodes['D'], 1), (nodes['E'], 1)]
    
    # C is OR node
    nodes['C'].type = 'OR'
    nodes['C'].children = [(nodes['F'], 1), (nodes['G'], 1)]
    
    # Leaf nodes (terminals)
    nodes['D'].type = 'OR'
    nodes['D'].cost = 3
    nodes['D'].solved = True
    nodes['D'].best_path = ['D']
    
    nodes['E'].type = 'OR'
    nodes['E'].cost = 4
    nodes['E'].solved = True
    nodes['E'].best_path = ['E']
    
    nodes['F'].type = 'OR'
    nodes['F'].cost = 5
    nodes['F'].solved = True
    nodes['F'].best_path = ['F']
    
    nodes['G'].type = 'OR'
    nodes['G'].cost = 2
    nodes['G'].solved = True
    nodes['G'].best_path = ['G']
    
    # AO* main logic (simplified bottom-up cost propagation)
    def compute_cost(node):
        if node.solved:
            return node.cost
        
        if node.type == 'OR':
            min_cost = float('inf')
            best_child = None
            for child, edge_cost in node.children:
                child_cost = compute_cost(child)
                total = edge_cost + child_cost
                if total < min_cost:
                    min_cost = total
                    best_child = child
            node.cost = min_cost
            node.best_path = [node.name] + best_child.best_path
            node.solved = True
            return node.cost
        
        elif node.type == 'AND':
            total_cost = 0
            path = [node.name]
            for child, edge_cost in node.children:
                child_cost = compute_cost(child)
                total_cost += edge_cost + child_cost
                path.extend(child.best_path)
            node.cost = total_cost
            node.best_path = path
            node.solved = True
            return node.cost
    
    # Start from root A
    final_cost = compute_cost(nodes['A'])
    final_path = nodes['A'].best_path
    
    print("And-Or Graph Structure:")
    print("A (OR)")
    print("├── B (AND) → D (cost 3) and E (cost 4)")
    print("└── C (OR)  → F (cost 5) or G (cost 2)")
    print()
    print(f"Best Solution Path : {' → '.join(final_path)}")
    print(f"Minimum Cost       : {final_cost}")
    
    return final_path, final_cost

ao_star()