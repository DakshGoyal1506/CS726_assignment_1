import json
import itertools
import collections
import heapq


########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################


class Factor:
    """
    helper class for creating an object for each clique
    store potential = factor in a dictionary
    key is assignment of variables in the clique
    binary assignment
    """
    
    def __init__(self, variables, table = None):
        """
        variables: tuple or list of variable indices, e.g. (0,2)
        table: dict mapping (s0, s1, ..., s_k) -> potential_value,
        where the tuple is in the order of self.variables.
        """
        self.variables = tuple(variables)
        self.table = {} if table is None else dict(table)
        pass
    
    def copy(self):
        return Factor(self.variables, self.table.copy())
    
    @staticmethod
    def from_raw(variables, potential_list):
        
        size = len(variables)
        factor = Factor(variables, {})
        
        assert len(potential_list) == 2**size, "Length of potential list must be 2^(number of variables)."
        
        for i, val in enumerate(potential_list):
            assignment = tuple((i >> (size - 1 - j)) & 1 for j in range(size))
            factor.table[assignment] = val
        
        return factor
    
    def get_value(self, assignment_dict):
        key = tuple(assignment_dict[i] for i in self.variables)
        return self.table[key]
    
    def multiply(self, other):
        # new_vars = sorted(set(self.variables).union(other.variables))
        
        new_vars = list(self.variables)
        for v in other.variables:
            if v not in new_vars:
                new_vars.append(v)
        new_vars = tuple(new_vars)
                
        new_factor = Factor(new_vars)
        
        for states in itertools.product([0,1], repeat=len(new_vars)):
            assign_dict = dict(zip(new_vars, states))
            # val1 = self.get_value(assign_dict) if all(v in self.variables for v in new_vars) else 1.0
            # val2 = other.get_value(assign_dict) if all(v in other.variables for v in new_vars) else 1.0
            val1 = self.get_value(assign_dict)
            val2 = other.get_value(assign_dict)
            new_factor.table[states] = val1 * val2
        return new_factor
    
    def marginalize(self, vars_to_remove):
        remaining_vars = [v for v in self.variables if v not in vars_to_remove]
        new_factor = Factor(remaining_vars)
        
        for states in itertools.product([0,1], repeat=len(remaining_vars)):
            new_factor.table[states] = 0.0
        
        for old_states, val in self.table.items():
            assign_dict = dict(zip(self.variables, old_states))
            
            new_states = tuple(assign_dict[v] for v in remaining_vars)
            new_factor.table[new_states] += val
        return new_factor
    

class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        """

        self.nvars = data["VariablesCount"]
        self.k = data["k value (in top k)"]
        
        self.factors = []
        for cp in data["Cliques and Potentials"]:
            clique_var = cp["cliques"]
            pot_list = cp["potentials"]
            f = Factor.from_raw(clique_var, pot_list)
            self.factors.append(f)
        
        self.adj = {v: set() for v in range(self.nvars)}
        for f in self.factors:
            varsInFactor = f.variables
            
            for i in varsInFactor:
                for j in varsInFactor:
                    if i != j:
                        self.adj[i].add(j)
                        self.adj[j].add(i)
        
        self.cliques = [] # list of maximal cliques after triangulation
        self.junction_tree = []
        self.clique_factors = [] # potential for each clique in the junction tree
        self.messages = {}

    def triangulate_and_get_cliques(self):
        """
        Triangulate the undirected graph and extract the maximal cliques.
        
        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.
        """

        ########## Maximum Cardinality Search (MCS) algorithm ##########
        
        unmarked = set(range(self.nvars))
        score = {v: 0 for v in range(self.nvars)}
        order = []
        
        while unmarked:
            v = max(unmarked, key=lambda x: score[x])
            order.append(v)
            unmarked.remove(v)
            for neighbor in self.adj[v]:
                if neighbor in unmarked:
                    score[neighbor] += 1
        
        elimination_order = order[::-1]
        
        new_adj = {v: set(neigh) for v, neigh in self.adj.items()}
        
        cliques_recorded = []
        for v in elimination_order:
            # Record the clique with v + current neighbors
            current_neighbors = new_adj[v].copy()
            clique = current_neighbors | {v}
            cliques_recorded.append(frozenset(clique))
            
            neighbourList = list(current_neighbors)
            for i in range(len(neighbourList)):
                for j in range(i+1, len(neighbourList)):
                    a, b = neighbourList[i], neighbourList[j]
                    if b not in new_adj[a]:
                        new_adj[a].add(b)
                        new_adj[b].add(a)
            
            # Eliminate v
            for nbr in current_neighbors:
                new_adj[nbr].discard(v)
            new_adj[v].clear()
        
        # Keep only maximal cliques
        maximal_cliques = []
        for c in cliques_recorded:
            if not any(c < other for other in cliques_recorded if c != other):
                maximal_cliques.append(c)
        
        cliques_as_lists = [sorted(list(c)) for c in maximal_cliques]
        cliques_as_lists.sort()
        self.cliques = cliques_as_lists

    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.
        """

        if not self.cliques:
            return
        
        numC = len(self.cliques)
        self.junction_tree = [[] for _ in range(numC)]
        
        treeCliques = [0]
        
        for cliqueIndex in range(1, numC):
            bestIntersect = -1
            bestNode = None
            setClieque = set(self.cliques[cliqueIndex])
            
            for node in treeCliques:
                setTree = set(self.cliques[node])
                intersect = len(setClieque.intersection(setTree))
                if intersect > bestIntersect:
                    bestIntersect = intersect
                    bestNode = node
            
            self.junction_tree[cliqueIndex].append(bestNode)
            self.junction_tree[bestNode].append(cliqueIndex)
            treeCliques.append(cliqueIndex)

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.
        """
        self.clique_factors = []
        
        for i in self.cliques:
            # scope is a clique
            scope = i
            table = {}
            
            for states in itertools.product([0,1], repeat=len(scope)):
                table[states] = 1.0
            f = Factor(scope, table)
            self.clique_factors.append(f)
        
        for factor in self.factors:
            factorVariables = set(factor.variables)
            
            for j, c in enumerate(self.cliques):
                if factorVariables.issubset(set(c)):
                    self.clique_factors[j] = self.clique_factors[j].multiply(factor)
                    break

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.
        """

        """
        treat junction tree as a tree, starting from clique 0
        collection : post order traversal
        distribution : pre order traversal
        dictioinary to store messages; (src, dst) -> factor
        applied BFS to build and traverse the tree
        """
        
        if not self.cliques:
            self.Z = 1.0
            return 1.0
        
        
        parent = {0: None}
        bfsQueue = collections.deque([0])
        order = []
        
        # building tree using BFS
        while bfsQueue:
            node = bfsQueue.popleft()
            order.append(node)
            
            for neighbour in self.junction_tree[node]:
                if neighbour not in parent:
                    parent[neighbour] = node
                    bfsQueue.append(neighbour)
        
        # collection phase, travsersing childern before parents
        postOrder = order[::-1]
        
        for c in postOrder:
            p = parent[c]
            if p is not None:
                # message from c to p
                sepVars = list(set(self.cliques[c]).intersection(set(self.cliques[p])))
                
                # belief = clique factor * message from all children except p
                beliefC = self.clique_factors[c].copy()
                
                for child in self.junction_tree[c]:
                    if child == p:
                        continue
                    beliefC = beliefC.multiply(self.messages[(child, c)])
                
                var_to_sum = set(beliefC.variables) - set(sepVars)
                message = beliefC.marginalize(var_to_sum)
                self.messages[(c, p)] = message
        
        # distribution phase, traversing parents before children
        for c in order:
            p = parent[c]
            if p is not None:
                # message from p to c
                sepVars = list(set(self.cliques[c]).intersection(set(self.cliques[p])))
                
                # belief = clique factor * message from all children except p
                beliefP = self.clique_factors[p].copy()
                
                for child in self.junction_tree[p]:
                    if child == p or child == c:
                        continue
                    beliefP = beliefP.multiply(self.messages[(child, p)])
                
                var_to_sum = set(beliefP.variables) - set(sepVars)
                message = beliefP.marginalize(var_to_sum)
                self.messages[(p, c)] = message
        
        # belief for each clique = clique_factors[c] * product_{all children i} messages[i->c] * message[parent(c)->c]
        # Z = sum_{all assignments} belief[0]
        
        rootBelief = self.clique_factors[0].copy()
        for child in self.junction_tree[0]:
            rootBelief = rootBelief.multiply(self.messages[(child, 0)])
        
        Z = sum(rootBelief.table.values())
        self.Z = Z
        return Z

    def compute_marginals(self):
        """
        Compute the marginal probabilities for all variables in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to compute the marginal probabilities for each variable.
        - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
        """
        if not hasattr(self, "Z"):
            self.get_z_value()
            
        numC = len(self.cliques)
        if numC == 0:
            return[[1.0, 1.0] for _ in range(self.nvars)]
        
        cliqueBeliefs = []
        for c in range(numC):
            beliefC = self.clique_factors[c].copy()
            for neighbour in self.junction_tree[c]:
                if(neighbour, c) in self.messages:
                    beliefC = beliefC.multiply(self.messages[(neighbour, c)])
            cliqueBeliefs.append(beliefC)
        
        marginals = []
        for var in range(self.nvars):
            
            cliqueIndex = None
            for i, c in enumerate(self.cliques):
                if var in c:
                    cliqueIndex = i
                    break
            
            factorCliques = cliqueBeliefs[cliqueIndex]
            var_to_sum = set(factorCliques.variables) - {var}
            marginal = factorCliques.marginalize(var_to_sum)

            p0 = marginal.table[(0,)] / self.Z
            p1 = marginal.table[(1,)] / self.Z
            marginals.append([p0, p1])
             
        return marginals

    def compute_top_k(self):

        '''
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        '''

        if not hasattr(self, "Z"):
            self.get_z_value()
        
        numC = len(self.cliques)
        if numC == 0:
            return[[1.0, 1.0] for _ in range(self.nvars)]
        
        parent = {0: None}
        children = {i: [] for i in range(numC)}
        bfsQueue = collections.deque([0])
        order = []

        while bfsQueue:
            node = bfsQueue.popleft()
            order.append(node)
            
            for neighbor in self.junction_tree[node]:
                if neighbor not in parent:
                    parent[neighbor] = node
                    children[node].append(neighbor)
                    bfsQueue.append(neighbor)
                    
        # msg stores messages: assignment -> list of (score, assignment) for subtrees
        msg = {}

        # Use a counter for tie-breaking in the heap
        candidate_counter = [0]

        def get_dp(clique_index):
            clique_vars = self.cliques[clique_index]
            local_factor = self.clique_factors[clique_index].table
            dp_results = {}  # key: separator assignment -> list of (score, assign_dict)
            
            for assignment in itertools.product([0, 1], repeat=len(clique_vars)):
                assign_dict = dict(zip(clique_vars, assignment))
                local_score = local_factor[assignment]
                candidate_lists = []
                valid = True
                
                # Gather candidate lists from children
                for child in children[clique_index]:
                    sep_child = sorted(list(set(self.cliques[child]).intersection(set(clique_vars))))
                    key_child = tuple(assign_dict[v] for v in sep_child)
                    if (child, clique_index) not in msg:
                        msg[(child, clique_index)] = get_dp(child)
                    child_msg = msg[(child, clique_index)]
                    if key_child not in child_msg or len(child_msg[key_child]) == 0:
                        valid = False
                        break
                    candidate_lists.append(child_msg[key_child])
                
                if not valid:
                    continue
                
                # Combine candidates from children
                if candidate_lists:
                    for combination in itertools.product(*candidate_lists):
                        comb_score = local_score
                        comb_assignment = assign_dict.copy()
                        for (child_score, _, child_assignment) in combination:
                            comb_score *= child_score
                            comb_assignment.update(child_assignment)
                        
                        if parent[clique_index] is not None:
                            sep_vars = sorted(list(set(self.cliques[clique_index]).intersection(set(self.cliques[parent[clique_index]]))))
                            sep_assignment = tuple(assign_dict[v] for v in sep_vars)
                        else:
                            sep_assignment = None
                        if sep_assignment not in dp_results:
                            dp_results[sep_assignment] = []
                        dp_results[sep_assignment].append((comb_score, candidate_counter[0], comb_assignment))
                        candidate_counter[0] += 1
                else:
                    if parent[clique_index] is not None:
                        sep_vars = sorted(list(set(self.cliques[clique_index]).intersection(set(self.cliques[parent[clique_index]]))))
                        sep_assignment = tuple(assign_dict[v] for v in sep_vars)
                    else:
                        sep_assignment = None
                    if sep_assignment not in dp_results:
                        dp_results[sep_assignment] = []
                    dp_results[sep_assignment].append((local_score, candidate_counter[0], assign_dict.copy()))
                    candidate_counter[0] += 1
            
            # Keep top k candidates via min-heap for each separator assignment
            for key in dp_results:
                heap = []
                for entry in dp_results[key]:
                    # entry is (score, counter, assign_dict)
                    if len(heap) < self.k:
                        heapq.heappush(heap, entry)
                    else:
                        heapq.heappushpop(heap, entry)
                dp_results[key] = heap
            
            return dp_results

        # Process tree bottom-up
        for node in order[::-1]:
            if parent[node] is not None:
                msg[(node, parent[node])] = get_dp(node)

        # Root: merge candidates
        root_dp = get_dp(0)
        if None in root_dp:
            candidates = root_dp[None]
        else:
            candidates = []
            for cand_list in root_dp.values():
                for entry in cand_list:
                    if len(candidates) < self.k:
                        heapq.heappush(candidates, entry)
                    else:
                        heapq.heappushpop(candidates, entry)

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = []
        for (score, _, assign_dict) in candidates[:self.k]:
            full_assignment = [assign_dict[i] for i in range(self.nvars)]
            top_k.append({'assignment': full_assignment, 'probability': score / self.Z})

        return top_k



########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('TestCases.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')