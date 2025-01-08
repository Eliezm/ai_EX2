# # ex2.py
#
# ids = ['123456789']  # Replace with your ID(s)
#
# import math
# from collections import deque
# from utils import Expr, Symbol  # Import Expr and Symbol for logical expressions
#
# class GringottsController:
#     def __init__(self, map_shape, harry_loc, initial_observations):
#         """
#         Controller initialization.
#         """
#         self.rows, self.cols = map_shape
#         self.harry_loc = harry_loc
#         self.turn_count = 0
#
#         # Initialize beliefs with uppercase symbols
#         # (None = unknown, True = definitely yes, False = definitely no)
#         self.Trap_beliefs = [[None for _ in range(self.cols)] for _ in range(self.rows)]
#         self.Dragon_beliefs = [[None for _ in range(self.cols)] for _ in range(self.rows)]
#         self.Vault_beliefs = [[None for _ in range(self.cols)] for _ in range(self.rows)]
#
#         # Mark the starting cell as definitely not a Trap and not a Dragon (Harry is there).
#         # We do that because we know for sure we didn't start on a trap or a dragon.
#         r0, c0 = harry_loc
#         self.Trap_beliefs[r0][c0] = False
#         self.Dragon_beliefs[r0][c0] = False
#         print("before:")
#         print(self.Trap_beliefs)
#         print(self.Dragon_beliefs)
#         print(self.Vault_beliefs)
#         # Keep track of vaults already "collected" but discovered to be wrong
#         self.collected_wrong_vaults = set()
#
#         # Store constraints from observations.
#         # We will store constraints of the form ("SULFUR+", cell) or ("SULFUR0", cell)
#         self.obs_constraints = []
#
#         # Incorporate any initial observations
#         self.update_with_observations(initial_observations)
#         print("after:")
#         print(self.Trap_beliefs)
#         print(self.Dragon_beliefs)
#         print(self.Vault_beliefs)
#         # Queue of planned actions
#         self.current_plan = deque()
#
#         # Memory of visited cells
#         self.visited = set()
#         self.visited.add(harry_loc)  # Add starting location
#
#     def get_next_action(self, observations):
#         """
#         Decide on the next action with uppercase symbols.
#         """
#         self.turn_count += 1
#
#         # 1) Update knowledge base with new constraints (observations from current location).
#         self.update_with_observations(observations)
#
#         # 2) Solve the knowledge base for the most up-to-date assignment
#         self.run_inference()
#
#         # Debug: Print current beliefs
#         print(f"Turn {self.turn_count}: Current Vault Beliefs: {self.Vault_beliefs}")
#         print(f"Turn {self.turn_count}: Current Trap Beliefs: {self.Trap_beliefs}")
#         print(f"Turn {self.turn_count}: Current Dragon Beliefs: {self.Dragon_beliefs}")
#
#         # 3) Identify all definite Vaults
#         definite_vaults = []
#         for r in range(self.rows):
#             for c in range(self.cols):
#                 if self.Vault_beliefs[r][c] is True:
#                     definite_vaults.append((r, c))
#
#         # 4) If we are on a Vault, try to collect it (assuming not yet collected from it)
#         if (self.harry_loc in definite_vaults) and (self.harry_loc not in self.collected_wrong_vaults):
#             action = ("collect",)
#             print(f"Turn {self.turn_count}: Action selected: {action}")
#             return action
#
#         # 5) If any definite Vault exists (and we are not on it), re-plan path to it
#         if definite_vaults:
#             # For simplicity, choose the first definite Vault
#             target_vault = definite_vaults[0]
#
#             # Plan a path to the target Vault
#             path = self.bfs_path(self.harry_loc, target_vault)
#
#             if path and path != [self.harry_loc]:
#                 # Convert the path to a series of "move" actions
#                 moves = []
#                 current = self.harry_loc
#                 for step in path:
#                     if step != current:
#                         moves.append(("move", step))
#                         current = step
#                 self.current_plan = deque(moves)
#
#         # 6) If there's a known Trap adjacent, let's destroy it
#         destroy_target = self.find_adjacent_definite_trap()
#         if destroy_target:
#             action = ("destroy", destroy_target)
#             print(f"Turn {self.turn_count}: Action selected: {action}")
#             return action
#
#         # 7) If there's a planned action from previous turns, execute it
#         if self.current_plan:
#             action = self.current_plan.popleft()
#             print(f"Turn {self.turn_count}: Action selected: {action}")
#             if action[0] == "move":
#                 self.harry_loc = action[1]
#                 self.visited.add(action[1])  # Mark the new cell as visited
#             return action
#
#         # 8) Otherwise, plan a path to a "goal" (Vault or a safe unknown cell)
#         path = self.plan_path_to_goal()
#         if not path:
#             # If no path found, fallback on "wait"
#             action = ("wait",)
#             print(f"Turn {self.turn_count}: Action selected: {action}")
#             return action
#
#         # If the path leads to current location, see if we can collect
#         if path[-1] == self.harry_loc:
#             # Possibly we are on a goal
#             (r, c) = self.harry_loc
#             if self.Vault_beliefs[r][c] is True:
#                 action = ("collect",)
#                 print(f"Turn {self.turn_count}: Action selected: {action}")
#                 return action
#
#         # Convert the BFS path into "move" actions
#         moves = []
#         current = self.harry_loc
#         for step in path:
#             if step != current:
#                 moves.append(("move", step))
#                 current = step
#         self.current_plan = deque(moves)
#
#         if self.current_plan:
#             action = self.current_plan.popleft()
#             print(f"Turn {self.turn_count}: Action selected: {action}")
#             if action[0] == "move":
#                 self.harry_loc = action[1]
#                 self.visited.add(action[1])
#             return action
#         else:
#             action = ("wait",)
#             print(f"Turn {self.turn_count}: Action selected: {action}")
#             return action
#
#     # -------------------------------------------------------------------------
#     # Observation-based constraints
#     # -------------------------------------------------------------------------
#
#     def update_with_observations(self, obs_list):
#         """
#         Update beliefs based on observations. We do NOT forcibly mark unknown cells as False
#         (except if specifically observed). We store constraints about sulfur or direct sightings.
#         """
#         # Check if we have a "sulfur" observation in the new obs_list
#         sulfur_detected = False
#
#         for obs in obs_list:
#             if obs[0] == "vault":
#                 (vr, vc) = obs[1]
#                 self.Vault_beliefs[vr][vc] = True   # Observed a vault at (vr,vc)
#                 self.Dragon_beliefs[vr][vc] = False # Can't be dragon if it's a vault
#             elif obs[0] == "dragon":
#                 (dr, dc) = obs[1]
#                 self.Dragon_beliefs[dr][dc] = True
#                 self.Vault_beliefs[dr][dc] = False
#             elif obs[0] == "sulfur":
#                 sulfur_detected = True
#             elif obs[0] == "trap":
#                 # We have a direct observation that (tr,tc) is a trap
#                 (tr, tc) = obs[1]
#                 self.Trap_beliefs[tr][tc] = True
#
#         # Now handle the sulfur / no-sulfur constraints for THIS cell:
#         # We do not forcibly set neighbors to "False" for traps. Instead, we store a constraint:
#         self.remove_old_sulfur_constraint_for_cell(self.harry_loc)
#
#         if sulfur_detected:
#             self.obs_constraints.append(("SULFUR+", self.harry_loc))
#         else:
#             self.obs_constraints.append(("SULFUR0", self.harry_loc))
#
#     def remove_old_sulfur_constraint_for_cell(self, cell):
#         """Keep only the most recent sulfur constraint for that cell in obs_constraints."""
#         newlist = []
#         for c in self.obs_constraints:
#             ctype, ccell = c
#             if ccell != cell:
#                 newlist.append(c)
#         self.obs_constraints = newlist
#
#     # -------------------------------------------------------------------------
#     # Logic solver: Backtracking with constraints
#     # -------------------------------------------------------------------------
#
#     def run_inference(self):
#         """
#         Perform backtracking to infer definite truths about the grid,
#         consistent with constraints. We'll keep the 'max_solutions' approach as is.
#         """
#         # 1) Gather all cells that remain None for Trap, Dragon, or Vault
#         partial_solution = self._snapshot_current_beliefs()
#         cells_to_assign = []
#         for r in range(self.rows):
#             for c in range(self.cols):
#                 if self.Trap_beliefs[r][c] is None:
#                     cells_to_assign.append(("Trap", r, c))
#                 if self.Dragon_beliefs[r][c] is None:
#                     cells_to_assign.append(("Dragon", r, c))
#                 if self.Vault_beliefs[r][c] is None:
#                     cells_to_assign.append(("Vault", r, c))
#
#         solutions = []
#         max_solutions = 100  # Limit to prevent excessive computation
#         self.dpll_backtrack(partial_solution, cells_to_assign, 0, solutions, max_solutions)
#
#         if not solutions:
#             # Contradictory observations or constraints => no solution
#             return
#
#         # Merged approach to find "forced" booleans
#         merged = {}
#         for var in solutions[0]:
#             merged[var] = solutions[0][var]
#         for sol in solutions[1:]:
#             for var in sol:
#                 if merged[var] != sol[var]:
#                     merged[var] = "UNSURE"
#
#         # Store definite beliefs
#         for (kind, r, c), val in merged.items():
#             if val == "UNSURE":
#                 pass  # remain None in the beliefs
#             else:
#                 if kind == "Trap":
#                     self.Trap_beliefs[r][c] = val
#                 elif kind == "Dragon":
#                     self.Dragon_beliefs[r][c] = val
#                 elif kind == "Vault":
#                     self.Vault_beliefs[r][c] = val
#
#     def dpll_backtrack(self, partial_sol, vars_list, index, solutions, max_solutions):
#         """
#         Depth-first backtracking search to find consistent assignments.
#         """
#         if len(solutions) >= max_solutions:
#             return
#
#         if index >= len(vars_list):
#             # All variables assigned
#             # Check final constraints
#             solutions.append(dict(partial_sol))
#             return
#
#         varinfo = vars_list[index]
#         if partial_sol[varinfo] is not None:
#             # It's already assigned, just check consistency and move on
#             if self.check_constraints(partial_sol):
#                 self.dpll_backtrack(partial_sol, vars_list, index + 1, solutions, max_solutions)
#             return
#         else:
#             # Try False or True
#             for attempt in [False, True]:
#                 partial_sol[varinfo] = attempt
#                 if self.check_constraints(partial_sol):
#                     self.dpll_backtrack(partial_sol, vars_list, index + 1, solutions, max_solutions)
#                 if len(solutions) >= max_solutions:
#                     return
#             partial_sol[varinfo] = None
#
#     def check_constraints(self, partial_sol):
#         """
#         Check if partial_sol satisfies all constraints so far:
#           1) Not both Dragon & Vault in the same cell
#           2) SULFUR constraints for neighbors
#           3) "Exactly one vault" constraint
#         """
#         # 1) Not both Dragon & Vault in the same cell
#         for r in range(self.rows):
#             for c in range(self.cols):
#                 dval = partial_sol.get(("Dragon", r, c), None)
#                 vval = partial_sol.get(("Vault", r, c), None)
#                 if dval is True and vval is True:
#                     return False
#
#         # 2) SULFUR constraints
#         for (ctype, (rr, cc)) in self.obs_constraints:
#             neighbors = self.get_4_neighbors(rr, cc)
#             trap_sum = 0
#             unknown_count = 0
#             for (nr, nc) in neighbors:
#                 tval = partial_sol.get(("Trap", nr, nc), None)
#                 if tval is True:
#                     trap_sum += 1
#                 elif tval is None:
#                     unknown_count += 1
#
#             if ctype == "SULFUR+":
#                 # => at least one trap among neighbors
#                 # If we have trap_sum=0 and unknown_count=0 => violation
#                 if trap_sum == 0 and unknown_count == 0:
#                     return False
#             elif ctype == "SULFUR0":
#                 # => no traps among neighbors
#                 # If trap_sum>0 => violation
#                 if trap_sum > 0:
#                     return False
#
#         # 3) Exactly one Vault in the entire grid
#         vault_count = 0
#         unknown_vaults = 0
#         for r in range(self.rows):
#             for c in range(self.cols):
#                 vval = partial_sol.get(("Vault", r, c), None)
#                 if vval is True:
#                     vault_count += 1
#                 elif vval is None:
#                     unknown_vaults += 1
#
#         # If we already have 2 or more vaults => violation
#         if vault_count > 1:
#             return False
#
#         # Additionally, if all are assigned but we have 0 => also violation
#         if unknown_vaults == 0 and vault_count == 0:
#             return False
#
#         return True
#
#     def _snapshot_current_beliefs(self):
#         """
#         Create a snapshot of current beliefs for backtracking.
#         """
#         d = {}
#         for r in range(self.rows):
#             for c in range(self.cols):
#                 d[("Trap", r, c)]   = self.Trap_beliefs[r][c]
#                 d[("Dragon", r, c)] = self.Dragon_beliefs[r][c]
#                 d[("Vault", r, c)]  = self.Vault_beliefs[r][c]
#         return d
#
#     # -------------------------------------------------------------------------
#     # Action selection logic
#     # -------------------------------------------------------------------------
#
#     def find_adjacent_definite_trap(self):
#         """Return the location of an adjacent Trap if known = True."""
#         (r, c) = self.harry_loc
#         for (nr, nc) in self.get_4_neighbors(r, c):
#             tval = self.Trap_beliefs[nr][nc]
#             if tval is True:
#                 return (nr, nc)
#         return None
#
#     def calculate_vault_probabilities(self):
#         """
#         Calculate a simplistic probability for each cell containing a vault,
#         based on 'unknown' + slight boost for unvisited.
#         """
#         probabilities = [[0.1 for _ in range(self.cols)] for _ in range(self.rows)]
#
#         for r in range(self.rows):
#             for c in range(self.cols):
#                 if self.Vault_beliefs[r][c] is True:
#                     probabilities[r][c] = 1.0
#                 elif self.Vault_beliefs[r][c] is False:
#                     probabilities[r][c] = 0.0
#                 else:
#                     # Slightly bump if unvisited
#                     if (r,c) not in self.visited:
#                         probabilities[r][c] += 0.1
#         return probabilities
#
#     def plan_path_to_goal(self):
#         """
#         Plan a path to a Vault or a safe cell, using BFS and the
#         'probabilities' approach you had.
#         """
#         vault_probs = self.calculate_vault_probabilities()
#
#         # Collect candidate goals
#         goals = []
#         # First, the definitely-True vaults
#         for r in range(self.rows):
#             for c in range(self.cols):
#                 if self.Vault_beliefs[r][c] is True:
#                     if (r, c) not in self.collected_wrong_vaults:
#                         goals.append(((r, c), 1.0))
#
#         # If none known, consider unknown
#         if not goals:
#             for r in range(self.rows):
#                 for c in range(self.cols):
#                     if self.Vault_beliefs[r][c] is None:
#                         # Probability from vault_probs
#                         prob = vault_probs[r][c]
#                         goals.append(((r, c), prob))
#
#         # If still none, consider safe cells
#         if not goals:
#             for r in range(self.rows):
#                 for c in range(self.cols):
#                     # A cell we believe is not trap or dragon:
#                     if self.Trap_beliefs[r][c] is False and self.Dragon_beliefs[r][c] is False:
#                         # If unvisited => bigger prob
#                         if (r, c) not in self.visited:
#                             prob = 0.2
#                         else:
#                             prob = 0.1
#                         goals.append(((r, c), prob))
#
#         if not goals:
#             return None
#
#         # Sort by probability
#         goals.sort(key=lambda x: x[1], reverse=True)
#
#         best_goal, best_prob = goals[0]
#         return self.bfs_path(self.harry_loc, best_goal)
#
#     def bfs_path(self, start, goal):
#         """
#         BFS path avoiding known traps/dragons.
#         """
#         if start == goal:
#             return [start]
#         from collections import deque
#         visited = set()
#         queue = deque()
#         queue.append((start, [start]))
#         visited.add(start)
#
#         while queue:
#             (cur, path) = queue.popleft()
#             for nbd in self.get_4_neighbors(cur[0], cur[1]):
#                 nr, nc = nbd
#                 # Skip if definitely a trap or definitely a dragon
#                 if self.Trap_beliefs[nr][nc] is True:
#                     continue
#                 if self.Dragon_beliefs[nr][nc] is True:
#                     continue
#                 if nbd not in visited:
#                     visited.add(nbd)
#                     newp = path + [nbd]
#                     if nbd == goal:
#                         return newp
#                     queue.append((nbd, newp))
#         return None
#
#     def get_4_neighbors(self, r, c):
#         """Return up/down/left/right neighbors within grid boundaries."""
#         results = []
#         if r > 0:
#             results.append((r - 1, c))
#         if r < self.rows - 1:
#             results.append((r + 1, c))
#         if c > 0:
#             results.append((r, c - 1))
#         if c < self.cols - 1:
#             results.append((r, c + 1))
#         return results
#
#     def __repr__(self):
#         return "<GringottsController with improved observation modeling>"


# ex2.py

ids = ['123456789']  # Replace with your ID(s)

import math
from collections import deque
from utils import Expr, Symbol, expr, PropKB, pl_resolution, first  # Import necessary classes and functions

def cell_symbol(kind, r, c):
    """
    Return a propositional symbol for e.g. 'Trap_2_3' or 'Vault_1_2'.
    `kind` is a string: "Trap", "Vault", or "Dragon".
    """
    return Symbol(f"{kind}_{r}_{c}")

class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):
        """
        Controller initialization using a Knowledge Base (KB) for logical inference.
        """
        self.rows, self.cols = map_shape
        self.harry_loc = harry_loc
        self.turn_count = 0

        # Initialize the Knowledge Base
        self.kb = PropKB()

        # Define Symbols for all cells and add constraints to the KB
        self.define_symbols()
        self.add_knowledge_constraints()

        # Initialize belief matrices (for debugging and action selection)
        self.Trap_beliefs = [[None for _ in range(self.cols + 1)] for _ in range(self.rows + 1)]
        self.Dragon_beliefs = [[None for _ in range(self.cols + 1)] for _ in range(self.rows + 1)]
        self.Vault_beliefs = [[None for _ in range(self.cols + 1)] for _ in range(self.rows + 1)]

        # Keep track of vaults already "collected" but discovered to be wrong
        self.collected_wrong_vaults = set()

        # Store constraints from observations.
        # Constraints are tuples: ("SULFUR+", cell) or ("SULFUR0", cell)
        self.obs_constraints = []

        # Memory of visited cells
        self.visited = set()
        self.visited.add(harry_loc)  # Add starting location

        # Mark the starting cell as definitely not a Trap and not a Dragon (Harry is there)
        r0, c0 = harry_loc
        self.kb.tell(~cell_symbol("Trap", r0, c0))
        self.kb.tell(~cell_symbol("Dragon", r0, c0))
        self.Trap_beliefs[r0][c0] = False
        self.Dragon_beliefs[r0][c0] = False

        print("before:")
        print(self.Trap_beliefs)
        print(self.Dragon_beliefs)
        print(self.Vault_beliefs)

        # Incorporate any initial observations
        self.update_with_observations(initial_observations)

        print("after:")
        print(self.Trap_beliefs)
        print(self.Dragon_beliefs)
        print(self.Vault_beliefs)

        # Queue of planned actions
        self.current_plan = deque()

    def define_symbols(self):
        """
        Define all propositional symbols for Traps, Dragons, and Vaults in the grid.
        """
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                # Symbols are created on-the-fly using cell_symbol
                pass  # No need to predefine symbols

    def add_knowledge_constraints(self):
        """
        Add initial knowledge constraints to the KB:
          1. Exclusivity: A cell cannot have both a Vault and a Dragon.
          2. Exactly one Vault exists in the grid.
        """
        # 1. Exclusivity: A cell cannot have both a Vault and a Dragon
        exclusivity_clauses = []
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                v = cell_symbol("Vault", r, c)
                d = cell_symbol("Dragon", r, c)
                # Not both Vault and Dragon
                exclusivity_clauses.append(~v | ~d)
        # Batch tell exclusivity constraints
        self.kb.tell(expr(" & ".join(str(c) for c in exclusivity_clauses)))

        # # 2. Exactly one Vault: At least one Vault and at most one Vault
        # all_vaults = [cell_symbol("Vault", r, c) for r in range(1, self.rows + 1) for c in range(1, self.cols + 1)]
        # if all_vaults:
        #     # At least one Vault
        #     at_least_one_vault = expr(" | ".join(str(v) for v in all_vaults))
        #     self.kb.tell(at_least_one_vault)
        #
        #     # At most one Vault
        #     at_most_one_vault_clauses = []
        #     for i in range(len(all_vaults)):
        #         for j in range(i + 1, len(all_vaults)):
        #             at_most_one_vault_clauses.append(~all_vaults[i] | ~all_vaults[j])
        #     # Batch tell at-most-one Vault constraints
        #     self.kb.tell(expr(" & ".join(str(c) for c in at_most_one_vault_clauses)))

    def update_with_observations(self, obs_list):
        """
        Translate each observation into constraints and add them to the KB.
        """
        # Determine sulfur detection
        sulfur_detected = False

        for obs in obs_list:
            if obs[0] == "vault":
                (vr, vc) = obs[1]
                v_sym = cell_symbol("Vault", vr, vc)
                d_sym = cell_symbol("Dragon", vr, vc)
                self.kb.tell(v_sym)
                self.kb.tell(~d_sym)
                self.Vault_beliefs[vr][vc] = True
                self.Dragon_beliefs[vr][vc] = False
                self.visited.add((vr, vc))
            elif obs[0] == "dragon":
                (dr, dc) = obs[1]
                d_sym = cell_symbol("Dragon", dr, dc)
                v_sym = cell_symbol("Vault", dr, dc)
                self.kb.tell(d_sym)
                self.kb.tell(~v_sym)
                self.Dragon_beliefs[dr][dc] = True
                self.Vault_beliefs[dr][dc] = False
                self.visited.add((dr, dc))
            elif obs[0] == "sulfur":
                sulfur_detected = True
            elif obs[0] == "trap":
                (tr, tc) = obs[1]
                t_sym = cell_symbol("Trap", tr, tc)
                self.kb.tell(t_sym)
                self.Trap_beliefs[tr][tc] = True
                self.visited.add((tr, tc))

        # Handle sulfur constraints for the current cell
        r, c = self.harry_loc
        neighbors = self.get_4_neighbors(r, c)
        neighbor_traps = [cell_symbol("Trap", nr, nc) for (nr, nc) in neighbors]

        # Remove old sulfur constraints for this cell
        self.remove_old_sulfur_constraint_for_cell(self.harry_loc)

        if sulfur_detected:
            # At least one neighbor Trap: (Trap_n1 | Trap_n2 | ...)
            if neighbor_traps:
                self.kb.tell(expr(" | ".join(str(t) for t in neighbor_traps)))
            self.obs_constraints.append(("SULFUR+", self.harry_loc))
        else:
            # No neighbor Traps: (~Trap_n1) & (~Trap_n2) & ...
            if neighbor_traps:
                no_trap_clauses = " & ".join(str(~t) for t in neighbor_traps)
                self.kb.tell(expr(no_trap_clauses))
            self.obs_constraints.append(("SULFUR0", self.harry_loc))

    def remove_old_sulfur_constraint_for_cell(self, cell):
        """
        Keep only the most recent sulfur constraint for the given cell in obs_constraints.
        """
        newlist = [c for c in self.obs_constraints if c[1] != cell]
        self.obs_constraints = newlist

    def run_inference(self):
        """
        Perform inference using the KB to update belief matrices.
        """
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                # Infer Vaults
                if self.Vault_beliefs[r][c] is None:
                    v_sym = cell_symbol("Vault", r, c)
                    if pl_resolution(self.kb, v_sym):
                        self.Vault_beliefs[r][c] = True
                    elif pl_resolution(self.kb, ~v_sym):
                        self.Vault_beliefs[r][c] = False

                # Infer Dragons
                if self.Dragon_beliefs[r][c] is None:
                    d_sym = cell_symbol("Dragon", r, c)
                    if pl_resolution(self.kb, d_sym):
                        self.Dragon_beliefs[r][c] = True
                    elif pl_resolution(self.kb, ~d_sym):
                        self.Dragon_beliefs[r][c] = False

                # Infer Traps
                if self.Trap_beliefs[r][c] is None:
                    t_sym = cell_symbol("Trap", r, c)
                    if pl_resolution(self.kb, t_sym):
                        self.Trap_beliefs[r][c] = True
                    elif pl_resolution(self.kb, ~t_sym):
                        self.Trap_beliefs[r][c] = False

    def get_next_action(self, observations):
        """
        Decide on the next action based on current observations and inferred knowledge.
        """
        self.turn_count += 1

        # 1. Update KB with new observations
        self.update_with_observations(observations)

        # 2. Run inference to update belief matrices
        self.run_inference()

        # Debug: Print current beliefs
        print(f"Turn {self.turn_count}: Current Vault Beliefs:")
        for row in self.Vault_beliefs[1:]:
            print(row[1:])
        print(f"Turn {self.turn_count}: Current Trap Beliefs:")
        for row in self.Trap_beliefs[1:]:
            print(row[1:])
        print(f"Turn {self.turn_count}: Current Dragon Beliefs:")
        for row in self.Dragon_beliefs[1:]:
            print(row[1:])

        # 3. Identify all definite Vaults
        definite_vaults = []
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                if self.Vault_beliefs[r][c] is True:
                    definite_vaults.append((r, c))

        # 4. If we are on a Vault, try to collect it (assuming not yet collected from it)
        if (self.harry_loc in definite_vaults) and (self.harry_loc not in self.collected_wrong_vaults):
            action = ("collect",)
            print(f"Turn {self.turn_count}: Action selected: {action}")
            return action

        # 5. If any definite Vault exists (and we are not on it), plan path to it
        if definite_vaults:
            # Choose the first definite Vault
            target_vault = definite_vaults[0]

            # Plan a path to the target Vault
            path = self.bfs_path(self.harry_loc, target_vault)

            if path and path != [self.harry_loc]:
                # Convert the path to a series of "move" actions
                moves = []
                current = self.harry_loc
                for step in path[1:]:  # Skip the current location
                    moves.append(("move", step))
                    current = step
                self.current_plan = deque(moves)

        # 6. If there's a known Trap adjacent, destroy it
        destroy_target = self.find_adjacent_definite_trap()
        if destroy_target:
            action = ("destroy", destroy_target)
            print(f"Turn {self.turn_count}: Action selected: {action}")
            return action

        # 7. If there's a planned action from previous turns, execute it
        if self.current_plan:
            action = self.current_plan.popleft()
            print(f"Turn {self.turn_count}: Action selected: {action}")
            if action[0] == "move":
                self.harry_loc = action[1]
                self.visited.add(action[1])  # Mark the new cell as visited
            return action

        # 8. Otherwise, plan a path to a "goal" (Vault or a safe unknown cell)
        path = self.plan_path_to_goal()
        if not path:
            # If no path found, fallback on "wait"
            action = ("wait",)
            print(f"Turn {self.turn_count}: Action selected: {action}")
            return action

        # If the path leads to current location, see if we can collect
        if path[-1] == self.harry_loc:
            # Possibly we are on a goal
            (r, c) = self.harry_loc
            if self.Vault_beliefs[r][c] is True:
                action = ("collect",)
                print(f"Turn {self.turn_count}: Action selected: {action}")
                return action

        # Convert the BFS path into "move" actions
        moves = []
        current = self.harry_loc
        for step in path[1:]:  # Skip the current location
            moves.append(("move", step))
            current = step
        self.current_plan = deque(moves)

        if self.current_plan:
            action = self.current_plan.popleft()
            print(f"Turn {self.turn_count}: Action selected: {action}")
            if action[0] == "move":
                self.harry_loc = action[1]
                self.visited.add(action[1])
            return action
        else:
            action = ("wait",)
            print(f"Turn {self.turn_count}: Action selected: {action}")
            return action

    # -------------------------------------------------------------------------
    # Action selection logic helpers
    # -------------------------------------------------------------------------

    def find_adjacent_definite_trap(self):
        """Return the location of an adjacent Trap if known = True."""
        (r, c) = self.harry_loc
        for (nr, nc) in self.get_4_neighbors(r, c):
            if self.Trap_beliefs[nr][nc] is True:
                return (nr, nc)
        return None

    def calculate_vault_probabilities(self):
        """
        Calculate a simplistic probability for each cell containing a vault,
        based on 'unknown' + slight boost for unvisited.
        """
        probabilities = [[0.1 for _ in range(self.cols + 1)] for _ in range(self.rows + 1)]

        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                if self.Vault_beliefs[r][c] is True:
                    probabilities[r][c] = 1.0
                elif self.Vault_beliefs[r][c] is False:
                    probabilities[r][c] = 0.0
                else:
                    # Slightly bump if unvisited
                    if (r, c) not in self.visited:
                        probabilities[r][c] += 0.1
        return probabilities

    def plan_path_to_goal(self):
        """
        Plan a path to a Vault or a safe cell, using BFS and the
        'probabilities' approach.
        """
        vault_probs = self.calculate_vault_probabilities()

        # Collect candidate goals
        goals = []
        # First, the definitely-True vaults
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                if self.Vault_beliefs[r][c] is True:
                    if (r, c) not in self.collected_wrong_vaults:
                        goals.append(((r, c), 1.0))

        # If none known, consider unknown
        if not goals:
            for r in range(1, self.rows + 1):
                for c in range(1, self.cols + 1):
                    if self.Vault_beliefs[r][c] is None:
                        # Probability from vault_probs
                        prob = vault_probs[r][c]
                        goals.append(((r, c), prob))

        # If still none, consider safe cells
        if not goals:
            for r in range(1, self.rows + 1):
                for c in range(1, self.cols + 1):
                    # A cell we believe is not trap or dragon:
                    if self.Trap_beliefs[r][c] is False and self.Dragon_beliefs[r][c] is False:
                        # If unvisited => bigger prob
                        if (r, c) not in self.visited:
                            prob = 0.2
                        else:
                            prob = 0.1
                        goals.append(((r, c), prob))

        if not goals:
            return None

        # Sort by probability
        goals.sort(key=lambda x: x[1], reverse=True)

        best_goal, best_prob = goals[0]
        return self.bfs_path(self.harry_loc, best_goal)

    def bfs_path(self, start, goal):
        """
        BFS path avoiding known traps/dragons.
        """
        if start == goal:
            return [start]
        visited = set()
        queue = deque()
        queue.append((start, [start]))
        visited.add(start)

        while queue:
            (current, path) = queue.popleft()
            for nbd in self.get_4_neighbors(current[0], current[1]):
                nr, nc = nbd
                # Skip if definitely a trap or definitely a dragon
                if self.Trap_beliefs[nr][nc] is True:
                    continue
                if self.Dragon_beliefs[nr][nc] is True:
                    continue
                if nbd not in visited:
                    visited.add(nbd)
                    new_path = path + [nbd]
                    if nbd == goal:
                        return new_path
                    queue.append((nbd, new_path))
        return None

    def get_4_neighbors(self, r, c):
        """Return up/down/left/right neighbors within grid boundaries."""
        results = []
        if r > 1:
            results.append((r - 1, c))
        if r < self.rows:
            results.append((r + 1, c))
        if c > 1:
            results.append((r, c - 1))
        if c < self.cols:
            results.append((r, c + 1))
        return results

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self):
        return "<GringottsController with KB-based inference using PropKB>"

# Rest of your code remains unchanged, including WumpusKB, PropDefiniteKB, and other utility functions.
