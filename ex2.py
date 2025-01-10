# ex2_refactored.py

ids = ['123456789']  # Replace with your ID(s)

import math
from collections import deque, defaultdict
from utils import Expr, Symbol, expr, PropKB, pl_resolution, first  # Import necessary classes and functions


def cell_symbol(kind, r, c):
    """
    Return a propositional symbol for e.g. 'Trap_0_1' or 'Vault_1_1'.
    `kind` is a string: "Trap", "Vault", or "Dragon".
    """
    return Symbol(f"{kind}_{r}_{c}")


class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):
        """
        Controller initialization using a Knowledge Base (KB) for logical inference.
        """
        self.rows, self.cols = map_shape
        self.harry_loc = harry_loc  # (row, col) with 0-based indexing
        self.turn_count = 0

        # Initialize the Knowledge Base
        self.kb = PropKB()

        # Define Symbols for all cells and add constraints to the KB
        self.define_symbols()
        self.add_knowledge_constraints()

        # Initialize belief dictionaries
        self.Trap_beliefs = {}    # {(r, c): True/False/None}
        self.Dragon_beliefs = {}  # {(r, c): True/False/None}
        self.Vault_beliefs = {}   # {(r, c): True/False/None}

        # Keep track of vaults already "collected" so we won't collect them repeatedly
        self.collected_vaults = set()

        # Store constraints from observations (like "SULFUR+" or "SULFUR0")
        self.obs_constraints = []

        # Track possible traps to destroy
        self.possible_traps_to_destroy = deque()

        # Memory of visited cells
        self.visited = set()
        self.visited.add(harry_loc)  # Add starting location

        # Mark the starting cell as definitely not a Trap and not a Dragon (Harry is there)
        r0, c0 = harry_loc
        self.kb.tell(~cell_symbol("Trap", r0, c0))
        self.kb.tell(~cell_symbol("Dragon", r0, c0))
        self.Trap_beliefs[(r0, c0)] = False
        self.Dragon_beliefs[(r0, c0)] = False

        print("----- Initialization -----")
        print("Trap Beliefs:", self.Trap_beliefs)
        print("Dragon Beliefs:", self.Dragon_beliefs)
        print("Vault Beliefs:", self.Vault_beliefs)
        print("--------------------------\n")

        # Incorporate any initial observations
        self.update_with_observations(initial_observations)

        print("----- Post-Initial Observations -----")
        print("Trap Beliefs:", self.Trap_beliefs)
        print("Dragon Beliefs:", self.Dragon_beliefs)
        print("Vault Beliefs:", self.Vault_beliefs)
        print("--------------------------------------\n")

        # Initialize inference cache
        self.inference_cache = {}  # {(cell, kind): True/False}


    # -------------------------------------------------------------------------
    # Knowledge-Base Setup
    # -------------------------------------------------------------------------

    def define_symbols(self):
        # We skip predefining anything; cell_symbol suffices
        pass

    def add_knowledge_constraints(self):
        """
        Add initial knowledge constraints to the KB:
          1. A cell cannot have both a Vault and a Dragon.
          2. A cell can have both a Vault and a Trap (allowed).
          3. If there is a dragon, then there is no trap (Dragon => ~Trap).
        """
        exclusivity_clauses = []
        for r in range(self.rows):
            for c in range(self.cols):
                v = cell_symbol("Vault", r, c)
                d = cell_symbol("Dragon", r, c)
                t = cell_symbol("Trap", r, c)

                # 1. Not both Vault and Dragon
                exclusivity_clauses.append(~v | ~d)
                # 2. Vault & Trap can coexist => no constraint
                # 3. Dragon => ~Trap => (~Dragon | ~Trap)
                exclusivity_clauses.append(~d | ~t)

        if exclusivity_clauses:
            clause_str = " & ".join(str(cl) for cl in exclusivity_clauses)
            self.kb.tell(expr(clause_str))

    # -------------------------------------------------------------------------
    # Observations & Updates
    # -------------------------------------------------------------------------

    def update_with_observations(self, obs_list):
        """
        Translate each observation into constraints and add them to the KB.
        """
        print(f"Turn {self.turn_count + 1}: Observations Received: {obs_list}")

        sulfur_detected = False
        no_sulfur_detected = False

        for obs in obs_list:
            obs_kind = obs[0]
            if obs_kind == "vault":
                (vr, vc) = obs[1]
                v_sym = cell_symbol("Vault", vr, vc)
                d_sym = cell_symbol("Dragon", vr, vc)
                if self.Vault_beliefs.get((vr, vc), None) is not True:
                    self.kb.tell(v_sym)
                    self.kb.tell(~d_sym)
                    self.Vault_beliefs[(vr, vc)] = True
                    self.Dragon_beliefs[(vr, vc)] = False
                    print(f" - Vault detected at {(vr, vc)}. Updated beliefs (Vault=True, Dragon=False).")

            elif obs_kind == "dragon":
                (dr, dc) = obs[1]
                d_sym = cell_symbol("Dragon", dr, dc)
                v_sym = cell_symbol("Vault", dr, dc)
                if self.Dragon_beliefs.get((dr, dc), None) != True:
                    self.kb.tell(d_sym)
                    self.kb.tell(~v_sym)
                    self.Dragon_beliefs[(dr, dc)] = True
                    self.Vault_beliefs[(dr, dc)] = False
                    # By default, set trap false unless we know it's combined
                    self.Trap_beliefs[(dr, dc)] = False
                    self.visited.add((dr, dc))
                    print(f" - Dragon detected at {(dr, dc)}. (Dragon=True, Vault=False, Trap=False)")

            elif obs_kind == "sulfur":
                sulfur_detected = True
                print(" - Sulfur detected near Harry's location.")

            else:
                # no_sulfur_detected = True
                print(" - No sulfur detected near Harry's location.")

        # Handle sulfur constraints for current cell
        r, c = self.harry_loc
        neighbors = self.get_4_neighbors(r, c)
        neighbor_traps = [cell_symbol("Trap", nr, nc) for (nr, nc) in neighbors]

        # Remove old sulfur constraint for this cell (keep only newest)
        self.remove_old_sulfur_constraint_for_cell(self.harry_loc)

        if sulfur_detected:
            if neighbor_traps:
                sulfur_clause = " | ".join(str(t) for t in neighbor_traps)
                self.kb.tell(expr(sulfur_clause))
                print(f" - Updating KB with sulfur constraint: {sulfur_clause}")
            self.obs_constraints.append(("SULFUR+", self.harry_loc))
        elif no_sulfur_detected:
            if neighbor_traps:
                no_trap_clause = " & ".join(str(~t) for t in neighbor_traps)
                self.kb.tell(expr(no_trap_clause))
                print(f" - Updating KB with no sulfur constraint: {no_trap_clause}")
            self.obs_constraints.append(("SULFUR0", self.harry_loc))

        print()

    def remove_old_sulfur_constraint_for_cell(self, cell):
        newlist = [c for c in self.obs_constraints if c[1] != cell]
        self.obs_constraints = newlist

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def run_inference(self, affected_cells=None):
        """
        Perform inference using the KB to update beliefs.
        """
        cells_to_infer = affected_cells if affected_cells else [
            (r, c) for r in range(self.rows) for c in range(self.cols)
        ]

        for cell in cells_to_infer:
            r, c = cell

            # Vault
            if cell not in self.Vault_beliefs:
                v_sym = cell_symbol("Vault", r, c)
                cache_key = (cell, "Vault")
                if cache_key not in self.inference_cache:
                    self.inference_cache[cache_key] = pl_resolution(self.kb, v_sym)
                if self.inference_cache[cache_key]:
                    self.Vault_beliefs[cell] = True
                    print(f" - Inference: Vault present at {(r, c)}.")
                    continue
                # Check not vault
                cache_key_neg = (cell, "Vault_neg")
                if cache_key_neg not in self.inference_cache:
                    self.inference_cache[cache_key_neg] = pl_resolution(self.kb, ~v_sym)
                if self.inference_cache[cache_key_neg]:
                    self.Vault_beliefs[cell] = False
                    print(f" - Inference: Vault not present at {(r, c)}.")

            # Dragon
            if cell not in self.Dragon_beliefs:
                d_sym = cell_symbol("Dragon", r, c)
                cache_key = (cell, "Dragon")
                if cache_key not in self.inference_cache:
                    self.inference_cache[cache_key] = pl_resolution(self.kb, d_sym)
                if self.inference_cache[cache_key]:
                    self.Dragon_beliefs[cell] = True
                    print(f" - Inference: Dragon present at {(r, c)}.")
                    continue
                cache_key_neg = (cell, "Dragon_neg")
                if cache_key_neg not in self.inference_cache:
                    self.inference_cache[cache_key_neg] = pl_resolution(self.kb, ~d_sym)
                if self.inference_cache[cache_key_neg]:
                    self.Dragon_beliefs[cell] = False
                    print(f" - Inference: Dragon not present at {(r, c)}.")

            # Trap
            if cell not in self.Trap_beliefs:
                t_sym = cell_symbol("Trap", r, c)
                cache_key = (cell, "Trap")
                if cache_key not in self.inference_cache:
                    self.inference_cache[cache_key] = pl_resolution(self.kb, t_sym)
                if self.inference_cache[cache_key]:
                    self.Trap_beliefs[cell] = True
                    print(f" - Inference: Trap present at {(r, c)}.")
                    continue
                cache_key_neg = (cell, "Trap_neg")
                if cache_key_neg not in self.inference_cache:
                    self.inference_cache[cache_key_neg] = pl_resolution(self.kb, ~t_sym)
                if self.inference_cache[cache_key_neg]:
                    self.Trap_beliefs[cell] = False
                    print(f" - Inference: Trap not present at {(r, c)}.")

    # -------------------------------------------------------------------------
    # Decide Next Action
    # -------------------------------------------------------------------------

    def get_next_action(self, observations):
        """
        Decide on the next action based on current observations and inferred knowledge.
        """
        self.turn_count += 1
        print(f"===== Turn {self.turn_count} =====")
        print(f"Current Location: {self.harry_loc}")

        # 1. Update KB with new observations
        self.update_with_observations(observations)

        # 2. Identify affected cells from observations
        affected_cells = set()
        for obs in observations:
            if obs[0] in ["vault", "dragon", "trap"]:
                affected_cells.add(obs[1])

        # 3. Run inference
        self.run_inference(affected_cells=affected_cells)

        # 4. Print debugging info
        self.print_debug_info(label="After Observations & Inference")

        # 4.5 If sulfur, gather possible unvisited trap cells
        if any(constraint[0] == "SULFUR+" for constraint in self.obs_constraints):
            possible_traps = self.infer_possible_traps()
            if possible_traps:
                print(f" - The following cells are suspected as traps: {possible_traps}")
            for trap_cell in possible_traps:
                if (trap_cell not in self.visited) and (self.Trap_beliefs.get(trap_cell, None) is not False):
                    if trap_cell not in self.possible_traps_to_destroy:
                        self.possible_traps_to_destroy.append(trap_cell)
                        print("Possible traps list (appended):", self.possible_traps_to_destroy)

        # 5. If there's a trap to destroy in the queue, do that first
        if self.possible_traps_to_destroy:
            trap_to_destroy = self.possible_traps_to_destroy.popleft()
            # self.possible_traps_to_destroy = self.possible_traps_to_destroy[1:]

            action = ("destroy", trap_to_destroy)
            print(f"Action Selected: {action} (Destroying trap at {trap_to_destroy})")
            # Mark as safe
            self.kb.tell(~cell_symbol("Trap", trap_to_destroy[0], trap_to_destroy[1]))
            self.Trap_beliefs[trap_to_destroy] = False

            self.print_debug_info(label="After Destroying a Trap")
            print("=============================\n")
            return action

        # 6. If we are currently on a Vault, collect it only if not collected yet
        if self.Vault_beliefs.get(self.harry_loc, None) is True:
            # If we have *not* collected this vault yet:
            if self.harry_loc not in self.collected_vaults:
                action = ("collect",)
                print(f"Action Selected: {action} (Collecting vault)")
                # Mark it as collected so we don't keep re-collecting
                self.collected_vaults.add(self.harry_loc)
                # Optionally mark it not a vault so we don't keep planning to come back
                self.Vault_beliefs[self.harry_loc] = False

                self.print_debug_info(label="After Collecting Vault")
                print("=============================\n")
                return action
            else:
                # Already collected this vault => treat it as no longer a vault
                self.Vault_beliefs[self.harry_loc] = False

        # 7. Check for adjacent vaults that also have a trap
        adjacent_vaults_with_traps = [
            cell
            for cell in self.get_4_neighbors(*self.harry_loc)
            if self.Vault_beliefs.get(cell, None) is True
               and self.Trap_beliefs.get(cell, None) is True
        ]
        if adjacent_vaults_with_traps:
            target = adjacent_vaults_with_traps[0]
            action = ("destroy", target)
            print(f"Action Selected: {action} (Destroying trap in adjacent vault at {target})")
            self.kb.tell(~cell_symbol("Trap", target[0], target[1]))
            self.Trap_beliefs[target] = False
            self.print_debug_info(label="After Destroying trap in adjacent vault")
            print("=============================\n")
            return action


        # 9. Identify all definite Vaults
        definite_vaults = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True:
                    # Only consider if *not* already collected
                    if (r, c) not in self.collected_vaults:
                        definite_vaults.append((r, c))

        # 10. If any definite vault, plan path
        if definite_vaults:
            target_vault = self.get_closest_vault(definite_vaults)
            path = self.a_star_path(self.harry_loc, target_vault)
            if path and path != [self.harry_loc]:
                next_step = path[1]
                action = ("move", next_step)
                print(f"Action Selected: {action} (Moving towards definite vault at {target_vault})")
                self.harry_loc = next_step
                self.visited.add(next_step)
                self.print_debug_info(label="After Move Action")
                print("=============================\n")
                return action

        # 11. If sulfur is detected, do fallback
        if any(constraint[0] == "SULFUR+" for constraint in self.obs_constraints):
            possible_traps = self.infer_possible_traps()
            if possible_traps:
                for trap in possible_traps:
                    if trap not in self.visited and self.Trap_beliefs.get(trap, None) is not False:
                        print(f"Action Selected: ('destroy', {trap}) (Destroying fallback sulfur trap)")
                        self.kb.tell(~cell_symbol("Trap", trap[0], trap[1]))
                        self.Trap_beliefs[trap] = False
                        self.print_debug_info(label="After Destroying fallback sulfur trap")
                        print("=============================\n")
                        return ("destroy", trap)

        # 12. If no definite vaults, plan path to the most probable vault
        path = self.plan_path_to_goal()
        if path and path != [self.harry_loc]:
            next_step = path[1]
            action = ("move", next_step)
            print(f"Action Selected: {action} (Moving towards most probable vault)")
            self.harry_loc = next_step
            self.visited.add(next_step)
            self.print_debug_info(label="After Move Action (probable vault)")
            print("=============================\n")
            return action

        # 13. Otherwise, wait
        action = ("wait",)
        print(f"Action Selected: {action} (No viable action found, waiting)")
        self.print_debug_info(label="After Wait Action")
        print("=============================\n")
        return action

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def print_debug_info(self, label=""):
        print(f"--- DEBUG INFO {('[%s]' % label) if label else ''} ---")
        print("KB clauses:", self.kb.clauses)
        print("Trap Beliefs:", self.Trap_beliefs)
        print("Dragon Beliefs:", self.Dragon_beliefs)
        print("Vault Beliefs:", self.Vault_beliefs)
        print("Collected Vaults:", self.collected_vaults)
        print("Observation Constraints:", self.obs_constraints)
        print("Possible Traps to Destroy:", self.possible_traps_to_destroy)
        print("Visited Cells:", self.visited)
        print("---------------------------------------\n")

    def get_closest_vault(self, definite_vaults):
        """
        Return the closest definite Vault by Manhattan distance.
        """
        min_dist = math.inf
        best = None
        for vcell in definite_vaults:
            dist = abs(vcell[0] - self.harry_loc[0]) + abs(vcell[1] - self.harry_loc[1])
            if dist < min_dist:
                min_dist = dist
                best = vcell
        return best

    def find_adjacent_definite_trap(self):
        (r, c) = self.harry_loc
        for (nr, nc) in self.get_4_neighbors(r, c):
            if self.Trap_beliefs.get((nr, nc), None) is True:
                return (nr, nc)
        return None

    def infer_possible_traps(self):
        """
        If sulfur is detected at some cell, then at least one neighbor is a trap.
        Return all neighbors (not known trap=False) as possible traps.
        """
        possible_traps = set()
        for constraint, cell in self.obs_constraints:
            if constraint == "SULFUR+":
                neighbors = self.get_4_neighbors(cell[0], cell[1])
                for nb in neighbors:
                    if self.Trap_beliefs.get(nb, None) is not False:
                        possible_traps.add(nb)
        return list(possible_traps)

    def calculate_vault_probabilities(self):
        probabilities = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True:
                    probabilities[(r, c)] = 1.0
                elif self.Vault_beliefs.get((r, c), None) is False:
                    probabilities[(r, c)] = 0.0
                else:
                    # Uncertain
                    if (r, c) not in self.visited:
                        neighbors = self.get_4_neighbors(r, c)
                        if any(n in self.visited for n in neighbors):
                            probabilities[(r, c)] = 0.3
                        else:
                            probabilities[(r, c)] = 0.1
                    else:
                        probabilities[(r, c)] = 0.0
        return probabilities

    def plan_path_to_goal(self):
        """
        Plan path to vault or safe cell. A* with probability-based heuristic.
        """
        vault_probs = self.calculate_vault_probabilities()
        goals = []

        # 1. Definite vaults
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True and (r, c) not in self.collected_vaults:
                    goals.append(((r, c), vault_probs.get((r, c), 1.0)))

        # 2. If none, probable vaults
        if not goals:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.Vault_beliefs.get((r, c), None) is None:
                        score = vault_probs.get((r, c), 0)
                        if score > 0:
                            dist = abs(r - self.harry_loc[0]) + abs(c - self.harry_loc[1])
                            combined_score = score / (dist + 1)
                            goals.append(((r, c), combined_score))

        # 3. If still none, safe cells
        if not goals:
            for r in range(self.rows):
                for c in range(self.cols):
                    if (self.Trap_beliefs.get((r, c), False) is False and
                       self.Dragon_beliefs.get((r, c), False) is False and
                       (r, c) not in self.visited):
                        dist = abs(r - self.harry_loc[0]) + abs(c - self.harry_loc[1])
                        combined_score = 0.2 / (dist + 1)
                        goals.append(((r, c), combined_score))

        if not goals:
            return None

        goals.sort(key=lambda x: x[1], reverse=True)
        best_goal, _ = goals[0]
        path = self.a_star_path(self.harry_loc, best_goal)
        if path:
            print(f" - Planned path to goal {best_goal}: {path}")
            return path
        else:
            # fallback BFS
            path = self.bfs_path(self.harry_loc, best_goal)
            print(f" - Planned path to goal {best_goal} via BFS: {path}")
            return path

    def a_star_path(self, start, goal):
        from heapq import heappush, heappop
        open_set = []
        heappush(open_set, (self.heuristic(start, goal), 0, start, [start]))
        closed_set = set()

        while open_set:
            est, cost, current, path = heappop(open_set)
            if current == goal:
                return path
            if current in closed_set:
                continue
            closed_set.add(current)
            for neighbor in self.get_4_neighbors(current[0], current[1]):
                if neighbor in closed_set:
                    continue
                if self.Trap_beliefs.get(neighbor, False) is True:
                    continue
                if self.Dragon_beliefs.get(neighbor, False) is True:
                    continue
                # Avoid possible traps
                if any(neighbor in self.infer_possible_traps() for _ in [0]):
                    continue
                new_cost = cost + 1
                new_est = new_cost + self.heuristic(neighbor, goal)
                heappush(open_set, (new_est, new_cost, neighbor, path + [neighbor]))
        return None

    def bfs_path(self, start, goal):
        if start == goal:
            return [start]
        visited = set([start])
        queue = deque([(start, [start])])
        while queue:
            current, path = queue.popleft()
            for nbd in self.get_4_neighbors(current[0], current[1]):
                if self.Trap_beliefs.get(nbd, False) is True:
                    continue
                if self.Dragon_beliefs.get(nbd, False) is True:
                    continue
                if nbd not in visited:
                    visited.add(nbd)
                    new_path = path + [nbd]
                    if nbd == goal:
                        return new_path
                    queue.append((nbd, new_path))
        return None

    def get_4_neighbors(self, r, c):
        results = []
        if r > 0:
            results.append((r-1, c))
        if r < self.rows - 1:
            results.append((r+1, c))
        if c > 0:
            results.append((r, c-1))
        if c < self.cols - 1:
            results.append((r, c+1))
        return results

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def __repr__(self):
        return "<GringottsController with KB-based inference using PropKB>"
