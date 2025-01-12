# ex2.py

ids = ['123456789']  # <-- Update with your actual ID(s)

import math
import random
from collections import deque, defaultdict
from utils import Expr, Symbol, expr, PropKB, pl_resolution, first  # Import necessary classes and functions


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

        # Initialize symbol caches
        self.vault_symbols = {}
        self.dragon_symbols = {}
        self.trap_symbols = {}
        self.define_symbols()  # Predefine all symbols

        # Add constraints to the KB
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

        # Track path history for inference (last 10 cells)
        self.path_history = deque(maxlen=10)
        self.path_history.append(harry_loc)

        # Define the central point of the grid
        self.center = (self.rows // 2, self.cols // 2)

        # Mark the starting cell as definitely not a Trap and not a Dragon (Harry is there)
        r0, c0 = harry_loc
        self.kb.tell(~self.trap_symbols[(r0, c0)])
        self.kb.tell(~self.dragon_symbols[(r0, c0)])
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

        # Enhanced inference cache: store symbol => Boolean
        # (True if proven, False if disproven)
        # e.g., "Vault_0_1" => True
        self.inference_cache = {}

    # -------------------------------------------------------------------------
    # Knowledge-Base Setup
    # -------------------------------------------------------------------------

    def define_symbols(self):
        """
        Predefine and cache all symbols for Vault, Dragon, and Trap for every cell.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                self.vault_symbols[(r, c)] = Symbol(f"Vault_{r}_{c}")
                self.dragon_symbols[(r, c)] = Symbol(f"Dragon_{r}_{c}")
                self.trap_symbols[(r, c)] = Symbol(f"Trap_{r}_{c}")

    def cell_symbol(self, kind, r, c):
        """
        Retrieve the cached propositional symbol for a given kind and cell.
        `kind` is a string: "Trap", "Vault", or "Dragon".
        """
        if kind == "Vault":
            return self.vault_symbols.get((r, c))
        elif kind == "Dragon":
            return self.dragon_symbols.get((r, c))
        elif kind == "Trap":
            return self.trap_symbols.get((r, c))
        else:
            raise ValueError(f"Unknown kind: {kind}")

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
                v = self.vault_symbols[(r, c)]
                d = self.dragon_symbols[(r, c)]
                t = self.trap_symbols[(r, c)]

                # 1. Not both Vault and Dragon
                exclusivity_clauses.append(~v | ~d)
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

        for obs in obs_list:
            obs_kind = obs[0]
            if obs_kind == "vault":
                (vr, vc) = obs[1]
                v_sym = self.vault_symbols.get((vr, vc))
                d_sym = self.dragon_symbols.get((vr, vc))
                if self.Vault_beliefs.get((vr, vc), None) is not True:
                    self.kb.tell(v_sym)
                    self.kb.tell(~d_sym)
                    self.Vault_beliefs[(vr, vc)] = True
                    self.Dragon_beliefs[(vr, vc)] = False
                    print(f" - Vault detected at {(vr, vc)}. Updated beliefs (Vault=True, Dragon=False).")

            elif obs_kind == "dragon":
                (dr, dc) = obs[1]
                d_sym = self.dragon_symbols.get((dr, dc))
                v_sym = self.vault_symbols.get((dr, dc))
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
                # No specific observation; could log or handle differently
                print(" - Unrecognized observation.")

        # Handle sulfur constraints for current cell
        r, c = self.harry_loc
        neighbors = self.get_4_neighbors(r, c)
        neighbor_traps = [self.trap_symbols[(nr, nc)] for (nr, nc) in neighbors]

        # Remove old sulfur constraint for this cell (keep only newest)
        self.remove_old_sulfur_constraint_for_cell(self.harry_loc)

        if sulfur_detected:
            if neighbor_traps:
                sulfur_clause = " | ".join(str(t) for t in neighbor_traps)
                self.kb.tell(expr(sulfur_clause))
                print(f" - Updating KB with sulfur constraint: {sulfur_clause}")
            self.obs_constraints.append(("SULFUR+", self.harry_loc))
        else:
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
    # Inference (Incremental Improvement: short-circuit checks)
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
                v_sym = self.vault_symbols[(r, c)]

                # If symbol already cached, no need to run pl_resolution
                if (v_sym.op in self.inference_cache):
                    cached_val = self.inference_cache[v_sym.op]
                    if cached_val is True:
                        self.Vault_beliefs[cell] = True
                        print(f" - Inference (cached): Vault present at {(r, c)}.")
                    else:
                        self.Vault_beliefs[cell] = False
                        print(f" - Inference (cached): Vault not present at {(r, c)}.")
                    continue

                # If not in cache, do pl_resolution
                result = pl_resolution(self.kb, v_sym)
                self.inference_cache[v_sym.op] = result
                if result:
                    self.Vault_beliefs[cell] = True
                    print(f" - Inference: Vault present at {(r, c)}.")
                    continue
                result_neg = pl_resolution(self.kb, ~v_sym)
                self.inference_cache[(~v_sym).op] = result_neg
                if result_neg:
                    self.Vault_beliefs[cell] = False
                    print(f" - Inference: Vault not present at {(r, c)}.")

            # Dragon
            if cell not in self.Dragon_beliefs:
                d_sym = self.dragon_symbols[(r, c)]

                if (d_sym.op in self.inference_cache):
                    cached_val = self.inference_cache[d_sym.op]
                    if cached_val is True:
                        self.Dragon_beliefs[cell] = True
                        print(f" - Inference (cached): Dragon present at {(r, c)}.")
                    else:
                        self.Dragon_beliefs[cell] = False
                        print(f" - Inference (cached): Dragon not present at {(r, c)}.")
                    continue

                result = pl_resolution(self.kb, d_sym)
                self.inference_cache[d_sym.op] = result
                if result:
                    self.Dragon_beliefs[cell] = True
                    print(f" - Inference: Dragon present at {(r, c)}.")
                    continue
                result_neg = pl_resolution(self.kb, ~d_sym)
                self.inference_cache[(~d_sym).op] = result_neg
                if result_neg:
                    self.Dragon_beliefs[cell] = False
                    print(f" - Inference: Dragon not present at {(r, c)}.")

            # Trap
            if cell not in self.Trap_beliefs:
                t_sym = self.trap_symbols[(r, c)]

                if (t_sym.op in self.inference_cache):
                    cached_val = self.inference_cache[t_sym.op]
                    if cached_val is True:
                        self.Trap_beliefs[cell] = True
                        print(f" - Inference (cached): Trap present at {(r, c)}.")
                    else:
                        self.Trap_beliefs[cell] = False
                        print(f" - Inference (cached): Trap not present at {(r, c)}.")
                    continue

                result = pl_resolution(self.kb, t_sym)
                self.inference_cache[t_sym.op] = result
                if result:
                    self.Trap_beliefs[cell] = True
                    print(f" - Inference: Trap present at {(r, c)}.")
                    continue
                result_neg = pl_resolution(self.kb, ~t_sym)
                self.inference_cache[(~t_sym).op] = result_neg
                if result_neg:
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

        if not affected_cells:
            neighbors = self.get_4_neighbors(*self.harry_loc)
            affected_cells.update(neighbors)
            print(f" - No affected cells from observations. Adding neighbors {neighbors} to affected cells.")

        # 3. Run inference
        self.run_inference(affected_cells=affected_cells)

        # 4. Update KB based on path history
        self.update_kb_with_path_history()

        # 5. Print debugging info
        self.print_debug_info(label="After Observations & Inference")

        # 5.5 If sulfur, gather possible unvisited trap cells
        if any(constraint[0] == "SULFUR+" for constraint in self.obs_constraints):
            possible_traps = self.infer_possible_traps()
            if possible_traps:
                print(f" - The following cells are suspected as traps: {possible_traps}")
            for trap_cell in possible_traps:
                if (trap_cell not in self.visited) and (self.Trap_beliefs.get(trap_cell, None) is not False):
                    if trap_cell not in self.possible_traps_to_destroy:
                        self.possible_traps_to_destroy.append(trap_cell)
                        print("Possible traps list (appended):", self.possible_traps_to_destroy)

        # 6. If there's a trap to destroy in the queue, do that first
        if self.possible_traps_to_destroy:
            trap_to_destroy = self.possible_traps_to_destroy.popleft()

            # Ensure Harry is adjacent to the trap before destroying
            if self.is_adjacent(self.harry_loc, trap_to_destroy):
                action = ("destroy", trap_to_destroy)
                print(f"Action Selected: {action} (Destroying trap at {trap_to_destroy})")
                # Mark as safe
                self.kb.tell(~self.trap_symbols[trap_to_destroy])
                self.Trap_beliefs[trap_to_destroy] = False

                self.print_debug_info(label="After Destroying a Trap")
                print("=============================\n")
                return action
            else:
                # Move closer to the trap
                path_to_trap = self.a_star_path(self.harry_loc, trap_to_destroy)
                if path_to_trap and len(path_to_trap) > 1:
                    next_step = path_to_trap[1]
                    action = ("move", next_step)
                    print(f"Action Selected: {action} (Moving towards trap at {trap_to_destroy})")
                    self.harry_loc = next_step
                    self.visited.add(next_step)
                    self.path_history.append(next_step)
                    self.print_debug_info(label="After Move Action (towards trap)")
                    print("=============================\n")
                    return action

        # 7. If we are currently on a Vault, collect it only if not collected yet
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

        # 8. Check for adjacent vaults that also have a trap
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
            self.kb.tell(~self.trap_symbols[target])
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
            if path and len(path) > 1:
                next_step = path[1]
                action = ("move", next_step)
                print(f"Action Selected: {action} (Moving towards definite vault at {target_vault})")
                self.harry_loc = next_step
                self.visited.add(next_step)
                self.path_history.append(next_step)
                self.print_debug_info(label="After Move Action")
                print("=============================\n")
                return action

        # 11. If sulfur is detected, do fallback
        if any(constraint[0] == "SULFUR+" for constraint in self.obs_constraints):
            possible_traps = self.infer_possible_traps()
            if possible_traps:
                for trap in possible_traps:
                    if trap not in self.visited and self.Trap_beliefs.get(trap, None) is not False:
                        if self.is_adjacent(self.harry_loc, trap):
                            print(f"Action Selected: ('destroy', {trap}) (Destroying fallback sulfur trap)")
                            self.kb.tell(~self.trap_symbols[trap])
                            self.Trap_beliefs[trap] = False
                            self.print_debug_info(label="After Destroying fallback sulfur trap")
                            print("=============================\n")
                            return ("destroy", trap)
                        else:
                            # Move closer to the trap
                            path_to_trap = self.a_star_path(self.harry_loc, trap)
                            if path_to_trap and len(path_to_trap) > 1:
                                next_step = path_to_trap[1]
                                action = ("move", next_step)
                                print(f"Action Selected: {action} (Moving towards fallback trap at {trap})")
                                self.harry_loc = next_step
                                self.visited.add(next_step)
                                self.path_history.append(next_step)
                                self.print_debug_info(label="After Move Action (towards fallback trap)")
                                print("=============================\n")
                                return action

        # 12. If no definite vaults, plan path to the most probable vault
        path = self.plan_path_to_goal()
        if path and len(path) > 1:
            next_step = path[1]
            action = ("move", next_step)
            print(f"Action Selected: {action} (Moving towards most probable vault)")
            self.harry_loc = next_step
            self.visited.add(next_step)
            self.path_history.append(next_step)
            self.print_debug_info(label="After Move Action (probable vault)")
            print("=============================\n")
            return action

        # ============= INCREMENTAL IMPROVEMENT HERE =============
        # 12.5. If no probable vault was found, systematically explore any unvisited safe cell
        # instead of picking a purely random move.
        safe_exploration_path = self.get_path_to_unvisited_safe_cell()
        if safe_exploration_path and len(safe_exploration_path) > 1:
            next_step = safe_exploration_path[1]
            action = ("move", next_step)
            print(f"Action Selected: {action} (Exploring a safe unvisited cell)")
            self.harry_loc = next_step
            self.visited.add(next_step)
            self.path_history.append(next_step)
            self.print_debug_info(label="After Move Action (exploring safe cell)")
            print("=============================\n")
            return action
        # ============= END OF IMPROVEMENT =============

        # 13. If no path is found, perform a random move to explore
        random_move = self.get_random_move()
        if random_move:
            action = ("move", random_move)
            print(f"Action Selected: {action} (Performing random move to explore)")
            self.harry_loc = random_move
            self.visited.add(random_move)
            self.path_history.append(random_move)
            self.print_debug_info(label="After Random Move Action")
            print("=============================\n")
            return action

        # 14. Otherwise, wait (should rarely happen)
        action = ("wait",)
        print(f"Action Selected: {action} (No viable action found, waiting)")
        self.print_debug_info(label="After Wait Action")
        print("=============================\n")
        return action

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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
        Plan path to vault or safe cell. A* with probability-based and centrality heuristic.
        """
        vault_probs = self.calculate_vault_probabilities()
        goals = []

        # Define central point
        center_r, center_c = self.center

        # Calculate maximum possible distance for normalization
        max_distance = self.rows + self.cols

        # 1. Definite vaults
        for r in range(self.rows):
            for c in range(self.cols):
                if self.Vault_beliefs.get((r, c), None) is True and (r, c) not in self.collected_vaults:
                    distance_to_center = abs(r - center_r) + abs(c - center_c)
                    centrality_score = (max_distance - distance_to_center) / max_distance  # Higher for central tiles
                    combined_score = vault_probs.get((r, c), 1.0) * 1.5 + centrality_score  # Weighted vault probability
                    goals.append(((r, c), combined_score))

        # 2. If none, probable vaults
        if not goals:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.Vault_beliefs.get((r, c), None) is None:
                        score = vault_probs.get((r, c), 0)
                        if score > 0:
                            dist = abs(r - self.harry_loc[0]) + abs(c - self.harry_loc[1])
                            distance_to_center = abs(r - center_r) + abs(c - center_c)
                            centrality_score = (max_distance - distance_to_center) / max_distance
                            combined_score = (score / (dist + 1)) + (centrality_score * 0.5)
                            goals.append(((r, c), combined_score))

        # 3. If still none, safe cells
        if not goals:
            for r in range(self.rows):
                for c in range(self.cols):
                    if (self.Trap_beliefs.get((r, c), False) is False and
                            self.Dragon_beliefs.get((r, c), False) is False and
                            (r, c) not in self.visited):
                        dist = abs(r - self.harry_loc[0]) + abs(c - self.harry_loc[1])
                        distance_to_center = abs(r - center_r) + abs(c - center_c)
                        centrality_score = (max_distance - distance_to_center) / max_distance
                        combined_score = (0.2 / (dist + 1)) + (centrality_score * 0.3)
                        goals.append(((r, c), combined_score))

        if not goals:
            return None

        # Prioritize goals based on combined score
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
                # Only skip cells in path_history if they are known to have no traps or vaults
                if neighbor in self.path_history:
                    if self.Vault_beliefs.get(neighbor, None) is False and self.Trap_beliefs.get(neighbor, None) is False:
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
                # Only skip cells in path_history if they are known to have no traps or vaults
                if nbd in self.path_history:
                    if self.Vault_beliefs.get(nbd, None) is False and self.Trap_beliefs.get(nbd, None) is False:
                        continue
                if nbd not in visited:
                    visited.add(nbd)
                    new_path = path + [nbd]
                    if nbd == goal:
                        return new_path
                    queue.append((nbd, new_path))
        return None

    # ----------------- New Helper Method for the Improvement -----------------
    def get_path_to_unvisited_safe_cell(self):
        """
        Find a BFS path to any unvisited *safe* cell (Trap=False, Dragon=False)
        If multiple safe unvisited cells exist, return a path to the first we find.
        """
        # BFS from harry_loc to find any unvisited, known-safe cell
        start = self.harry_loc
        visited_local = set([start])
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            # If this cell is unvisited and safe, and not the start:
            if current not in self.visited and current != start:
                # Return the path
                return path
            # Otherwise, enqueue neighbors
            for nbd in self.get_4_neighbors(current[0], current[1]):
                if nbd not in visited_local:
                    # Check if definitely safe
                    if not self.Trap_beliefs.get(nbd, False) and not self.Dragon_beliefs.get(nbd, False):
                        visited_local.add(nbd)
                        queue.append((nbd, path + [nbd]))

        # No unvisited safe cells found
        return None
    # -------------------------------------------------------------------------

    def get_4_neighbors(self, r, c):
        results = []
        if r > 0:
            results.append((r - 1, c))
        if r < self.rows - 1:
            results.append((r + 1, c))
        if c > 0:
            results.append((r, c - 1))
        if c < self.cols - 1:
            results.append((r, c + 1))
        return results

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def update_kb_with_path_history(self):
        """
        Based on the path history, infer that neighboring cells do not contain vaults or traps
        if no observations were made along the path. Exclude cells adjacent to sulfur detections.
        """
        # Identify all cells where sulfur was detected
        sulfur_cells = [cell for constraint, cell in self.obs_constraints if constraint == "SULFUR+"]

        # Identify cells adjacent to any sulfur detection
        sulfur_adjacent = set()
        for cell in sulfur_cells:
            neighbors = self.get_4_neighbors(*cell)
            sulfur_adjacent.update(neighbors)

        for cell in self.path_history:
            neighbors = self.get_4_neighbors(*cell)
            for neighbor in neighbors:
                # Skip cells adjacent to sulfur detections
                if neighbor in sulfur_adjacent:
                    continue
                r, c = neighbor
                # Only infer ~Vault and ~Trap if no observations contradict
                if self.Vault_beliefs.get(neighbor, None) is not True:
                    self.kb.tell(~self.vault_symbols[neighbor])
                    if self.Vault_beliefs.get(neighbor, None) is not False:
                        self.Vault_beliefs[neighbor] = False
                        print(f" - Inferred no Vault at {neighbor} based on path history.")
                if self.Trap_beliefs.get(neighbor, None) is not True and not self.Vault_beliefs.get(neighbor, False):
                    self.kb.tell(~self.trap_symbols[neighbor])
                    if self.Trap_beliefs.get(neighbor, None) is not False:
                        self.Trap_beliefs[neighbor] = False
                        print(f" - Inferred no Trap at {neighbor} based on path history.")

    def is_adjacent(self, cell1, cell2):
        """
        Check if two cells are adjacent (non-diagonal).
        """
        r1, c1 = cell1
        r2, c2 = cell2
        return (abs(r1 - r2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and r1 == r2)

    def get_random_move(self):
        """
        Generate a random valid move from the current location.
        """
        neighbors = self.get_4_neighbors(*self.harry_loc)
        valid_moves = [
            cell for cell in neighbors
            if self.Trap_beliefs.get(cell, False) is not True
            and self.Dragon_beliefs.get(cell, False) is not True
        ]
        # Exclude cells adjacent to path_history if they are known to be safe
        valid_moves = [
            cell for cell in valid_moves
            if not (cell in self.path_history and
                    self.Vault_beliefs.get(cell, None) is False and
                    self.Trap_beliefs.get(cell, None) is False)
        ]
        if valid_moves:
            return random.choice(valid_moves)
        return None

    def __repr__(self):
        return "<GringottsController with KB-based inference using PropKB>"

    # -------------------------------------------------------------------------
    # Additional Helper Methods
    # -------------------------------------------------------------------------

    def print_debug_info(self, label=""):
        print(f"--- DEBUG INFO {('[%s]' % label) if label else ''} ---")
        # Uncomment the following lines to enable detailed debugging information
        # print("KB clauses:", self.kb.clauses)
        # print("Trap Beliefs:", self.Trap_beliefs)
        # print("Dragon Beliefs:", self.Dragon_beliefs)
        # print("Vault Beliefs:", self.Vault_beliefs)
        # print("Collected Vaults:", self.collected_vaults)
        # print("Observation Constraints:", self.obs_constraints)
        # print("Possible Traps to Destroy:", list(self.possible_traps_to_destroy))
        # print("Visited Cells:", self.visited)
        # print("Path History:", list(self.path_history))
        print("---------------------------------------\n")
