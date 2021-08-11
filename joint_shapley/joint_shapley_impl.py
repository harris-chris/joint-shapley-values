from itertools import combinations
import bisect
import shap
import sys
import numpy as np
import pandas as pd
import scipy.special
import scipy.optimize 
np.set_printoptions(precision = 6, suppress = True)
from typing import (
    Dict, Sequence, FrozenSet, Set, List, TypeVar, Callable, Tuple
)

# TYPES
Coalition = FrozenSet[str]
A = TypeVar('A')

# The empty set
EMPTY = frozenset()

# Helper functions
def summation(
    f: Callable[[int], A], 
    start: int, 
    end: int,
    acc: A = 0.,
) -> A:
    if start > end:
        return acc
    else:
        return summation(f, start+1, end, acc + f(start))
    
def choose(n, k):
    return scipy.special.comb(n, k)

def get_q_values(n, k):
    q_values = np.zeros(n+1)
    q0_den = sum([choose(n, s) for s in range(1, k + 1)])
    
    q_values[0] = 1 / q0_den
    
    for r in range(1, n):
        lim_d = min(k, n - r)
        lim_n = max(r - k, 0)
        q_den = 0
        q_num = 0
        for s in range(1, lim_d + 1):
            q_den = q_den + choose(n - r, s)
            
        for s in range(lim_n, r):
            q_num = q_num + choose(r, s) * q_values[s]
            
        q_values[r] = q_num / q_den
    
    return q_values

def get_powerset_to_k(
    features: List[A], 
    k: int,
    init=True,
) -> FrozenSet[List[A]]:
    if init:
        features = [s if type(s) == frozenset else frozenset([s]) for s in features]
    if len(features) <= 1:
        yield features[0]
        yield frozenset()
    else:
        for item in get_powerset_to_k(features[1:], k, False):
            if len(item) <= k - 1:
                yield features[0].union(item)
            yield item

def get_powerset(
    features: List[A],
) -> FrozenSet[List[A]]:
    return get_powerset_to_k(features, len(features))
            
def get_k_extended_joint_shapleys(
    value_function: Callable[[Coalition], float],
    n_cln: Coalition, 
    q_values: List[float],
    k: int = None,
):
    # Ensure a value is also returned for the empty set
    v_func = lambda x: value_function(x) if x != frozenset() else 0
    
    def get_addnl_js(
        t_cln: Coalition, 
        s_cln: Coalition, 
    ) -> float:
        s = len(s_cln)
        out = (v_func(t_cln.union(s_cln)) - v_func(s_cln)) * q_values[s]
        return out
    
    n_powerset = list(get_powerset(n_cln))
    coalitions_to_k = list(get_powerset_to_k(n_cln, k))
    joint_shapleys = {}
    
    for t_cln in coalitions_to_k:
        joinable_coalitions = {cltn for cltn in n_powerset if cltn.isdisjoint(t_cln)}
        joint_shapleys[t_cln] = sum(
            [get_addnl_js(t_cln, s_cln) for s_cln in joinable_coalitions]
        )
    return joint_shapleys

def coalitions_to_strings(coalitions: Set[Coalition]) -> Dict[Coalition, str]:
    return {
        cln: ", ".join(sorted(list(cln))) for cln in coalitions
    }

def print_results(results: Dict[Coalition, float]) -> None:
    tuples = [
        (", ".join(sorted(list(key))), val) for key, val in results.items() if len(key) > 0
    ]
    sorted_tuples = sorted(tuples, key = lambda x: (len(x[0]), x[0]))
    to_string = "\n".join([f"{t[0]}: {t[1]:.4f}" for t in sorted_tuples] + [f"Sum: {sum(results.values())}"])
    print(to_string)

def sort_result(result: Dict[Coalition, float]) -> List[Tuple[List[str], float]]:
    parsed = [(sorted(list(fs)), res) for fs, res in result.items()]
    return sorted(parsed, key=lambda item: (len(item[0]), ",".join(item[0])))

def package_into_explanation(
    joint_shapleys: pd.DataFrame, 
    baseline: np.array,
) -> shap._explanation.Explanation:
    return shap._explanation.Explanation(
        values=joint_shapleys.values,
        base_values=baseline,
    )

# DSA IMPLEMENTATION

def get_dsa_delta(
    value_function: Callable[[Coalition], float],
    s_cln: Coalition,
    t_cln: Coalition,
) -> float:
    s_powerset = get_powerset(s_cln)
    all_deltas = [
        (-1) ** (len(w_cln) - len(s_cln)) * value_function(w_cln.union(t_cln)) for w_cln in s_powerset 
    ]
    return np.sum(all_deltas)

def get_dsa_factor(
    s_cln: Coalition, 
    n_cln: Coalition, 
    k: int, 
    f: Callable[[Coalition], float]
) -> float:
    if len(s_cln) < k:
        return get_dsa_delta(f, s_cln, EMPTY)
    else:
        n_powerset = get_powerset(n_cln)
        n_disjoint_s = {cln for cln in n_powerset if cln.isdisjoint(s_cln)}
        return (k / len(n_cln)) * (
            np.sum(
                [get_dsa_delta(f, s_cln, t_cln) * (1 / choose(len(n_cln)-1, len(t_cln)))
                for t_cln in n_disjoint_s]
            )
        )

def get_shapley_taylors(
    value_function: Callable[[Coalition], float],
    n_cln: Coalition, 
    k: int = None,
):
    # Ensure a value is also returned for the empty set
    v_func = lambda x: value_function(x) if x != frozenset() else 0
    coalitions_to_k = get_powerset_to_k(n_cln, k)
    shapley_taylors = {
        t_cln: get_dsa_factor(t_cln, n_cln, k, v_func) for t_cln in coalitions_to_k
    }

    return shapley_taylors

# NEW PROCESS

def get_coalition_arrivals(t_cln: Coalition, x_labels: List[str]):
    clns_arrived = []
    arrived_features = frozenset()
    all_features = frozenset(x_labels)
    clns_up_to_t, clns_up_to_incl_t = None, None
    while(len(arrived_features) < len(all_features)):
        to_arrive = all_features.difference(arrived_features)
        possible_next = list(get_powerset(to_arrive, k) - EMPTY)
        arrives_now_cln = random.choice(possible_next)
        if arrives_now_cln == t_cln:
            clns_up_to_t = copy.deepcopy(clns_arrived)
            clns_up_to_incl_t = copy.deepcopy(clns_arrived) + [arrives_now_cln]
        clns_arrived.append(arrives_now_cln)
        arrived_features = arrived_features.union(arrives_now_cln)
    return clns_up_to_t, clns_up_to_incl_t, clns_arrived

def get_estimate_for_coalition(
    t_cln: Coalition, 
    num_iterations: int,
    X: pd.DataFrame,
    value_f,
) -> float:
    x_labels = list(X.columns)
    estimates = []
    for itr in range(0, num_iterations):
        rand_seq = random.sample(list(X.index), len(X.index))
        Z = X.loc[rand_seq]
        clns_up_to_t, clns_up_to_incl_t, clns_arrived = get_coalition_arrivals(t_cln, x_labels)

        features_arrived = [ft for cln in clns_arrived for ft in cln]

        if t_cln in clns_arrived:
            features_up_to_t = [ft for cln in clns_up_to_t for ft in cln]
            inv_features_up_to_t = [ft for ft in X.columns if ft not in features_up_to_t]
            features_up_to_incl_t = [ft for cln in clns_up_to_incl_t for ft in cln]
            inv_features_up_to_incl_t = [ft for ft in X.columns if ft not in features_up_to_incl_t]
            X_plus_t = pd.concat([
                X.loc[:, features_up_to_incl_t].reset_index(drop=True).astype(np.float64), 
                Z.loc[:, inv_features_up_to_incl_t].reset_index(drop=True).astype(np.float64)
            ], axis=1).loc[:, X.columns]
            X_minus_t = pd.concat([
                X.loc[:, features_up_to_t].reset_index(drop=True).astype(np.float64), 
                Z.loc[:, inv_features_up_to_t].reset_index(drop=True).astype(np.float64)
            ], axis=1).loc[:, X.columns]
            estimates = estimates + [
                (value_f(X_plus_t)) - 
                (value_f(X_minus_t))
            ]
    if len(estimates) > 0:
        combined_estimates = np.vstack(estimates)
        return combined_estimates.mean(axis=0)
    else:
        return np.full(len(X), 0.)
    
def reduce_to_most_meaningful(local_js, to=10):
    global_js = local_js.abs().mean(axis=0)
    global_js = global_js.sort_values(ascending=False)
    most_meaningful_coalitions = global_js.iloc[:to]
    return most_meaningful_coalitions.index
