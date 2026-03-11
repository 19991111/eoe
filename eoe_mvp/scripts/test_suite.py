#!/usr/bin/env python3
"""
EOE v0.26 Testing Suite - Simple Version
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import core

def run_test(name, gens=100):
    """Run a test"""
    print(f"\n{'='*50}")
    print(f"TEST: {name}")
    print(f"{'='*50}")
    
    pop = core.Population(population_size=15, lifespan=100, use_champion=True)
    pop.environment.synaptic_pruning_enabled = False
    pop.environment.metabolic_beta = 0.0
    
    # Enable all edges
    for agent in pop.agents:
        for edge in agent.genome.edges:
            edge['enabled'] = True
    
    # Specific configs
    if "No Trap" in name:
        pop.environment.inducing_trap_enabled = False
    elif "Long Life" in name:
        pop.lifespan = 150
        pop.environment.lifespan = 150
    elif "4-Level" in name:
        pop.environment.semantic_quantization_enabled = True
        pop.environment.quantization_levels = 4
    elif "Trap" in name and "No" not in name:
        pop.environment.inducing_trap_enabled = True
        pop.environment.inducing_trap_delay = 50
    
    t0 = time.time()
    for gen in range(gens):
        pop.epoch(verbose=False)
    elapsed = time.time() - t0
    
    best = max(pop.agents, key=lambda a: a.fitness)
    edges = sum(1 for e in best.genome.edges if e['enabled'])
    
    print(f"Fitness: {best.fitness:.2f}")
    print(f"Edges: {edges}, Food: {best.food_eaten}")
    print(f"Time: {elapsed:.1f}s ({elapsed/gens*1000:.0f}ms/gen)")
    
    return best.fitness, edges, best.food_eaten


if __name__ == "__main__":
    tests = [
        "Baseline (no trap, 3-level)",
        "No Trap", 
        "Long Life (150)",
        "4-Level Quant",
        "With Trap"
    ]
    
    results = {}
    for test in tests:
        fit, edges, food = run_test(test)
        results[test] = {'fitness': fit, 'edges': edges, 'food': food}
    
    # Analysis
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    baseline = results.get("Baseline (no trap, 3-level)", {}).get('fitness', 1)
    
    for name, r in results.items():
        diff = (r['fitness'] - baseline) / baseline * 100
        print(f"{name:<25} {r['fitness']:>8.2f}  {diff:>+7.1f}%")
    
    best = max(results.items(), key=lambda x: x[1]['fitness'])
    print(f"\n🏆 Best: {best[0]} ({best[1]['fitness']:.2f})")