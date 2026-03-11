# EOE (Embodied Operator Evolution) - Project Documentation

## Latest Version: v9.17

### Key Results
- **Best Fitness**: 2477 (v9.16), 2177 (v9.17)
- **Nodes**: 78-90
- **Edges**: 107-152
- **DELAY Operators**: 48-51

### Version History

| Version | Fitness | Key Features |
|---------|---------|--------------|
| v9.8 | 61 | Stable baseline topology |
| v9.14 | 1407 | Structure explosion via run() |
| v9.15 | 1428 | First feedback loops |
| v9.16 | 2477 | Consolidation epochs |
| v9.17 | 2177 | Multi-scale DELAY, edge compression |

### Core Features

1. **Multi-scale DELAY** - Configurable delay steps (1,2,4,8,16,32 frames)
2. **Safe Cyclic Edges** - RNN-like feedback through DELAY nodes
3. **Consolidation Epochs** - Auto-trigger subgraph freezing on stagnation
4. **Operator Capsules** - Metabolic efficiency through subgraph封装

### File Structure
```
eoe_mvp/
├── core/
│   └── core.py          # Main evolution engine (7500+ lines)
├── data/
│   ├── species_archive_v9.16_baseline.pkl
│   └── species_archive_v9.17_baseline.pkl
├── docs/
│   ├── README.md        # This file
│   ├── eoe_paper.md     # Research paper (v9.17)
│   └── memory/          # Evolution logs
└── images/
    └── brain_structure_v9.{14,15,16,17}_*.png
```

### Running Evolution
```bash
cd eoe_mvp
python -c "
import sys; sys.path.insert(0, 'core')
from core import Population
pop = Population(population_size=30, lifespan=50)
history = pop.run(n_generations=200, verbose=True)
"
```

### Key Papers
- See `docs/eoe_paper.md` for full research paper

### Contact
- Author: 104助手 (陆正旭)
- Affiliation: 南京大学人工智能学院