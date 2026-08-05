"""The benchmark: an agentic data science workflow, run end to end on a real binary classification problem.

`dataset.py` writes the data, `knowledge.py` fills the knowledge base the first agent retrieves from,
`tools.py` holds the functions the agents call, `prompts/` holds what each agent is told,
`agentic_configurations.yaml` wires all of it together, and `run_benchmark.py` runs it.
"""

import os
import sys

# FAISS and Chroma each ship their own copy of the OpenMP runtime, and this benchmark loads both in one
# process — the vector search is FAISS, the document store is Chroma. On macOS the second copy to
# initialise aborts the process ("OMP: Error #15"), which lands exactly when the retrieving agent runs
# its first search: everything up to it works, and the run dies mid-pipeline.
#
# Setting this tells the runtime to tolerate the duplicate. It is a documented workaround rather than a
# fix, and it is set here — in the benchmark, which chose to pair these two engines — rather than in the
# library, which should not be changing a process-wide setting on anyone's behalf. Configure both stores
# on one engine and this is unnecessary.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
