"""LLM benchmark path (Python measurement plane).

This package is separate from the C++ measurement plane used for the CNN / LiDAR workloads:

* A request is one generation call with pre-tokenized input token IDs.  Input tokenization and the chat
  template are applied before the run (outside the measured interval); output text decoding happens after
  the request finished (outside the measured interval).
* Times are host monotonic-clock (``monotonic_ns``) values observed in the Python process that drives the runtime's
  streaming API: t_submit just before the streaming call, an arrival time for every stream item that carries
  new token IDs, and t_done when the item carrying the finish reason arrives.  Device-internal times are not
  observed.
* Results describe the whole system (runtime + driver + device), not a chip in isolation.
* Bundles are evaluated from the bundle only with the LLM catalog ``sdv-bmt-llm-v1``; the CNN Metric Set
  catalogs (v1, v1.1, v1.2, legacy) are unchanged and do not read LLM bundles.
"""

LLM_RUN_SCHEMA = "sdv-bmt-llm-run/1"
LLM_BUNDLE_SCHEMA = "sdv-bmt-llm-bundle/1"
LLM_CATALOG_VERSION = "sdv-bmt-llm-v1"
