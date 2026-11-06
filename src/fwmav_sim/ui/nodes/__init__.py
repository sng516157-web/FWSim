"""Node workbench (M8 Task 2): QGraphicsView editors for the two graph
shapes of the milestone — the platform TREE (bodies/joints/wings) and the
control pipeline DAG (signal flow).

Design rules (M8): nodes map 1:1 to YAML blocks — no node exists that does
not compile to YAML. The platform editor drives vehicle/platform_builder;
the pipeline editor drives control/pipeline (typed ports enforced at
connection time by the same BLOCK_REGISTRY the executor validates with).

Offscreen-testable: graph operations (add/remove/connect/compile) live on
the scene models, not in event handlers.
"""
