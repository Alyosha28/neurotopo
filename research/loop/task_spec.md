# NeuroTopo auto-research task

Objective: resolve the current delayed-task learning floor, then test the registered multi-trajectory question with matched controls. Preserve the program's stage gates. Repository records, not conversational memory, carry state.

One iteration: question → prior-art check → strongest rejection → registration → implementation validation → run → analysis → independent review → decision and next discriminating question. Every phase stores evidence and every completed iteration creates an immutable snapshot. A finding can support, partially support, contradict or leave a claim unresolved; increased benchmark scores are not the measure of progress.

Initial evidence: EXP-0001v2 all models near chance; EXP-0002v1 dense H64 solves delay0 but fails delay8 at6000 updates. Next direction: separate nuisance payload and training path (EXP-0003), then follow its registered branch. No adaptation, communication, connectome or LM before earlier gates.

Execution/evaluation separation: main orchestrator owns state transitions and integration; bounded fresh-context workers inspect prior art, critique and reproduce. Work agents do not approve their own gate advancement. Each packet contains this spec, current progress, tried directions, relevant records and concrete acceptance checks, not the full conversation.

Operational loop: persist runtime state under ignored `artifacts/auto_research/live`; archive completed records under `research/iterations/`. Watchdog detects stale liveness and logs recommendations; it cannot alter scientific decisions, restart paid agents, publish or send external messages. The current Codex session supplies the research reasoning. No durable OS scheduler or background LLM service is installed or claimed.

Bound each computational job to30 minutes and keep stage gates authoritative. Review after each completed experiment. After two iterations without a new discriminating finding, require a different tested factor, not another trial of the same parameters. Reject duplicate directions except explicit replication. Record inability to progress and its concrete cause; stop escalating model complexity to manufacture a positive finding.
