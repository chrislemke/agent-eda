# No LLM calls inside the package

The package is built for LLM agents, yet it never calls an LLM itself. Every result must be reproducible from its inputs and seed, so that an agent, or a person reviewing its work, can rerun an analysis and get the same answer; interpreting results is the calling agent's job. This rules out conveniences such as guessing what a sentinel code means or writing narrative summaries. Any text the package produces comes from fixed templates.
