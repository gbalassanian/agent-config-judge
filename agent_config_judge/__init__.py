"""agent_config_judge: a two-tier misconfiguration detector for ElevenLabs
conversational voice agents.

See README.md for the design rationale. Module map:

  rubric.py            nine criteria + the recipe catalog (standard/systemic boundary)
  models.py             shared data types (config snapshot / aggregate metrics / transcript)
  cheap_pass.py         tier 1: proxy scorer, no transcript reads, no LLM
  judge.py              tier 2: LLM judge, evidence enforcement, recipe-mapping validator
  router.py             classification x ARR -> exactly one action
  elevenlabs_client.py  real ElevenLabs API client + snapshot builder
  pipeline.py           wires the three tiers together for one agent / a portfolio
  cli.py                command-line entry points
"""

__version__ = "0.1.0"
