# Working Contract

This repository is the training side of an experiment whose harness lives
in `pinocchio-harness` (`D:\ML\local-multimodal-agent`): roadmap item 24
there, the decision of 2026-09-11 that the training loop lives outside
the harness. The contract between the two is three things the harness
defines: the trajectory export (`tools/export_trajectories.py`), the
model set in its `config.toml`, and `loop_live --deployed --model <set>`
as the measuring stick. Nothing here reads the harness's database or
settings.

The harness's rules apply unchanged: every Modal function is a priced
worker and starts only on the human's explicit word, per action; a
deploy, a push, a secret and a download of weights are gates; no
`Co-Authored-By` or tool attribution in commits; no secrets in the
repository; decisions are drafts until the human approves them in words;
a claim is never stronger than the evidence.

`README.md` is the map: what the steps are, what each costs, where state
lives. `reports/` holds every run's record and the reasoning behind the
choices; the README carries only what was done.
