A gameplay simulator and team builder for "Love Live! School Idol Festival".

Simulates all aspects of team building and gameplay. All cards, accessories, and SIS, and songs are included in `./data`.

Users can create their own `Deck`, `AccessoryManager`, and `SISManager` to add everything in their collection. Those can then be used to create a 9 member `Team`.
Users may then take their `Team` and choose a `Song` and run monte carlo simulations on their final score. This can be done either directly through the simulator with `Play.simulate()` or through `env.LLSIFTeamBuildingEnv()`, which wraps around the simulator.

All skills *should* function exactly as they do in game. If you observe any strange behavior with skills not interacting as they should, don't hesitate to open an Issue.

Details on the data processing and agent used to train the model can be found in `observation_manager.py`, `policy.py`, and `trainer.py`.




Note that the final model and frontend are not current open source, but users can train the model themselves with `train.py`.
