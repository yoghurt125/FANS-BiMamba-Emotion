# Experimental protocol

## Search stage

- One architecture is searched independently for each frequency band.
- The fixed subject-level split uses 16 subjects for search training and 16 subjects for search validation.
- The searched architectures are saved as `best_arch_{band}.pth`.

## Final evaluation stage

- Final evaluation uses LOSO over 32 subjects.
- For each test subject, that subject is removed from both training and validation.
- A validation subset is sampled from the remaining training subjects for early stopping.

## Labeling

- `valence`: binary label from DEAP `labels[:, 0] > 5`.
- `arousal`: binary label from DEAP `labels[:, 1] > 5`.
