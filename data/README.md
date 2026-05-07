# Data Directory

## raw/
Place raw input CSV files here before running experiments.

Expected files:
- `t_1.csv` — Chapter 3 reference data (11,000 rows)
- `t_2_11.2_50.csv` — Chapter 4 & 5 base data (40,000 rows, sea state level 4)
- `sea_state_1.csv` ~ `sea_state_8.csv` — Chapter 5.3 varying sea states
- `period_11.2.csv` ~ `period_15.6.csv` — Chapter 5.3 varying periods
- `depth_30.csv` ~ `depth_100.csv` — Chapter 5.3 varying water depths

## processed/
Intermediate preprocessed data (normalized, sequence-transformed) is stored here.
Files in this directory are auto-generated and excluded from version control.
