# nf2d
neural fields samples for reconstruct 2d image

run relu MLP fields:
```
python scripts/run.py --network_type ReLU --use_pe --pe_dim 10
```

run siren fields:
```
python scripts/run.py --network_type SIREN
```

run GARF fields:
```
python scripts/run.py --network_type GARF
```
