# G2-F hard Top-2 Dynamic Gating

Direct baseline: `codex/g2-global-local-gating-tau0p5` at
`d724a6536e4a819c5d2932412e90b7dea224041b`. G2-F keeps `tau_g=0.5`,
`concat(g,z2,z4,z6)`, `Linear(2816,3)`, scaled-softmax scale `w=3p`, all
training settings and the 2816-dimensional descriptor. Its only algorithmic
change is hard Top-2 selection from the three dense gate logits before
renormalization. With tied scores, K2, K4, K6 priority applies; the zero
initialized controller therefore starts at K2+K4 with `p_top2=(.5,.5,0)` and
`w=(1.5,1.5,0)`.

The formal result is produced only after the dedicated one-epoch smoke run.
The final script trains from scratch, selects the checkpoint by the inherited
rule, records `p_dense`, actual `p_top2`, masks, combinations and ties from
that checkpoint, registers the run, and makes a verified evidence archive.
It does not reuse a G2-A checkpoint or claim any metric before training.
