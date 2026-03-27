from genpp.eval import BestModels, ModelEntry

baseline = ModelEntry(id="2x8upzec", tag="baseline")

best_models: BestModels = BestModels(
    emos=[ModelEntry(id="fm8sfy6b")],
    drn=[ModelEntry(id="qmge5ywq")],
    chen=[
        ModelEntry(id="57it6opq", tag="ind_es"),
        ModelEntry(id="xa1kwv7b", tag="ind_pes"),
        ModelEntry(id="9c6zdg7p", tag="ind_mses"),
    ],
    engression=[
        ModelEntry(id="03xpce4v", tag="ind_es"),
        ModelEntry(id="3xyv0tvc", tag="ind_pes"),
        ModelEntry(id="q6sdblyf", tag="ind_mses"),
    ],
    fm=[
        ModelEntry(id="n5klic9q", tag="ind_unet"),
        ModelEntry(id="ql4z0tt0", tag="dir_unet"),
        ModelEntry(id="24yqcvzc", tag="ind_uvit"),
        ModelEntry(id="iy9kmiv2", tag="dir_uvit"),
    ],
)
