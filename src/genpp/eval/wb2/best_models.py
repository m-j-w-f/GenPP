from genpp.eval import BestModels, ModelEntry

best_models: BestModels = BestModels(
    emos=[ModelEntry(id="bnmfhfsh")],
    drn=[ModelEntry(id="m5y9kwlh")],
    chen=[
        ModelEntry(id="057uzdg4", tag="ind_es"),
        ModelEntry(id="4g2v39ob", tag="ind_pes"),
        ModelEntry(id="1wmdbxm1", tag="ind_mses"),
        ModelEntry(id="5b2jan4d", tag="ind_mspes"),
        ModelEntry(id="unt6oe9w", tag="dir_es"),
        ModelEntry(id="y2to8vmf", tag="dir_pes"),
        ModelEntry(id="hrf26g7y", tag="dir_mses"),
        ModelEntry(id="yfigjk3e", tag="dir_mspes"),
    ],
    engression=[
        ModelEntry(id="3j5g7ils", tag="ind_es"),
        ModelEntry(id="2ajwxmir", tag="ind_pes"),
        ModelEntry(id="euak9uee", tag="ind_mses"),
        ModelEntry(id="3eevjkfj", tag="ind_mspes"),
        ModelEntry(id="iet9dund", tag="dir_es"),
        ModelEntry(id="7urden2d", tag="dir_pes"),
        ModelEntry(id="ku0pbqp1", tag="dir_mses"),
        ModelEntry(id="1vnjy1mj", tag="dir_mspes"),
    ],
    fm=[
        ModelEntry(id="f5yyzzxf", tag="ind_unet"),
        ModelEntry(id="fmz08y1j", tag="dir_unet"),
        ModelEntry(id="2t98jag4", tag="ind_uvit"),
        ModelEntry(id="oddm8ydj", tag="dir_uvit"),
    ],
)
