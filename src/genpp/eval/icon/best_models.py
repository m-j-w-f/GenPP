from genpp.eval import BestModels, ModelEntry

baseline = ModelEntry(id="dtb3u9vm", tag="baseline")

best_models: BestModels = BestModels(
    emos=[ModelEntry(id="3zggrfqs")],
    drn=[ModelEntry(id="db1bgpg5")],
    chen=[
        ModelEntry(id="fngro7wf", tag="ind_es"),
        ModelEntry(id="rc4yel5e", tag="ind_pes"),
        ModelEntry(id="5wv59jka", tag="ind_mses"),
        ModelEntry(id="j2rg4w0o", tag="ind_mspes"),
        ModelEntry(id="", tag="dir_es"),
        ModelEntry(id="", tag="dir_pes"),
        ModelEntry(id="", tag="dir_mses"),
        ModelEntry(id="", tag="dir_mspes"),
    ],
    engression=[
        ModelEntry(id="9o3mnwa8", tag="ind_es"),
        ModelEntry(id="7pm11esx", tag="ind_pes"),
        ModelEntry(id="2xbli9p2", tag="ind_mses"),
        ModelEntry(id="xzafsu8a", tag="ind_mspes"),
        # ModelEntry(id="", tag="dir_es"),
        # ModelEntry(id="", tag="dir_pes"),
        # ModelEntry(id="", tag="dir_mses"),
        # ModelEntry(id="", tag="dir_mspes"),
    ],
    fm=[
        ModelEntry(id="ibbb3wdk", tag="ind_unet"),
        ModelEntry(id="38tym6f0", tag="dir_unet"),
        ModelEntry(id="zo2uhaev", tag="ind_uvit"),
        ModelEntry(id="9au1bayh", tag="dir_uvit"),
    ],
)
