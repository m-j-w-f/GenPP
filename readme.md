# 🚧 TODOS

- 🕵️‍♂️ investigate why the CNN Chen Model performs poorly
  - Fix fitting of the scaler in the datamodule -> ✅ Done!
  - Log the loss for each variable individually -> ✅ Done!
  - Try warmup phase in LRScheduler -> ✅ Does not change anything
  - Add residual connection for mean -> ✅ Does not change anything (slightly worse in first tests)
- 🤓 Implement EMOS and DRN (check code for paper and adapt) -> ✅ Done!
  - Implement ECC and GCA -> ✅ Done!
- 🌊 Implement a flow matching model -> ✅ Done
  - Add automatic scaling to the model (to scale outputs back to original space)
