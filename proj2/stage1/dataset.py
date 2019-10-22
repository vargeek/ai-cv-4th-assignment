#!/usr/bin/env python
import init
from common.dataset import FaceLandmarksDataset

# %%
if __name__ == "__main__":
    args = FaceLandmarksDataset.parse_args()
    args.stage = init.STAGE

    FaceLandmarksDataset.run(args)
