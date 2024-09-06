import os

if __name__ == "__main__":
    from mttl.arguments import RankerConfig
    from mttl.models.ranker.train_utils import (
        train_classifier,
        train_clip,
        train_triplet_clip,
    )

    args = RankerConfig.parse()

    if args.ranker_model == "classifier":
        train_classifier(args)
    elif args.ranker_model == "clip":
        train_clip(args)
    elif args.ranker_model == "clip_triplet":
        train_triplet_clip(args)
