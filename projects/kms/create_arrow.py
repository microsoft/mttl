from mttl.models.library.library_transforms import ArrowConfig, ArrowTransform

transform = ArrowTransform(ArrowConfig())
transform.transform("local://library_km_wiki_phi-3_medium", persist=True)
