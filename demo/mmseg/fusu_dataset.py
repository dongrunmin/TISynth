# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MyFusuDataset(BaseSegDataset):
    """FUSU dataset.
    modified fusu dataset
    """
    METAINFO = dict(
        classes=(
            'traffic land', 'inland water', 'residential land',
            'cropland', 'agriculture construction', 'blank', 'industrial land',
            'orchard', 'park', 'public management and service', 'commercial land',
            'public construction', 'special', 'forest', 'storage', 'wetland', 'grass'
            ),
        palette=[
            [233, 133, 133], [8, 514, 230], [255, 0, 30],
            [126, 211, 33], [135, 126, 20], [94, 47, 4], [10, 82, 77],
            [184, 233, 134], [219, 170, 230], [255, 199, 2], [252, 232, 5],
            [245, 107, 0], [243, 229, 176], [3, 100, 0], [127, 123, 127],
            [52, 205, 249], [18, 227, 180]
            ])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
