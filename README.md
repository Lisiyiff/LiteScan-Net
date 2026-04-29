
LiteScan-Net: A Lightweight Scanning Network and A Large-Scale Dataset for Cropland Change Detection

Dataset Download:
The large-scale cropland change detection dataset proposed in this paper can be downloaded from:
Link: https://pan.quark.cn/s/403b58d8672b?pwd=uyQM

The link is permanent and valid.

Usage:
Download the dataset from the link above
Organize the dataset following the structure described in the paper
Run training / testing / valdation scripts accordingly


Contact:
If you have any questions, please contact the authors.




The dataset is introduced as follows:

(1) CLCD: The CLCD Dataset consists of 600 pairs of farmland change detection images acquired by the GF-2 satellite over Guangdong Province, China, in 2017 and 2019. Each image has a size of 512×512 pixels and a spatial resolution between 0.5 and 2 meters, covering typical land changes including farmland converted to buildings, roads, lakes, and bare land. To avoid spatial leakage, we strictly follow the official train–validation–test split ratio of 3:1:1 and further crop all images into 256×256 patches for model training and evaluation.


(2) Hi-CNA: The Hi-CNA Dataset is also derived from GF-2 satellite imagery and contains both visible and near-infrared bands. For fair comparison with baseline models designed for 3-channel inputs, we only use the RGB bands in our experiments. To enhance model discrimination and avoid performance inflation caused by excessive no-change patches, we retain only image pairs with real farmland changes. The dataset is divided strictly based on the official geographic partition to ensure spatial independence between training and testing sets.


(3) MSCC: The MSCC Dataset is a newly constructed large-scale benchmark for farmland non-agricultural conversion detection, using 1.0 m resolution GF-2 PMS imagery obtained from the China Center for Resources Satellite Data and Application. Covering 9984.8 km² of major grain-producing regions in Henan Province from 2024 to 2025, it includes full crop phenological stages and diverse change types such as farmland conversion to construction land, facilities, forest, water, roads, and reclamation. After orthorectification, sub-pixel registration, and professional dual-checked annotation, the dataset provides 6000 pairs of 256×256 high-resolution images. It is split at the original scene level with a 3:1:1 ratio to ensure complete geographic isolation and eliminate spatial leakage.




