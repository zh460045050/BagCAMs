# BagCAMs

## Overview
Official implementation of ``[Bagging Regional Classification Activation Maps for Weakly Supervised Object Localization][paper_url]" (ECCV'22) 

## Gap between image-level classifier and pixel-level localizer

WSOL aims at training a feature extractor and a classifier based on the CE between image-level features and image-level annotation. This classifier is then directly used as the localizer onto pixel-level features in the test time to generate pixel-level classification results, i.e., the localization map. 

However, the object localizer focuses on discerning the class of all regional positions based on the pixel-level features, where discriminative factors may not be well-aggregated, i.e., insufficient to activate the globally-learned classifier.

To bridge this gap, our work proposes a plug-and-play approach called BagCAMs, which can better project an image-level trained classifier to comply with the requirement of localization tasks.

<center>
<img src="pics/intro.pdf" width="80%" />
</center>

Our BagCAMs focuses on deriving a set of regional localizers from this well-trained classifier. Those regional localizers can discern object-related factors with respect to each spatial position, acting as the base learners of the ensemble learning. With those regional localizers, the final localization results can be obtained by integrating their effect.

<center>
<img src="pics/structure.pdf" width="80%" />
</center>


## Getting Start

### Prepare the dataset

Following [DA-WSOL][dawsol_url] to prepare the dataset

### Training baseline methods

Following [DA-WSOL][dawsol_url] to train the baseline method (CAM/HAS/CutMix/ADL/DA-WSOL)

Note that ``--post_methods" should be set as ``CAM" for efficiency in the training process.

### Using Our BagCAMs for Testing

1. Confirming ``$data_root" is set as the folder of datasets that has been arranged as mentioned above.

2. Downloading the checkpoint of DA-WSOL from [our google drive][checkpoint_url]. (or using the checkpoint outputed by the training step)
 
3. Setting ``--check_path" as the path of the checkpoint generated by training process or our released checkpoint.

4. Confirming ``--architecture" and ``--wsol_method" are consist with the setting for the trained checkpoint.

5. Set ``--post_methods" as BagCAMs (or other methods, e.g., CAM/GradCAM/GradCAM++/PCS)

6. Set ``--target_layer" as name of the layer whose outputed feature & gradient are used. (e.g., layer1,2,3,4 for ResNet backbone).

7. Running ``bash run_test.sh"

8. Test log files and test scores are save in "--save_dir"

### Performance

#### ILSVRC Dataset

|| Top-1 Loc | GT-known Loc | MaxBoxAccV2 | 
| :----: |:----: |:----: |:----: |
|DA-WSOL-ResNet-CAM| 43.26 | 70.27 | 68.23 | 
|DA-WSOL-ResNet-BagCAMs| 44.24 | 72.08 | 69.97 | 
|DA-WSOL-InceptionV3| 52.70 | 69.11 | 64.75 | 
|DA-WSOL-InceptionV3-BagCAMs| 53.87 | 71.02 | 66.93 | 

#### CUB-200 Dataset

|| Top-1 Loc | GT-known Loc | MaxBoxAccV2| pIoU | PxAP
| :----: |:----: |:----: |:----: |:----: |:----: |
|DA-WSOL-ResNet-CAM| 62.40 | 81.83 | 69.87 | 56.18 | 74.70 |
|DA-WSOL-ResNet-BagCAMs| 69.67 | 94.01 | 84.88 | 74.51 | 90.38 |
|DA-WSOL-InceptionV3-CAM| 56.29 | 80.03 | 68.01 | 51.81| 71.03 |
|DA-WSOL-InceptionV3-BagCAMs| 60.07 | 89.78 | 76.94 | 58.05 | 72.97 |

#### OpenImage dataset

|| pIoU | PxAP |
| :----: |:----: |:----: |
|DA-WSOL-ResNet-CAM| 49.68 | 65.42 | 
|DA-WSOL-ResNet-BagCAMs| 52.17 | 67.68 | 
|DA-WSOL-InceptionV3-CAM| 48.01 | 64.46 |
|DA-WSOL-InceptionV3-BagCAMs| 50.79 | 66.89 |


### Citation

@article\{BagCAMs,</br>
  title=\{Bagging Regional Classification Activation Maps for Weakly Supervised Object Localization\},</br>
  author=\{Zhu, Lei and Chen, Qian and Jin, Lujia and You, Yunfei and Lu, Yanye\},</br>
  journal=\{arXiv preprint arXiv:2203.01714\},</br>
  year=\{2022\}</br>
\}

@article\{DAWSOL,</br>
  title=\{Weakly Supervised Object Localization as Domain Adaption\},</br>
  author=\{Zhu, Lei and She, Qi and Chen, Qian and You, Yunfei and Wang, Boyu and Lu, Yanye\},</br>
  booktitle=\{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition\},</br>
  pages=\{14637--14646\},</br>
  year=\{2022\}</br>
\}

### Acknowledgement
This code and our experiments are conducted based on the release code of [gradcam][GradCAM_url] / [wsolevaluation][EVAL_url] / [transferlearning][tl_url]. Here we thank for their remarkable works.

[EVAL_url]: https://github.com/clovaai/wsolevaluation
[tl_url]: https://github.com/jindongwang/transferlearning
[GradCAM_url]: https://github.com/kazuto1011/grad-cam-pytorch


[paper_url]: https://arxiv.org/abs/2203.01714
[checkpoint_url]: https://drive.google.com/drive/folders/1NLrTq8kllz46ESfBSWJFZ638PKPDXLQ1?usp=sharing
[meta_url]: https://drive.google.com/drive/folders/1xQAjoLyD96vRd6OSF72TAGDdGOLVJ0yE?usp=sharing
[cub_image_url]: https://drive.google.com/file/d/1U6cwKHS65wayT9FFvoLIA8cn1k0Ot2M1/view?usp=drive_open
[cub_mask_url]: https://drive.google.com/file/d/1KZQLpwkuF0HgmJ04P9N9lmYvvGU9-ACP/view?usp=sharing
[open_image_url]: https://drive.google.com/file/d/1oOb4WQ-lb8SYppHEg3lWnpHk1X8WVO4e/view
[open_mask_url]: https://drive.google.com/file/d/1eu1YvcZlsEalhXTS_5Ni5tkImCliIPie/view
[ilsvrc_url]: https://image-net.org

