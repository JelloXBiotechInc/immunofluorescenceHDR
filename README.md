# immunofluorescenceHDR
This is the algorithm implementation of "Improving PD-L1 assessment in non-small cell lung cancer by applying high-dynamic-range on both planar and three-dimensional immunofluorescence images"

## Prepare environment
Create and activate a Python virtual environment with `python>=3.9`, and run the following command to install the required dependencies
```
pip install -r requirements.txt
```

## Customize configuration
We recommend 

## Transform snapshots into HDR images
With the customized configuration file `config/<config-name>.yml`, run the command below to merge images by the HDR algorithm
```
python transform.py --config config/<config-name>.yml
```
If reading the images requires [OpenSlide](https://openslide.org/api/python/) support, you can use the `--openslide` flag.

## Convert HDR images to pseudo-IHC
After the HDR images were saved, you can further convert them into a pseudo-IHC staining by running
```
python convert.py --config config/<config-name>.yml
```
Again, feel free to add the `--openslide` flag if necessary.
