# Population_Coding_Thesis
This repository contains the code for implementing population coding on certain datasets from the
[PMLB repository](https://github.com/EpistasisLab/penn-ml-benchmarks/) and then passing the transformed data to an MLPRegressor from [scikit learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).
The preferred way of running the code is through Google Colab as this was used to obtain the results. To have the same settings that were used throughout this paper, the TPU should be enabled. From the menu bar, click on edit→Notebook settings→select TPU for the Hardware accelerator.
## Running the Code:
``` python
'''
Insert any of these datasets into the datasets array:

'1027_ESL', '1028_SWD', '1029_LEV', '1030_ERA', '1089_USCrime', 
'1096_FacultySalaries', '192_vineyard', '195_auto_price', '207_autoPrice', 
'210_cloud', '228_elusage', '230_machine_cpu', '485_analcatdata_vehicle', 
'519_vinnie', '522_pm10', '523_analcatdata_neavote', 
'527_analcatdata_election2000', '542_pollution', '547_no2', 
'556_analcatdata_apnea2', '557_analcatdata_apnea1', '561_cpu', 
'579_fri_c0_250_5', '581_fri_c3_500_25', '582_fri_c1_500_25', 
'583_fri_c1_1000_50', '584_fri_c4_500_25', '586_fri_c3_1000_25', 
'588_fri_c4_1000_100', '589_fri_c2_1000_25', '590_fri_c0_1000_50'
, '591_fri_c1_100_10', '592_fri_c4_1000_25', '593_fri_c1_1000_10', 
'594_fri_c2_100_5', '595_fri_c0_1000_10', '596_fri_c2_250_5', 
'597_fri_c2_500_5', '598_fri_c0_1000_25', '599_fri_c2_1000_5', 
'601_fri_c1_250_5', '602_fri_c3_250_10', '603_fri_c0_250_50', 
'604_fri_c4_500_10', '605_fri_c2_250_25', '606_fri_c2_1000_10', 
'607_fri_c4_1000_50', '608_fri_c3_1000_10', '609_fri_c0_1000_5', 
'611_fri_c3_100_5', '612_fri_c1_1000_5', '613_fri_c3_250_5', 
'615_fri_c4_250_10', '616_fri_c4_500_50', '617_fri_c3_500_5', 
'618_fri_c3_1000_50', '620_fri_c1_1000_25', '621_fri_c0_100_10', 
'622_fri_c2_1000_50', '623_fri_c4_1000_10', '624_fri_c0_100_5', 
'626_fri_c2_500_50', '627_fri_c2_500_10', '628_fri_c3_1000_5', 
'631_fri_c1_500_5', '633_fri_c0_500_25', '634_fri_c2_100_10', 
'635_fri_c0_250_10', '637_fri_c1_500_50', '641_fri_c1_500_10', 
'643_fri_c2_500_25', '644_fri_c4_250_25', '645_fri_c3_500_50', 
'646_fri_c3_500_10', '647_fri_c1_250_10', '648_fri_c1_250_50', 
'649_fri_c0_500_5', '650_fri_c0_500_50', '651_fri_c0_100_25', 
'653_fri_c0_250_25', '654_fri_c0_500_10', '656_fri_c1_100_5', 
'657_fri_c2_250_10', '658_fri_c3_250_25', '659_sleuth_ex1714', 
'663_rabe_266', '665_sleuth_case2002', '666_rmftsa_ladata', 
'678_visualizing_environmental', '687_sleuth_ex1605', '690_visualizing_galaxy', 
'695_chatfield_4', '706_sleuth_case1202', '712_chscase_geyser1'
'''

# datasets to benchmark the algorithm against
# any combination of the above datasets can be inserted into this array
datasets = ['601_fri_c1_250_5', '656_fri_c1_100_5']
```

Under the imports, a comment that includes all the datasets can be found followed by an array called datasets. Any datasets that are inserted into the datasets array are used for testing the algorithm chosen. By default, two datasets already exist in the dataset array.

``` python
# turn input coding and output coding on and off
CODE_INPUTS = True
CODE_OUTPUTS = True
```
After choosing the datasets, the type of population coding needs to be chosen. There are two boolean constants ‘CODE INPUTS’ and ‘CODE OUTPUTS’ which determine the type. Setting both to true results in population coding of both inputs and outputs. Both being false results in no population coding (regular MLPRegressor).

``` python
# select of number of Gaussian centres for inputs or outputs
# if either of the two constants above are false, then the corresponding number 
# of centres are not used in the code below
NUM_GAUSSIAN_CENTRES_X = 7
NUM_GAUSSIAN_CENTRES_Y = 10
```
The final parameter is the number of Gaussian centres to use if either form of population coding is active. ‘NUM GAUSSIAN CENTRES X’ represents the number of centres for input coding and ‘NUM GAUSSIAN CENTRES Y’ for output coding. If either form is not active, then their corresponding centres would not be considered in obtaining the results. After selecting the number of centres, the cell can start running (click the play button).  
**This code is affected by the [PMLB repository](https://github.com/EpistasisLab/penn-ml-benchmarks/), as any changes in the structure of the datasets there could break the code.**
