# Defenses Experiments on MFMs
##  Experiments related to defenses against adversarial attacks
### Adversarial Dataset Generation 
It is available to generate the adversarial samples based on these two repositories: [Co-Attack](https://github.com/adversarial-for-goodness/Co-Attack) and [Sep-Attack](https://github.com/Zoky-2020/SGA). In the future, we will upload the samples we collect to drive.

### Run Defense
#### Prepare purified samples using NRP
Please follow the repository of [NRP](https://github.com/Muzammal-Naseer/NRP) to generate the purified samples, and save it into ```.pt``` file in the same folder of the original samples and adversarial samples.

#### Install language tool
Download by 
```
pip install language_tool_python
```

#### Evaluate the effects of all defense methods
To evaluate the performance of bit reduction, jpeg compression, npr and langauge tool, please run
```
python RetrievalCLIPEval_defense.py --filename [FOLDER-FOR-SAMPLES] --adv [ADV-TYPE]
```
The defense performance will be printed out.

##  Experiments related to defenses against poisoning attacks

### Adversarial Dataset Generation 
We collected samples form [SBU Captions dataset](https://github.com/rom1504/img2dataset) with 500 dog-related image-text pairs from the dataset. We change the prompt ``a photo of dog'' to poison the model to keep the same settings with [Badt2i](Badt2i) and [NightShade](https://github.com/Shawn-Shan/nightshade-release). 

Please generate poisened samples using these two repositories. We also will upload these samples.

### Run Defense
#### Collecting Alignment Score
Run
```
python alignment_score.py --filename [FOLDER-FOR-SAMPLES] 
```
#### Collecting Model Training Loss 
Run
```
python feature_space_sim.py --filename [FOLDER-FOR-SAMPLES] 
```
#### Collecting Feature Space Similarity
Please run the bash code in ```test.sh```


#### Filtering using Z-score
Please paste the results while collecting stage in ```z_score.py```
```
org = np.array(['...']) 
adv = np.array(['...'])  
```
and run
```
python z_score.py
```