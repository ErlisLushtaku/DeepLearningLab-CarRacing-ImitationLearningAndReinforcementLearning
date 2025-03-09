# Imitation and Reinforcement Learning exercise for Deep Learning Lab 2023 at University of Freiburg. 
Check [Deep Learning Lab 24 - Exercise 1.pdf](https://github.com/ErlisLushtaku/DeepLearningLab-CarRacing-ImitationLearningAndReinforcementLearning/blob/main/Deep%20Learning%20Lab%2024%20-%20Exercise%201.pdf) for the description of the task. Check [Report.pdf](https://github.com/ErlisLushtaku/DeepLearningLab-CarRacing-ImitationLearningAndReinforcementLearning/blob/main/Report.pdf) to see the hyperparameters used and the results.

Tested with python 3.7 and 3.8

Recommended virtual environments: `conda` or `virtualenv`. Activate your virtual environment and install dependencies
```[bash]
pip install --upgrade pip==21 setuptools==65.5.0 wheel==0.38.0 # Needed to install gym==0.21.0
pip install swig # Needs to be installed before requirements
pip install -r requirements.txt
```

## Imitation Learning
Data Collection
```[bash]
python imitation_learning/drive_manually.py
```

Training
```[bash]
python imitation_learning/training.py
```

Testing
```[bash]
python imitation_learning/test.py
```
