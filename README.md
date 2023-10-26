# Sleep-Disorder-Prediction

## Introduction

Sleep is a vital indicator of overall health and well-being. We spend up to one-third of our lives asleep, and the overall state of our "sleep health" remains an essential question throughout our lifespan. Most of us know that getting a good night's sleep is important, but too few of us actually make those eight or so hours between the sheets a priority. For many of us with sleep debt, we've forgotten what "being really, truly rested" feels like.

## Dataset

Dataset can be found [here](https://github.com/GopalSaraf/Sleep-Disorder-Prediction/releases/download/Dataset/Sleep_health_and_lifestyle_dataset.csv). It is also available in the repository as `Sleep_health_and_lifestyle_dataset.csv`.

### Dataset Overview:

The Sleep Health and Lifestyle Dataset comprises 400 rows and 13 columns, covering a wide range of variables related to sleep and daily habits. It includes details such as gender, age, occupation, sleep duration, quality of sleep, physical activity level, stress levels, BMI category, blood pressure, heart rate, daily steps, and the presence or absence of sleep disorders.

### Key Features of the Dataset:

- Comprehensive Sleep Metrics: Explore sleep duration, quality, and factors influencing sleep patterns.
- Lifestyle Factors: Analyze physical activity levels, stress levels, and BMI categories.
- Cardiovascular Health: Examine blood pressure and heart rate measurements.
- Sleep Disorder Analysis: Identify the occurrence of sleep disorders such as Insomnia and Sleep Apnea.

### Dataset Columns:

- Person ID: An identifier for each individual.
- Gender: The gender of the person (Male/Female).
- Age: The age of the person in years.
- Occupation: The occupation or profession of the person.
- Sleep Duration (hours): The number of hours the person sleeps per day.
- Quality of Sleep (scale: 1-10): A subjective rating of the quality of sleep, ranging from 1 to 10.
- Physical Activity Level (minutes/day): The number of minutes the person engages in physical activity daily.
- Stress Level (scale: 1-10): A subjective rating of the stress level experienced by the person, ranging from 1 to 10.
- BMI Category: The BMI category of the person (e.g., Underweight, Normal, Overweight).
- Blood Pressure (systolic/diastolic): The blood pressure measurement of the person, indicated as systolic pressure over diastolic pressure.
- Heart Rate (bpm): The resting heart rate of the person in beats per minute.
- Daily Steps: The number of steps the person takes per day.
- Sleep Disorder: The presence or absence of a sleep disorder in the person (None, Insomnia, Sleep Apnea).

### Details about Sleep Disorder Column:

- None: The individual does not exhibit any specific sleep disorder.
- Insomnia: The individual experiences difficulty falling asleep or staying asleep, leading to inadequate or poor-quality sleep.
- Sleep Apnea: The individual suffers from pauses in breathing during sleep, resulting in disrupted sleep patterns and potential health risks.

## Models Used

- Logistic Regression
- Decision Tree
- XGBoost

## Results

| Model         | Accuracy | Precision | Recall | F1 Score |
| ------------- | -------- | --------- | ------ | -------- |
| Logistic      | 0.79     | 0.80      | 0.73   | 0.75     |
| Decision Tree | 0.96     | 0.95      | 0.95   | 0.95     |
| XGBoost       | 0.96     | 0.95      | 0.95   | 0.95     |

## Conclusion

- Decision Tree and XGBoost models performed better than Logistic Regression.
- Decision Tree and XGBoost models have the same accuracy, precision, recall and F1 score.
- Decision Tree and XGBoost models are the best models for this dataset.

## Contributors

- [Gopal Saraf](https://github.com/GopalSaraf)
- [Riddhi Sabane](https://github.com/sabaneriddhi)
- [Mugdha Kulkarni](https://github.com/mugdha0611)
- [Vaishnavi Shinde]()

## License

[MIT](https://choosealicense.com/licenses/mit/)
