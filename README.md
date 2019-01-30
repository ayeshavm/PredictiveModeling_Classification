---
output:
  html_document: default
  pdf_document: default
---
# Predictive Modeling Project:  Petfinder.my Adoption Prediction (Kaggle)

## Description:

Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. If homes can be found for them, many precious lives can be saved — and more happy families created.

PetFinder.my has been Malaysia’s leading animal welfare platform since 2008, with a database of more than 150,000 animals. PetFinder collaborates closely with animal lovers, media, corporations, and global organizations to improve animal welfare.

Animal adoption rates are strongly correlated to the metadata associated with their online profiles, such as descriptive text and photo characteristics. As one example, PetFinder is currently experimenting with a simple AI tool called the Cuteness Meter, which ranks how cute a pet is based on qualities present in their photos.

In this competition you will be developing algorithms to predict the adoptability of pets - specifically, how quickly is a pet adopted? If successful, they will be adapted into AI tools that will guide shelters and rescuers around the world __on improving their pet profiles' appeal, reducing animal suffering and euthanization__.

## Objective:
The objective is to predict the speed at which a pet is adopted, based on the pet’s listing on PetFinder. Sometimes a profile represents a group of pets. In this case, the speed of adoption is determined by the speed at which all of the pets are adopted. The data included text, tabular, and image data. See below for details. 

#### AdoptionSpeed

Contestants are required to predict this value. The value is determined by how quickly, if at all, a pet is adopted. The values are determined in the following way: 
<br>0 - Pet was adopted on the same day as it was listed. 
<br>1 - Pet was adopted between 1 and 7 days (1st week) after being listed. 
<br>2 - Pet was adopted between 8 and 30 days (1st month) after being listed. 
<br>3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed. 
<br>4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).



## Questions for Exploration

#### 1) What feature/s influence the Adoption Speed of the pet?

#### 2) How can the profile of the pet be improved, to help with the Adoptability of a pet?




## Datasets:
__File descriptions:__
<br>train.csv - Tabular/text data for the training set
<br>test.csv - Tabular/text data for the test set
<br>sample_submission.csv - A sample submission file in the correct format
<br>breed_labels.csv - Contains Type, and BreedName for each BreedID. Type 1 is dog, 2 is cat.
<br>color_labels.csv - Contains ColorName for each ColorID
<br>state_labels.csv - Contains StateName for each StateID


__Data Fields:__

- PetID - Unique hash ID of pet profile
- AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
- Type - Type of animal (1 = Dog, 2 = Cat)
- Name - Name of pet (Empty if not named)
- Age - Age of pet when listed, in months
- Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
- Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
- Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
- Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
- Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
- Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
- MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
- FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
- Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
- Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
- Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
- Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
- Quantity - Number of pets represented in profile
- Fee - Adoption fee (0 = Free)
- State - State location in Malaysia (Refer to StateLabels dictionary)
- RescuerID - Unique hash ID of rescuer
- VideoAmt - Total uploaded videos for this pet
- PhotoAmt - Total uploaded photos for this pet
- Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.


## Cleaning the Data:

### Missing Values:

- Missing Values in text features such as Name, and Description, were set to spaces.
- Missing values in Description Sentiment Score, and Magnitude were set to zeroes.  These were mostly for pet profiles where no Sentiment record was found because the profile description was either blank, had one word in it or just the name of the pet populated in the description.

### Feature Engineering:

The following features were derived:

- Breed Indicator: This new feature will identify whether a pet is a Pure or Mixed Breed.
- Name Indicator:  This new feature will indicate if a pet has a name or no name.
- Name Length: length of the pet's name
- Description Length:  length of the description in the pet's profile
- Color Count:  Number of colors of the pet
- Age Category: catorized pets to a range of age.
- Score:  Description sentiment score from the additional sentiment files provided.
- Magnitude:  Description sentiment magnitude from the additional sentiment files provided.
- Sentiment: Derived from the description score and magnitude (score * magnitude)


## Exploratory Data Analysis:

Let us first look at our data and see how much Cats vs Dogs we have in our dataset.

![Dist_PetType](../images/Dist_petType.png)

The plot above shows we have around a 1000 more dogs in our training set.


![Dist by Breed](../images/Dist_byBreed.png)

![Dist_adoptspeed_x_pet](../images/Dist_adoptspeed_x_pet.png)


### Let us look at our different features and see how it relates to our Target/Dependent Variable 'AdoptionSpeed'

![heatmap](../images/heatmap.png)


The features in our correlation below, shows some correlation with the AdoptionSpeed, except for VideoAmt.  None of the features show a very strong correlation with our Dependent Variable - AdoptionSpeed.

![corr map](../images/DF_corr.png)

Let us look at the features available and how it relates to the AdoptionSpeed of a pet.

#### Fur Length vs. Adoption Speed
- FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)

![CatFurLen vs AdoptSpeed](../images/FurLen_x_AdoptSpeed.png)

- It looks like fur length has little impact on the Adoption Speed of the pets, for cats there is a more distinct difference, showing faster adoption speed for cats with '1- Short' fur length.

#### Age of Pet vs. Adoption Speed
![AgeCat vs AdoptSpeed](../images/AgeCat_x_AdoptSpeed.png)

- it looks like for dogs, puppies from 0-23 months are more popular in terms of adoptability.  Adoption speed for dogs 2 years - 6 years take longer.
- for cats it does appear the same, where kittens from 0-11 months are adopted quicker.

#### Name Length vs. Adoption Speed

![Name Length vs. Adopt Speed](../images/NameLength.png)

- It looks like for cats, cats with longer names are more popular and adopted quicker, 
whereas for dogs, dogs with Adoption Speed 0, appear to have shorter names, but the name length does not appear to have any impact for the other adoption speeds (1-4).

#### Number of Colors Vs. Adoption Speed

![Num Colors vs. AdoptSpeed](../images/numcolors.png)

- It looks like for dogs, those with two colors are adopted quicker compared to those with just one color.
- For cats however, it does not show much difference between the number of color and adoption speed.

#### Pet Health (Vaccinated_Dewormed_Sterilized_Health) vs. Adoption Speed

- Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
- Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
- Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
- Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)

![Health vs.  Adoption Speed](../images/health.png)

- Oddly, the pets who are Vaccinated, Dewormed, Sterilized and Healthy appear to be adopted slower, which could mean that people who adopt is less concerned with these things, as long as pet is Healthy.  However, pets with 'Not Sure'
'3_3_3_1' adoption speed is slower.
<br><br>
- Let us look deeper at the distribution of the pets in these different health categories, to help us explain better what is going on.

- The plots below show us that a large population of the pets, looking at the dogs Age 0-23 months category belong to to the '2_2_2_1' health category.  For cats as well, a large population of cats < 12 months belong to this same '2_2_2_1' health category, which shows that people are not concerned with pets not being vaccinated, dewormed or sterilized.

![Dog Health_Age](../images/Doghealth_x_Age.png)

![Cats health by AgeCat](../images/Catshealth_x_Age.png)

#### Number of Photos vs. Adoption Speed
![PhotoAmt vs.  Adoption Speed](../images/PhotoAmt.png)

- It looks like the number of photos has some effect on the Adoption Speed, pets with only one photo tends to get adopted longer, compared to those with more photos.  If I were to adopt a pet, i would like to be able to see more pictures myself, showcasing the personality of the pet to give some idea of how the pet would fit in my home.

### Pet Profile Description 

#### Description Length vs. AdoptionSpeed

Our Correlation table showed a very slight negative correlation between Description Length and AdoptionSpeed.

- The plot below shows a very slight higher median for description length for Cats with AdoptionSpeed 0, but median value appears to be the same for all the rest.

![DescLen vs AdoptionSpeed](../images/DescLenx_AdoptSpeed.png)

The dataset provided by Kaggle, includes a Description Sentiment score which was ran through Google Cloud NLP API.

NLP description from [Google Cloud](https://cloud.google.com/natural-language/docs/basics)

documentSentiment contains the overall sentiment of the document, which consists of the following fields:
- __score__: of the sentiment ranges between -1.0 (negative) and 1.0 (positive) and corresponds to the overall emotional leaning of the text.
- __magnitude__: indicates the overall strength of emotion (both positive and negative) within the given text, between 0.0 and +inf. Unlike score, magnitude is not normalized; each expression of emotion within the text (both positive and negative) contributes to the text's magnitude (so longer text blocks may have greater magnitudes).

#### Sentiment Magnitude vs. AdoptionSpeed
![Description: Magnitude vs AdoptSpeed](../images/Desc_magnitude_x_AdoptSpeed.png)

#### Sentiment Score vs. AdoptionSpeed
![Description: Score vs AdoptSpeed](../images/Desc_score_x_AdoptSpeed.png)

#### Score * Magnitude vs. Adoption Speed
![Sentiment vs AdoptSpeed](../images/Sentiment_x_AdoptSpeed.png)

- The correlation table showed us a slight positive correlation between the sentiment 'score' and AdoptionSpeed, and 'sentiment' field which was computed by multiplying the magnitude with the score, we can see this correlation in the plots above.  A higher sentiment value means a more positive sentiment.  It is quite surprising that we have a positive correlation with the Adoption Speed, which means higher sentiment takes longer for Adoption.

## Model Selection:

The problem presented for this Petfinder Kaggle challenge, is a __multinomial classification__ challenge.  The following models were used and evaluated to determine which would perform best and give the highest kappa score:

<br>1) Hard Voting Classification with (RandomForestClassifier, ExtraTreesClassifier, SVC)
<br>2) Bagging with Random Forest
<br>3) Gradient Boosting


### Evaluation:

This specific competition is scored based on the __quadratic weighted kappa__, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, the metric may go below 0. The quadratic weighted kappa is calculated between the scores which are expected/known and the predicted scores.

Results have 5 possible ratings, 0,1,2,3,4.  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix O is constructed, such that Oi,j corresponds to the number of adoption records that have a rating of i (actual) and received a predicted rating j. An N-by-N matrix of weights, w, is calculated based on the difference between actual and predicted rating scores:

$$\begin{equation*}
w_{ij} = \frac{(i-j)^2}{(N-1)^2}
\end{equation*} $$


An N-by-N histogram matrix of expected ratings, E, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between the actual rating's histogram vector of ratings and the predicted rating's histogram vector of ratings, normalized such that E and O have the same sum.

From these three matrices, the quadratic weighted kappa is calculated as: 

$$\begin{equation*}
\kappa = 1 - \frac{\sum_(w_{ij} O_{ij})}{\sum_(w_{ij} E_{ij})}
\end{equation*} $$


*For model evaluation, both the Kappa Score, and Accuracy score were compared, but the Kappa Score was used to train the models, and for finding the best estimators.

### Hyper-parameter Tuning and Model Evaluation:

GridSearchCV was used to tune hyperparameters for the different models, and evaluated performance using Cross-validation.

### Hard Voting Classification (with RandomForestClassifier, ExtraTreesClassifier, SVC)
__accuracy_score__: 0.4155
<br>__kappa Score__: 0.3639
<br>__Kaggle Kappa Submission Score__: 0.231 (#1 Kaggle LB Score: 0.475)

![VC Class Report](../images/VC_classreport.png)

### Bagging Classifier with RandomForestClassifier
__accuracy_score__: 0.4211
<br>__kappa Score__: 0.3829
<br>__Kaggle Kappa Submission Score__: 0.270 (#1 Kaggle LB Score: 0.475)

![BC Class Report](../images/BC_classreport.png)

### Gradient Boosting Classifier
__accuracy_score__: 0.4175
<br>__kappa Score__: 0.3916
<br>__Kaggle Kappa Submission Score__: 0.257 (#1 Kaggle LB Score: 0.475)

![GB Class Report](../images/GB_classreport.png)


We can see that even if the Gradient Boosting classifier model gave us the highest Kappa Score during evaluation,
the Bagging Classifier gave us the higher score when ran against the unseen dataset from Kaggle.

If we look at the cross validation scores for the two models with cv=10, the Bagging Classifier gives us a higher cross-validation accuracy score.

#### Gradient Boosting CV Scores:
[0.29135275, 0.29784795, 0.32572931, 0.3627333,  0.32873088, 
<br>0.33027593, 0.34609186, 0.33352571, 0.33143808, 0.32882137]

- __Cross-Validation Accuracy Score:  0.32765__
 
#### Bagging Classifier CV Scores
[0.29127628, 0.3069401, 0.31245184, 0.36159175, 0.33222491, 
<br>0.35090539, 0.33442636, 0.37653168, 0.3406257, 0.31178901]
 
- __Cross-Validation Accuracy Score:  0.33187__

## In an attempt to increase the current score... 
the image metadata provided by Kaggle was processed to add features, re-tuned the Bagging Classifier and Gradient Boosting Classifier models, and gave us the following score -- it increased our previous score from 2.7 to 2.81! 

### Bagging Classifier with RandomForestClassifier
__accuracy_score__: 0.4355
<br>__kappa Score__: 0.4044
<br>__Kaggle Kappa Submission Score__: 0.281 (#1 Kaggle LB Score: 0.475)

![BC Class Report with meta](../images/BC_withmeta_classreport.png)

### Gradient Boosting Classifier
__accuracy_score__: 0.4199
<br>__kappa Score__: 0.3904
<br>__Kaggle Kappa Submission Score__: 0.222 (#1 Kaggle LB Score: 0.475)

![GB Class Report](../images/GB_withmeta_classreport.png)



## Summary:

### To answer the questions we had initially:

__-What feature/s influence the Adoption Speed of the pet?__
<br>__- How can the profile of the pet be improved, to help with the Adoptability of a pet?__

- __Health Profile:__ <br>Our findings showed that: Vaccinated, Dewormed, and Sterilized pets were not a concern to potential pet owners, however those with 'Unsure' value in these Health indicators show to have a longer adoption speed. This is something that Petfinder.my can look into to ensure that these Health indicators are checked and updated, to be populated with either a Yes or No, instead of an 'Unsure' value.<br>

- __Photo Amt:__ <br> We did see that pets who have more photos in their profile appear to be adopted quicker, than those with just one (1) photo in their profile.  Adding more pictures to showcase the pet could help increase their chances of getting adopted faster.

- __Description:__<br> The correlation table showed us a slight positive correlation between the sentiment 'score' and AdoptionSpeed, and 'sentiment' field which was computed by multiplying the magnitude with the score.  A higher sentiment value means a more positive sentiment.  Surprisingly, this shows a positive correlation with the Adoption Speed, which means higher sentiment takes longer for Adoption.  It would be hard to tell with the data provided how the description affects the AdoptionSpeed, for us to be able to recommend how to improve it.  This will require further text analysis to gain insight on the existing descriptions.


### Model Performance Evaluation and Further Improvement(s):

- Based on the Kappa Score we received after processing the unseen data with the models defined in this project, we can see that the Kappa Score is significantly lower than the results of the  Model evaluation.  This shows that the models are overfitting, further actions can be done to improve overfitting:

   - Further tune hyperparameters in the models to reduce variance - increase bias in the models.
   - Try to remove features which will not significantly impact the accuracy of the model


- Our models are still underperforming, accuracy and Kappa scores can be improved by:

  - Processing image files to add new features, and see how this will improve the accuracy and Kappa score.

  - Explore other algorithms such as XGBoost, or LGBM appears to be a popular model for this competition in Kaggle.


