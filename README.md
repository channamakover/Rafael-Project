# Rafael-Project
Project goal:
Classifying rockets to different types using data science algorithms 
and research various types of machine learning methods.

we got a CSV file with data about 16 types of rokets. the data contains a 30 seconds of radar record of about 30000 rockets.
First we had to build a histogram that would visualise the velocity of each type of rocket.
then we start with destinguishing between set of pairs of rockets, for each pair we devide its data to:'training set' and 'test set'.
We started analyzing the data of the 'trainig set' so we would be able to build a model that will learn the featurs of each rocket and than, run on the test set and classify each rocket according to its individual features.
We saw when plotting the record of the radar for those specific types, we couldnt point out enough features so we would be able to run through the test set and get more then 95% of success. So we added to the table the sum of potential energy ans kinetic energy in every point, then we plotted it again and we could tell perfecetly what are the features of each rocket.
Then, we run the confusion matrix and we got 100% of success.
Other phase was to take a 4 types and do the same with ML libraries. We started with fitting with the train set with LogisticRegression, then, we did the same with RandomForestClassifier, and when running the test set, we saw that RandomForestClassifier gave higher score.
We then change the params of RandomForestClassifier and figure out that from a some number of trees the score of training raises but the score of the set decreses-what mean 'over fitting'
