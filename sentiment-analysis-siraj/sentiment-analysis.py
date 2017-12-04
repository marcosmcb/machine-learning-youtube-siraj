import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import nltk


if __name__ == '__main__':
    #[1] Read the data
    train = pd.read_csv( os.path.join( os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test  = pd.read_csv( os.path.join( os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )

    print ('The first review is ')     
    print (train['review'][0])

    input( "Press Enter to continue ..." )

    #[2] Clean the training data
    print ('Download text data sets')
    nltk.download()

    clean_train_reviews = []
    print('Cleaning and parsing the training set movie reviews...\n')
    for i in range( 0, len( train['review'] ) ):
        clean_train_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True) ) )
    
    #[3] Create the bag of words
    print ('Creating the bag of words...\n')
    vectorizer = CountVectorizer( analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=5000 )
    train_data_features = vectorizer.fit_transform( clean_train_reviews )
    train_data_features = train_data_features.toarray()

    #[4] Train the classifier
    print( "Training the random forest (this may take a while)..." )
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit( train_data_features, train["sentiment"] )
    clean_train_reviews = []

    #[5] Format the testing data
    print("Cleaning and parsing the test set movie reviews... \n")
    for i in range( 0, len(test["review"]) ):
        clean_train_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True) ) )
    
    test_data_features = vectorizer.transform(clean_train_reviews)
    test_data_features = test_data_features.toarray()

    #[6] Predict reviews in the testing data
    print("Predicting test labels...\n")
    result = forest.predict( test_data_features )
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( os.path.join( os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv' ), index=False, quoting=3 )

    print("Wrote results to Bag_of_Words_model.csv")





