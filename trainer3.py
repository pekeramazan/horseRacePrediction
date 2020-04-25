import csv
import pickle
import logging
import numpy as np
from sklearn.svm import SVR

###
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model(object):

    def __init__(self):
        pass

    def _get_data(self, filename):

        training_data = csv.reader(open('data/%s' % filename, 'r',errors='ignore'))

        logging.info('Training Finish Position')

        y = []  # Target to train on
        X = []  # Features

        for i, row in enumerate(training_data):
            # Skip the first row since it's the headers
            if i == 0:
                continue

            # Get the target
            y.append(float(row[3]))

            # Get the features
            data = np.array(
                [float(_ if len(str(_)) > 0 else 0) for _ in row[4:]]
            )
            X.append(data)

        return X, y

    def train(self):

        clf = SVR(C=1.0, epsilon=0.1, cache_size=1000)
        X, y, = self._get_data('dataset.csv')

        # Fit the model
        clf.fit(X, y)

        # Pickle the model so we can save and reuse it
        s = pickle.dumps(clf)
        logging.info('Training Complete')

        # Save the model to a file
       # f = open('finish_pos.model', 'wb')
        f = open('output.model', 'wb')

        f.write(s)
        f.close()

    def predict(self):
        f = open('output.model', 'rb')
        clf = pickle.loads(f.read())
        f.close()

        #validation_file = csv.reader(open('data/validation-dataset.csv', 'r',errors='ignore')
        validation_data = csv.DictReader(open('data/validation-dataset.csv', 'r',errors='ignore')

        )

        races = {}
        for row in validation_data:
            race_id = row["EntryID"]
            finish_pos = float(row["Placement"])

            if race_id not in races:
                races[race_id] = []

            if finish_pos < 1:
                continue

            data = np.array([
                float(valid if len(str(valid)) > 0 else 0)
                for valid in list(row.values())[4:]
            ])
            data = data.reshape(1, -1)
            races[race_id].append(
                {
                    "data":data,
                    "prediction":None,
                    "finish_pos":finish_pos
                }
            )


        
        num_races = 0
        num_correct_pred_win = 0
        num_correct_pred_wps = 0
        for race_id, horses in races.items():
            for horse in horses:
                horse['prediction'] = clf.predict(horse['data'])
            horses.sort(key=lambda h: h["prediction"])
            num_races += 1
        # If the horse that won the race is in first place, then the
        # model is correct, add a point
            if horses[0]['finish_pos'] == 1:
                num_correct_pred_win += 1
        # If the model predicted the horse in first three, then it's
        # also somewhat good
            if horses[0]['finish_pos'] <= 3:
                num_correct_pred_wps += 1

        print(('Total of predicted races :%s\nTotal of correct win prediction :%s\nTotal count of correct first three places predictions :%s' % (
            num_races,
            num_correct_pred_win,
            num_correct_pred_wps
        )))


if __name__ == '__main__':

    trn = Model()
    trn.train()
    trn.predict()
