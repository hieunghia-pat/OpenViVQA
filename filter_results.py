import json
import os

new2old = json.load(open("mapping_new2old.json"))
new2old = {key: str(value) for key, value in new2old.items()}

predictions = json.load(open(os.path.join("private_test", "private_test.json")))
filter_predictions = {}
for id, answer in predictions.items():
    if new2old[id] != "-1":
        filter_predictions[new2old[id]] = answer

json.dump(filter_predictions, open(os.path.join("private_test", "results.json"), "w+"), ensure_ascii=False)
