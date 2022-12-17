import json

data = json.load(open("evjvqa_private_test.json"))
converted_data = {}
for annotation in data["annotations"]:
    converted_data[annotation["id"]] = annotation["answer"]

json.dump(converted_data, open("ground_truth.json", "w+"), ensure_ascii=False)