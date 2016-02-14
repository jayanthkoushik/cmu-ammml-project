import sys
import csv
import shelve


def main():
	scoreMap = shelve.open("shelved_label_map.tmp")
	filepath = sys.argv[1]
	with open(filepath, 'rb') as csvfile:
		reader = csv.DictReader(csvfile, delimiter=',')
		for row in reader:
			key = row["Input.videoLink"]
			key = key.split("/")[-1].split(".")[0]
			score = row["Answer.q7_persuasive"]
			score = float(score.split("_")[0])
			if key in scoreMap:
				scoreMap[key] += score
			else:
				scoreMap[key] = score
	for key in scoreMap:
		scoreMap[key] = scoreMap[key]/3
	scoreMap.close()
	

if __name__ == "__main__" : main()