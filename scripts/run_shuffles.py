import random, os

import numpy as np

#experiment_dir = "experiments/Z3"

#command = "python -m segmenter.probabilistic -o {}/segmented.txt -v -n 0 -m venk {}/prepared2.txt".format(experiment_dir, experiment_dir)

languages = ["Basque", "Cantonese", "Croatian", "Danish", "Dutch", "English", "Estonian", "Farsi", "French", "German", "Greek", "Hungarian", "Icelandic", "Indonesian", "Irish", "Italian", "Japanese", "Korean", "Mandarin", "Norwegian", "Portuguese", "Romanian", "Serbian", "Spanish", "Swedish", "Turkish"]

#for experiment_dir in ["experiments/L-{}-M".format(language) for language in languages[20:]]:
#for experiment_dir in ["experiments/DM14A0"]:

for experiment_dir in ["experiments/E"]:

	stress = "{}/stress.txt".format(experiment_dir)
	gold = "{}/gold.txt".format(experiment_dir)
	prepared = "{}/prepared.txt".format(experiment_dir)
	n = 10
	data = {}

	for i in range(1, n+1):

		stress2 = "{}/stress{}.txt".format(experiment_dir, i)
		prepared2 = "{}/prepared{}.txt".format(experiment_dir, i)
		gold2 = "{}/gold{}.txt".format(experiment_dir, i)

		#command = "python -m segmenter.dynamicmulticue -o {}/segmented{}.txt -n 3,1 -d both -P ent,mi,bp -L both -X {}/stress{}.txt -a 0 {}/prepared{}.txt".format(experiment_dir, i, experiment_dir, i, experiment_dir, i)
		#command = "python -m segmenter.dynamicmulticue -o {}/segmented{}.txt -v -n 4,3,2,1 -d both -P sv -a 0 {}/prepared{}.txt".format(experiment_dir, i, experiment_dir, i)
		command = "python -m segmenter.dynamicmulticue -o {}/segmented{}.txt -v -n 4,3,2,1 -d both -P sv,bp -L both -a 0 {}/prepared{}.txt".format(experiment_dir, i, experiment_dir, i)
		#command = "python -m segmenter.probabilistic -o {}/segmented{}.txt -v -n 0 -m blanch {}/prepared{}.txt".format(experiment_dir, i, experiment_dir, i)

		evall = "{}/eval{}.txt".format(experiment_dir, i)
		evalcommand = "wordseg-eval -r {}/prepared{}.txt {}/segmented{}.txt {}/gold{}.txt > {}/eval{}.txt".format(experiment_dir, i, experiment_dir, i, experiment_dir, i, experiment_dir, i)

		random.seed(i)
		goldlines = open(gold, 'r').readlines()
		preparedlines = open(prepared, 'r').readlines()
		#stresslines = open(stress, 'r').readlines()
		if i != 0:
			temp = list(zip(goldlines, preparedlines))
			random.shuffle(temp)
			goldlines, preparedlines = zip(*temp)
		print("running with seed: {}".format(i))
		with open(gold2, 'w') as gold2file:
			gold2file.writelines(goldlines)
		with open(prepared2, 'w') as prepared2file:
			prepared2file.writelines(preparedlines)
		#with open(stress2, 'w') as stress2file:
		#	stress2file.writelines(stresslines)
		os.system(command)
		os.system(evalcommand)
		os.system("python -m segmenter.evaluation {}/segmented{}.txt {}/gold{}.txt {}/prepared{}.txt >> {}/eval{}.txt".format(experiment_dir, i, experiment_dir, i, experiment_dir, i, experiment_dir, i))
		evallines = open(evall, 'r').readlines()
		for line in evallines:
			[key, num] = line.strip().split('\t')
			if not key in data:
				data[key] = [float(num)]
			else:
				data[key].append(float(num)) 

	for key in data:
		print("{} average: {}".format(key, round(100*np.mean(data[key]),1)))
		print("{} stddev: {}".format(key, round(100*np.std(data[key]),3)))
	printkeys = ["boundary_noedge_precision", "boundary_noedge_recall", "boundary_noedge_fscore", "token_precision", "token_recall", "token_fscore", "type_precision", "type_recall", "type_fscore", "undersegmentation", "oversegmentation"]
	print(' & '.join([str(round(100*np.mean(data[printkey]), 1)) for printkey in printkeys]))
	print(data)
