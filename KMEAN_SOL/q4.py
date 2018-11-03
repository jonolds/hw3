import numpy as np

def findTop5(Tau, showNameList, name):
	showList = list(Tau[499, 0:100])
	showList = [(i, showList[i]) for i in xrange(len(showList))]
	showList.sort(key = lambda x : (-x[1], x[0]))
	f = open("res" + name + ".txt", "w")
	for i in xrange(5):
		show = showList[i]
		f.write("{}\t{}\t{}\n".format(show[0], showNameList[show[0]], show[1]))
	f.close()

if __name__ == "__main__":
	# read shows
	shows = []
	with open("shows.txt") as f1:
		for line in f1:
			shows.append(line.strip("\n"))                        # len(shows): 563

	# read R
	RList = []
	with open("user-shows.txt", "r") as fin:
		for line in fin:
			data = line.split(" ");
			data = [float(p) for p in data]
			RList.append(data)
	R = np.array(RList)                                           # R.shape  (9985, 563)

	P = np.sum(R, axis=1)
	PT = P.reshape((P.shape[0], 1))                               # P.shape: (9985, 1)
	Q = np.sum(R, axis=0)
	QT = Q.reshape((Q.shape[0], 1))                               # Q.shape: (563, 1)

	# item-item R * (QStar * R^T * R QStar)
	QStar = np.power(Q, -0.5)                                     # Q.shape: (563, 1)
	QStarT = np.power(QT, -0.5)
	SI = np.multiply(np.multiply(QStar, np.dot(R.T, R)), QStarT)  # SI.shape:  (563, 563)
	Tau_I = np.dot(R, SI)                                         # Tau_I.shape:  (9985, 563)
	findTop5(Tau_I, shows, "Movie")

	# user-user (PStar * R  * R^T * PStar) * R
	PStar = np.power(P, -0.5)
	PStarT = np.power(PT, -0.5)
	SU = np.multiply(np.multiply(PStar, np.dot(R, R.T)), PStarT)  # SU.shape:  (9985, 9985)
	Tau_U = np.dot(SU, R)                                         # Tau_U.shape:  (9985, 563)
	findTop5(Tau_U, shows, "User")