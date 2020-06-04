# MyAdaBoost vs Scikit's AdaBoost - comparison

---
## Dataset 'german' 
### Fold no. 0
**Classifier: ScikitAdaBoost**

	Stump no. 0:
		feature:	0
		type:		numerical
		threshold:	2.5
		Amount of Say:	0.2
	Stump no. 1:
		feature:	1
		type:		numerical
		threshold:	34.5
		Amount of Say:	0.4
	Stump no. 2:
		feature:	5
		type:		numerical
		threshold:	2.5
		Amount of Say:	0.2
	Stump no. 3:
		feature:	2
		type:		numerical
		threshold:	3.5
		Amount of Say:	0.2
	Stump no. 4:
		feature:	1
		type:		numerical
		threshold:	8.5
		Amount of Say:	0.4
**Classifier: MyAdaBoost**

	Stump no. 0:
		feature:	0
		type:		categorical
		pool:	(1, 2)
		Amount of Say:	-0.255
	Stump no. 1:
		feature:	1
		type:		numerical
		threshold:	34.5
		Amount of Say:	-0.451
	Stump no. 2:
		feature:	0
		type:		categorical
		pool:	(1, 3)
		Amount of Say:	-0.26
	Stump no. 3:
		feature:	3
		type:		categorical
		pool:	(1, 3, 8)
		Amount of Say:	0.11
	Stump no. 4:
		feature:	2
		type:		categorical
		pool:	(4,)
		Amount of Say:	0.045
### Fold no. 1
**Classifier: ScikitAdaBoost**

	Stump no. 0:
		feature:	0
		type:		numerical
		threshold:	2.5
		Amount of Say:	0.2
	Stump no. 1:
		feature:	1
		type:		numerical
		threshold:	15.5
		Amount of Say:	0.2
	Stump no. 2:
		feature:	2
		type:		numerical
		threshold:	3.5
		Amount of Say:	0.4
	Stump no. 3:
		feature:	2
		type:		numerical
		threshold:	1.5
		Amount of Say:	0.4
	Stump no. 4:
		feature:	3
		type:		numerical
		threshold:	0.5
		Amount of Say:	0.2
**Classifier: MyAdaBoost**

	Stump no. 0:
		feature:	0
		type:		categorical
		pool:	(1, 2)
		Amount of Say:	-0.299
	Stump no. 1:
		feature:	2
		type:		categorical
		pool:	(0, 1)
		Amount of Say:	-0.528
	Stump no. 2:
		feature:	1
		type:		numerical
		threshold:	15.0
		Amount of Say:	-0.182
	Stump no. 3:
		feature:	2
		type:		categorical
		pool:	(2, 3)
		Amount of Say:	-0.153
	Stump no. 4:
		feature:	3
		type:		categorical
		pool:	(0, 5, 6)
		Amount of Say:	-0.239
### Fold no. 2
**Classifier: ScikitAdaBoost**

	Stump no. 0:
		feature:	0
		type:		numerical
		threshold:	2.5
		Amount of Say:	0.2
	Stump no. 1:
		feature:	1
		type:		numerical
		threshold:	15.5
		Amount of Say:	0.2
	Stump no. 2:
		feature:	5
		type:		numerical
		threshold:	3.5
		Amount of Say:	0.2
	Stump no. 3:
		feature:	2
		type:		numerical
		threshold:	3.5
		Amount of Say:	0.2
	Stump no. 4:
		feature:	3
		type:		numerical
		threshold:	0.5
		Amount of Say:	0.2
**Classifier: MyAdaBoost**

	Stump no. 0:
		feature:	0
		type:		categorical
		pool:	(1, 2)
		Amount of Say:	-0.271
	Stump no. 1:
		feature:	1
		type:		numerical
		threshold:	15.0
		Amount of Say:	-0.119
	Stump no. 2:
		feature:	3
		type:		categorical
		pool:	(1, 3, 8, 10)
		Amount of Say:	0.057
	Stump no. 3:
		feature:	2
		type:		categorical
		pool:	(0, 1)
		Amount of Say:	-0.524
	Stump no. 4:
		feature:	2
		type:		categorical
		pool:	(2, 3)
		Amount of Say:	-0.163
### Fold no. 3
**Classifier: ScikitAdaBoost**

	Stump no. 0:
		feature:	0
		type:		numerical
		threshold:	2.5
		Amount of Say:	0.2
	Stump no. 1:
		feature:	1
		type:		numerical
		threshold:	31.5
		Amount of Say:	0.4
	Stump no. 2:
		feature:	2
		type:		numerical
		threshold:	1.5
		Amount of Say:	0.2
	Stump no. 3:
		feature:	1
		type:		numerical
		threshold:	8.5
		Amount of Say:	0.4
	Stump no. 4:
		feature:	12
		type:		numerical
		threshold:	30.5
		Amount of Say:	0.2
**Classifier: MyAdaBoost**

	Stump no. 0:
		feature:	0
		type:		categorical
		pool:	(1, 2)
		Amount of Say:	-0.29
	Stump no. 1:
		feature:	2
		type:		categorical
		pool:	(0, 1)
		Amount of Say:	-0.514
	Stump no. 2:
		feature:	1
		type:		numerical
		threshold:	31.5
		Amount of Say:	-0.206
	Stump no. 3:
		feature:	3
		type:		categorical
		pool:	(0, 4, 6)
		Amount of Say:	-0.147
	Stump no. 4:
		feature:	1
		type:		numerical
		threshold:	8.0
		Amount of Say:	-0.167
### Fold no. 4
**Classifier: ScikitAdaBoost**

	Stump no. 0:
		feature:	0
		type:		numerical
		threshold:	2.5
		Amount of Say:	0.2
	Stump no. 1:
		feature:	1
		type:		numerical
		threshold:	15.5
		Amount of Say:	0.2
	Stump no. 2:
		feature:	2
		type:		numerical
		threshold:	3.5
		Amount of Say:	0.2
	Stump no. 3:
		feature:	5
		type:		numerical
		threshold:	2.5
		Amount of Say:	0.2
	Stump no. 4:
		feature:	9
		type:		numerical
		threshold:	2.5
		Amount of Say:	0.2
**Classifier: MyAdaBoost**

	Stump no. 0:
		feature:	0
		type:		categorical
		pool:	(1, 2)
		Amount of Say:	-0.29
	Stump no. 1:
		feature:	1
		type:		numerical
		threshold:	24.0
		Amount of Say:	-0.402
	Stump no. 2:
		feature:	3
		type:		categorical
		pool:	(1, 3, 4, 8)
		Amount of Say:	0.099
	Stump no. 3:
		feature:	2
		type:		categorical
		pool:	(0, 1, 2, 3)
		Amount of Say:	-0.011
	Stump no. 4:
		feature:	2
		type:		categorical
		pool:	(4,)
		Amount of Say:	0.0
### ScikitAdaBoost
accuracy results : [0.755, 0.705, 0.775, 0.71, 0.73]
accuracy (average): 0.735
accuracy (std deviation): 0.027
### MyAdaBoost
accuracy results : [0.69, 0.705, 0.72, 0.715, 0.64]
accuracy (average): 0.694
accuracy (std deviation): 0.029
### Statistical comparison
alpha = 0.05
T = 2.1962534944228795
p=0.09304057014251862
No statistically important difference between classifiers (p > alpha).
