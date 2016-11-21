__author__ = 'Reede'

import random
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import numpy as np


class Gauss(object):
	def __init__(self, mean, size):
		self.mean = mean
		self.var = 0.1
		self.pointMemberships = []
		for i in range(size):
			self.pointMemberships.append(0)


class DataPoint(object):
	def __init__(self, k, val):
		self.val = val
		self.gaussMemberships = []
		for i in range(k):
			self.gaussMemberships.append(0)


def	graphAll(data, gaussians):
	points = [d.val for d in data]
	plt.axis([min(points)-.5, max(points)+.5, -1, 1])
	for p in points:
		plt.plot(p, 0, 'b*')
	for g in gaussians:
		plt.plot(g.mean, 0, 'rd')
	plt.show()

def diff(a,b):
	return a-b


def emAlgo(data, k):
	l = len(data)
	maximum = 0
	minimum = 10000
	for d in data:
		if d.val > maximum: maximum = d.val
		if d.val < minimum: minimum = d.val
	gaussians = []
	for i in range(k):
		c = random.uniform(minimum, maximum)
		gaussians.append(Gauss(c, l))
	notDone = True
	c = 0
	while(notDone):
		c+=1
		graphAll(data, gaussians)
		e(data, gaussians)
		old_means, new_means = m(data, k, gaussians)
		for k in range(len(old_means)):
			if abs(diff(old_means[k],new_means[k])) <= 0.001:
				notDone = False
	print("iterations till converge: ", c)
	print()
	c2 =1
	for g3 in gaussians:
		print("gauss #", c2 )
		print("\tgauss final var: ", g3.var)
		print("\tgauss final mean: ", g3.mean)
		print()
		c2 +=1


def e(data, gaussians):
	l = len(data)
	for gauss in gaussians:
		m = gauss.mean
		v = gauss.var
		count = 0
		for d in data: # calculate probability of membership (gauss to points)
			exponent = -(math.pow((d.val - m), 2) / (2 * v * v) + 0.00001)
			membershipProbability = math.pow((1 / math.sqrt(2 * math.pi * v * v)), exponent)
			gauss.pointMemberships[count] = membershipProbability
			count += 1
		gauss.w = sum(gauss.pointMemberships)/l
	# second pass: soft cluster (points to gauss)
	nums = [_ for _ in range(len(gaussians))]
	for i in nums:
		c2 = 0
		for d2 in data:
			numerator = gaussians[i].pointMemberships[c2] * gaussians[i].w
			nums2 = nums
			denominator = numerator
			for j in nums2:
				if j != i:
					denominator += gaussians[j].pointMemberships[c2] * gaussians[j].w
			if denominator != 0:
				d2.gaussMemberships[i] = numerator/denominator
			else:
				d2.gaussMemberships[i] = 0.0
			c2 += 1

def mean(data, list_probs):
	d = [d.val for d in data]
	mult = [a * b for a, b in zip(list_probs, d)]
	s = sum(list_probs)
	if (s) == 0:
		s+=.001
	return sum(mult) / s

def var(data, mean, list_probs):
	d = [i.val for i in data]
	temp = [b * math.pow((a - mean), 2)  for a, b in zip(d, list_probs)]
	return sum(temp) / sum(list_probs)

def m(data, k, gaussians):
	olds = []
	news = []
	for g in gaussians:
		olds.append(g.mean)
		new_mean = mean(data, g.pointMemberships)
		news.append(new_mean)
		g.mean = new_mean
		new_var = var(data, new_mean, g.pointMemberships)
		# print("OLD VAR: ", g.var)
		# print("NEW VAR: ", new_var)
		g.var = new_var
	return olds, news
	# count = 0
	# old_means = []
	# new_means = []
	# for g in gaussians:
	# 	old_means.append(g.mean)
	# 	new_mean = 0
	# 	for x in data:
	# 		new_mean += x.val * x.gaussMemberships[count]
	# 		new_moean = mean(data, )
	# 	new_mean = new_mean / sum(x.gaussMemberships)
	# 	new_means.append(new_mean)
	# 	g.mean = new_mean
	# 	count += 1
	# return 	old_means, new_means



k = 3
fn = 'emData.txt'
data = [DataPoint(k, float(object)) for object in open(fn).read().splitlines()]
emAlgo(data, 3)


