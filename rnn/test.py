from rnn import *
# this will test the implementation of predict, acc_deltas, and acc_deltas_bptt in rnn.py, for a simple 3x2 RNN

y_exp = array([[ 0.39411072,  0.32179748,  0.2840918 ], [ 0.4075143,   0.32013043,  0.27235527], [ 0.41091755,  0.31606385,  0.2730186 ], [ 0.41098376,  0.31825833,  0.27075792]])
s_exp = array([[ 0.66818777,  0.64565631], [ 0.80500806,  0.80655686], [ 0.85442692,  0.79322425], [ 0.84599959,  0.8270955 ], [ 0., 0.        ]])
U_exp = array([[ 0.89990596,  0.79983619], [ 0.5000714,   0.30009787]])
V_exp = array([[ 0.69787081,  0.30129314,  0.39888647], [ 0.60201076,  0.89866058,  0.70149262]])
W_exp = array([[ 0.57779081,  0.47890397], [ 0.22552931,  0.62294835], [ 0.39667988 , 0.19814768]])

deltaU_1_exp = array([[-0.00355893, -0.00498617], [-0.00021364,  0.00027208]])
deltaV_1_exp = array([[-0.04754913,  0.02089739, -0.02723592], [ 0.03501492, -0.0319887,   0.024652  ]])
deltaW_1_exp = array([[-0.44418377, -0.42192061], [ 0.5105861,   0.45896705], [-0.06640233,-0.03704644]])

deltaU_3_exp = array([[-0.00188081, -0.0032761 ], [ 0.0014281,   0.00195745]])
deltaV_3_exp = array([[-0.05285491,  0.01969119, -0.02415013], [ 0.03149499, -0.03368772,  0.02810167]])
deltaW_3_exp = array([[-0.44418377, -0.42192061], [ 0.5105861,   0.45896705], [-0.06640233, -0.03704644]])

vocabsize = 3
hdim = 2
# RNN with vocab size 3 and 2 hidden layers
r = RNN(vocabsize,hdim)
r.V[0][0]=0.7
r.V[0][1]=0.3
r.V[0][2]=0.4
r.V[1][0]=0.6
r.V[1][1]=0.9
r.V[1][2]=0.7

r.W[0][0]=0.6
r.W[0][1]=0.5
r.W[1][0]=0.2
r.W[1][1]=0.6
r.W[2][0]=0.4
r.W[2][1]=0.2

r.U[0][0]=0.9
r.U[0][1]=0.8
r.U[1][0]=0.5
r.U[1][1]=0.3

x = array([0,1,2,1])
d = [1,2,1,0]

print("### predicting y")
y,s = r.predict(x)
# print("y expected\n{0}".format(y_exp))
# print("y received\n{0}".format(y))
# print("\ns expected\n{0}".format(s_exp))
# print("s received\n{0}".format(s))
#
# print("\n### standard BP")
r.acc_deltas(x,d,y,s)
# print("deltaU expected\n{0}".format(deltaU_1_exp))
# print("deltaU received\n{0}".format(r.deltaU))
# print("\ndeltaV expected\n{0}".format(deltaV_1_exp))
# print("deltaV received\n{0}".format(r.deltaV))
# print("\ndeltaW expected\n{0}".format(deltaW_1_exp))
# print("deltaW received\n{0}".format(r.deltaW))
#
#
print("\n### BPTT with 3 steps")
r.deltaU.fill(0)
r.deltaV.fill(0)
r.deltaW.fill(0)

r.acc_deltas_bptt(x,d,y,s,3)
print("deltaU expected\n{0}".format(deltaU_3_exp))
print("deltaU received\n{0}".format(r.deltaU))
print("\ndeltaV expected\n{0}".format(deltaV_3_exp))
print("deltaV received\n{0}".format(r.deltaV))
print("\ndeltaW expected\n{0}".format(deltaW_3_exp))
print("deltaW received\n{0}".format(r.deltaW))
