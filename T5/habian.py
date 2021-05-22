
X=[[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,1]]
Y=[-1,-1,-1,1] 

def NetValue(W):
    FinalW1,FinalW2,FinalBias=W
    Net=[]
    for i in X:
        NetV=FinalW1*i[0]+FinalW2*i[1]+FinalBias#formula used to calculate net value wrt imputs x1,x2
        Net.append(NetV)
        
    return Net


def ANDwithHebb_LR():
    weight1=weight2=b=0#initial weights and bias
    WeightBiasMat=[]#matrix to hold updated weights wrt inputs
    count=0
    while count<4:#for learning and updating weights
        for i in X:
            temp=[]
            weight1+=i[0] * Y[count]#formula to update weights Weight(new)=Weight(old)+x*y
            weight2+=i[0]*Y[count]
            
            b+=1 *Y[count]#formula to update bias; bias(new)=bias(old)+y
            
            temp.append(weight1)
            temp.append(weight2)
            temp.append(b)
            
            WeightBiasMat.append(temp) 
            
            count+=1
    print("Weight Matrix>>>",WeightBiasMat) 
    print()
    ResultFromNet=NetValue(WeightBiasMat[-1]) 
    Prediction=[] 
    print("NetWeightMatrix",ResultFromNet)
    print()
    for i in ResultFromNet:
        if i>0:
            Prediction.append(1)
        else:
            Prediction.append(-1)
    print("Required >>>{}\nPredicted From Hebbian Learning Rule >>>{}".format(Y,Prediction))    
    
ANDwithHebb_LR()    
