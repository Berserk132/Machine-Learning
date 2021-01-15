import pandas as pd
import numpy as np

class info:
    def __init__(self,_feature1,_feature2,_acc,_w,_b):
        self._feature1=_feature1
        self._feature2=_feature2
        self._acc=_acc
        self._w=_w
        self._b=_b


def normalize(list, mean , rnge):
    newList=(list-mean)/rnge
    return newList;

def rnge(list):
    return np.max(list)-np.min(list)+1;


def hypothesis(w,features,b):
    h=np.dot(features,w)+b
    return h;



def gradiantDescent(alpha,w,features,b,y,itr,lam):
    featuresSize=features.shape[0]
    for i in range(itr):
        for j in range(0,featuresSize):
            cost = y[j]*hypothesis(w,features[j],b)
            if(cost>=1):
                w-=alpha * 2 * lam * w
                b -= alpha*2*lam * b
            else:
                w+= alpha*(np.dot(features[j],y[j])-2*lam*w)
                b += alpha*(y[j]-2*lam*b)
                #b-= alpha*y[j]

    return w,b



def acc(w,b,feature,y):
    counter=0
    featuresSize=feature.shape[0]
    for j in range(featuresSize):
        cost = y[j] * hypothesis(w, feature[j], b)
        if (cost >= 1):
           counter+=1
    return counter/featuresSize * 100



data = pd.read_csv("heart.csv")
data=data.sample(frac=1)
features= data.iloc[: , :13]
y = data["target"]
y=np.where(y<=0 , -1 ,1)

trainFeature=features.iloc[:202]
trainY=y[:202]

testFeature=features.iloc[202:]
testY=y[202:]

for i in range(1,14):
    features.iloc[:,i-1:i]= normalize(features.iloc[:,i-1:i],
                        np.mean(features.iloc[:,i-1:i]),rnge(features.iloc[:,i-1:i]))


names=["","age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

alphas=[3,1,0.1,0.01,0.001]
itr=1000
lam=1/itr
bestAcc=0
bestFeat1=[]
bestFeat2=[]
bestW=[]
bestB=0
bestAlpha=0
arrays=[]
bestW=[]
bestB=0
ans=0


for i in range(1,len(names)):
    feat1=trainFeature.iloc[:,i-1:i]
    for j in range (1,len(names)):
        if(j<=i):
            continue
        print(names[i] ," ", names[j])
        for alpha in alphas:
            w = np.zeros(2)
            b = 0
            feat2=trainFeature.iloc[:,j-1:j]
            newFeat=np.concatenate((feat1,feat2),axis=1)
            w,b=gradiantDescent(alpha,w,newFeat,b,y,itr,lam)
            ans = max(acc(w,b,newFeat,trainY),ans)
            accuracy = acc(w,b,newFeat,y)
            arrays.append(info(names[i],names[j],accuracy,w,b))
            if(accuracy> bestAcc):
                bestAcc=accuracy
                bestFeat1=i
                bestFeat2=j
                bestW=w
                bestB=b
                bestAlpha=alpha
            print("alpha = " , alpha ," acc = " ,accuracy)

arrays.sort(key=lambda x:x._acc ,reverse=True)
for i in arrays:
    print(i._feature1 , " " , i._feature2, " " ,i._acc)

print("best accuracy " ,bestAcc, " best features" , names[bestFeat1] ," and ",
      names[bestFeat2]," best w ",bestW ,"best b",bestB, "best alpha",bestAlpha)

newFeat = np.concatenate((testFeature.iloc[:,bestFeat1-1:bestFeat1],
                          testFeature.iloc[:,bestFeat2-1:bestFeat2]), axis=1)

print("ans = " , ans)

